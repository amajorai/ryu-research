"""Ryu content sealing for a Python satellite (vendored; phase-2 at-rest design).

Wire-compatible with the Rust (``ryu-crypto``) and TypeScript
(``@ryu/sidecar-runtime``) implementations and pinned by the shared vectors
(``tests/fixtures/content-v1.json``):

- ``encb:v1:`` text and ``RYU-CB01`` bytes: ChaCha20-Poly1305 under the store's
  content key ``HKDF(root, "ryu-content-crypto-v1", "ryu-content-crypto-v1:" + store)``
  with the length-prefixed AAD ``("ryu-content-seal-v1", "store", store,
  "purpose", purpose, (label, value)*)``;
- ``bi1:`` blind tokens and full-text term tokens (HMAC-SHA256 under
  ``HKDF(root, "ryu-blind-index-v1", ...)``) and the shared tokenizer;
- the key check value ``HMAC(root, "ryu-kcv-v1")[:16]``.

Key custody (design section 11). At import this module REMOVES ``RYU_SEAL_KEY``,
``RYU_SEAL_SCOPE``, ``RYU_SEAL_RETIRED_KEYS`` and ``RYU_MASTER_KEY`` from
``os.environ`` so no child process inherits them, and resolves the process
keys lazily on first use:

1. ``RYU_SEAL_KEY`` delivered by Core (its scope must be ``plugin:<id>``);
2. ``RYU_MASTER_KEY`` -> this plugin's subkey (the key Core would deliver);
3. run by Core (``RYU_PLUGIN_DATA_DIR`` set) with no key -> compat mode;
4. standalone -> ``<data dir>/secrets/standalone-root.key`` (0600), generated
   on first use; its KCV is pinned and a later different key is refused.

The AEAD uses the ``cryptography`` package when it is installed and otherwise
a small pure-Python ChaCha20-Poly1305 (RFC 8439) that passes the same vectors,
so the stdlib-only engine keeps starting on a bare Python.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import shutil
import sqlite3
import struct
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

try:  # pragma: no cover - exercised implicitly when installed
    from cryptography.hazmat.primitives.ciphers.aead import (
        ChaCha20Poly1305 as _LibChaCha,
    )
    from cryptography.exceptions import InvalidTag as _LibInvalidTag
except ImportError:  # pragma: no cover - the bare-Python path
    _LibChaCha = None
    _LibInvalidTag = None

TEXT_PREFIX = "encb:v1:"
BYTES_MAGIC = b"RYU-CB01"
RYU_MAGICS = (b"RYU-CB01", b"RYU-SF01", b"RYU-SL01")
KEY_LENGTH = 32
NONCE_LENGTH = 12
TAG_LENGTH = 16

SEAL_KEY_ENV = "RYU_SEAL_KEY"
SEAL_SCOPE_ENV = "RYU_SEAL_SCOPE"
SEAL_RETIRED_KEYS_ENV = "RYU_SEAL_RETIRED_KEYS"
MASTER_KEY_ENV = "RYU_MASTER_KEY"
SEAL_ENV_VARS = (SEAL_KEY_ENV, SEAL_SCOPE_ENV, SEAL_RETIRED_KEYS_ENV, MASTER_KEY_ENV)
PLUGIN_DATA_DIR_ENV = "RYU_PLUGIN_DATA_DIR"
STANDALONE_ROOT_FILE = "standalone-root.key"
STANDALONE_KCV_FILE = "standalone-root.kcv.json"
COMPAT_HEALTH = "waiting for a Ryu update"
SEAL_MIN_READER = 2

_CONTENT_SALT = b"ryu-content-crypto-v1"
_CONTENT_INFO = b"ryu-content-crypto-v1:"
_SUB_SALT = b"ryu-content-sub-v1"
_SUB_INFO = b"ryu-content-sub-v1:"
_BLIND_SALT = b"ryu-blind-index-v1"
_BLIND_INFO = b"ryu-blind-index-v1:"
_PLUGIN_SALT = b"ryu-plugin-crypto-v1"
_PLUGIN_INFO = b"ryu-plugin-crypto-v1:"
_AAD_LABEL = b"ryu-content-seal-v1"
_KCV_LABEL = b"ryu-kcv-v1"
_VERSIONED_TEXT = re.compile(r"^encb:v\d+:")
_RYU_MAGIC = re.compile(rb"^RYU-[A-Z]{2}\d{2}$")

STOPWORDS = frozenset(
    "a an and are as at be but by for from has have he her his i if in into is it"
    " its me my no not of on or our she so that the their them there they this to"
    " was we were what when which who will with you your".split()
)


class SealError(Exception):
    """A key could not be resolved, or the store refuses to seal."""


class NewerStoreFormatError(SealError):
    """A newer build sealed the store under a reader format this build lacks."""


class SealUnavailableError(SealError):
    """Writes are refused: no usable key for a store that holds sealed data."""


class ContentSealError(Exception):
    """A value does not open (``code`` names why). Never carries key material."""

    def __init__(self, code: str) -> None:
        super().__init__(f"sealed content does not open ({code})")
        self.code = code


# ── primitives ──────────────────────────────────────────────────────────────


def hkdf32(ikm: bytes, salt: bytes, info: bytes) -> bytes:
    """HKDF-SHA256 with a 32-byte output (RFC 5869)."""
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    return hmac.new(prk, info + b"\x01", hashlib.sha256).digest()


def kcv(root: bytes) -> str:
    """Key check value of a root key."""
    return hmac.new(root, _KCV_LABEL, hashlib.sha256).digest()[:16].hex()


def derive_plugin_key(master: bytes, plugin_id: str) -> bytes:
    """The plugin subkey Core derives from the master key."""
    return hkdf32(master, _PLUGIN_SALT, _PLUGIN_INFO + plugin_id.encode())


def _length_prefixed(parts: Iterable[bytes]) -> bytes:
    return b"".join(struct.pack(">Q", len(part)) + part for part in parts)


def _chacha20_keystream(key: bytes, counter: int, nonce: bytes, length: int) -> bytes:
    """ChaCha20 (RFC 8439) keystream of ``length`` bytes from block ``counter``.

    Unrolled with local variables: the pure-Python fallback is several times
    faster this way than with a per-call quarter-round helper.
    """
    M = 0xFFFFFFFF  # noqa: N806
    k0, k1, k2, k3, k4, k5, k6, k7 = struct.unpack("<8L", key)
    n0, n1, n2 = struct.unpack("<3L", nonce)
    pack = struct.Struct("<16L").pack
    blocks = []
    for index in range((length + 63) // 64):
        c = (counter + index) & M
        x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15 = (
            0x61707865, 0x3320646E, 0x79622D32, 0x6B206574,
            k0, k1, k2, k3, k4, k5, k6, k7, c, n0, n1, n2,
        )
        for _ in range(10):
            x0 = (x0 + x4) & M; t = x12 ^ x0; x12 = ((t << 16) & M) | (t >> 16)
            x8 = (x8 + x12) & M; t = x4 ^ x8; x4 = ((t << 12) & M) | (t >> 20)
            x0 = (x0 + x4) & M; t = x12 ^ x0; x12 = ((t << 8) & M) | (t >> 24)
            x8 = (x8 + x12) & M; t = x4 ^ x8; x4 = ((t << 7) & M) | (t >> 25)
            x1 = (x1 + x5) & M; t = x13 ^ x1; x13 = ((t << 16) & M) | (t >> 16)
            x9 = (x9 + x13) & M; t = x5 ^ x9; x5 = ((t << 12) & M) | (t >> 20)
            x1 = (x1 + x5) & M; t = x13 ^ x1; x13 = ((t << 8) & M) | (t >> 24)
            x9 = (x9 + x13) & M; t = x5 ^ x9; x5 = ((t << 7) & M) | (t >> 25)
            x2 = (x2 + x6) & M; t = x14 ^ x2; x14 = ((t << 16) & M) | (t >> 16)
            x10 = (x10 + x14) & M; t = x6 ^ x10; x6 = ((t << 12) & M) | (t >> 20)
            x2 = (x2 + x6) & M; t = x14 ^ x2; x14 = ((t << 8) & M) | (t >> 24)
            x10 = (x10 + x14) & M; t = x6 ^ x10; x6 = ((t << 7) & M) | (t >> 25)
            x3 = (x3 + x7) & M; t = x15 ^ x3; x15 = ((t << 16) & M) | (t >> 16)
            x11 = (x11 + x15) & M; t = x7 ^ x11; x7 = ((t << 12) & M) | (t >> 20)
            x3 = (x3 + x7) & M; t = x15 ^ x3; x15 = ((t << 8) & M) | (t >> 24)
            x11 = (x11 + x15) & M; t = x7 ^ x11; x7 = ((t << 7) & M) | (t >> 25)
            x0 = (x0 + x5) & M; t = x15 ^ x0; x15 = ((t << 16) & M) | (t >> 16)
            x10 = (x10 + x15) & M; t = x5 ^ x10; x5 = ((t << 12) & M) | (t >> 20)
            x0 = (x0 + x5) & M; t = x15 ^ x0; x15 = ((t << 8) & M) | (t >> 24)
            x10 = (x10 + x15) & M; t = x5 ^ x10; x5 = ((t << 7) & M) | (t >> 25)
            x1 = (x1 + x6) & M; t = x12 ^ x1; x12 = ((t << 16) & M) | (t >> 16)
            x11 = (x11 + x12) & M; t = x6 ^ x11; x6 = ((t << 12) & M) | (t >> 20)
            x1 = (x1 + x6) & M; t = x12 ^ x1; x12 = ((t << 8) & M) | (t >> 24)
            x11 = (x11 + x12) & M; t = x6 ^ x11; x6 = ((t << 7) & M) | (t >> 25)
            x2 = (x2 + x7) & M; t = x13 ^ x2; x13 = ((t << 16) & M) | (t >> 16)
            x8 = (x8 + x13) & M; t = x7 ^ x8; x7 = ((t << 12) & M) | (t >> 20)
            x2 = (x2 + x7) & M; t = x13 ^ x2; x13 = ((t << 8) & M) | (t >> 24)
            x8 = (x8 + x13) & M; t = x7 ^ x8; x7 = ((t << 7) & M) | (t >> 25)
            x3 = (x3 + x4) & M; t = x14 ^ x3; x14 = ((t << 16) & M) | (t >> 16)
            x9 = (x9 + x14) & M; t = x4 ^ x9; x4 = ((t << 12) & M) | (t >> 20)
            x3 = (x3 + x4) & M; t = x14 ^ x3; x14 = ((t << 8) & M) | (t >> 24)
            x9 = (x9 + x14) & M; t = x4 ^ x9; x4 = ((t << 7) & M) | (t >> 25)
        blocks.append(pack(
            (x0 + 0x61707865) & M, (x1 + 0x3320646E) & M,
            (x2 + 0x79622D32) & M, (x3 + 0x6B206574) & M,
            (x4 + k0) & M, (x5 + k1) & M, (x6 + k2) & M, (x7 + k3) & M,
            (x8 + k4) & M, (x9 + k5) & M, (x10 + k6) & M, (x11 + k7) & M,
            (x12 + c) & M, (x13 + n0) & M, (x14 + n1) & M, (x15 + n2) & M,
        ))
    return b"".join(blocks)[:length]


def _chacha20_block(key: bytes, counter: int, nonce: bytes) -> bytes:
    return _chacha20_keystream(key, counter, nonce, 64)


def _chacha20_xor(key: bytes, counter: int, nonce: bytes, data: bytes) -> bytes:
    if not data:
        return b""
    stream = _chacha20_keystream(key, counter, nonce, len(data))
    mixed = int.from_bytes(data, "little") ^ int.from_bytes(stream, "little")
    return mixed.to_bytes(len(data), "little")


def _poly1305(key: bytes, message: bytes) -> bytes:
    r = int.from_bytes(key[:16], "little") & 0x0FFFFFFC0FFFFFFC0FFFFFFC0FFFFFFF
    s = int.from_bytes(key[16:], "little")
    p = (1 << 130) - 5
    accumulator = 0
    for offset in range(0, len(message), 16):
        block = message[offset : offset + 16] + b"\x01"
        accumulator = ((accumulator + int.from_bytes(block, "little")) * r) % p
    return ((accumulator + s) & ((1 << 128) - 1)).to_bytes(16, "little")


def _pad16(data: bytes) -> bytes:
    return b"\x00" * (-len(data) % 16)


def _pure_tag(key: bytes, nonce: bytes, aad: bytes, ciphertext: bytes) -> bytes:
    otk = _chacha20_block(key, 0, nonce)[:32]
    mac_data = (
        aad
        + _pad16(aad)
        + ciphertext
        + _pad16(ciphertext)
        + struct.pack("<QQ", len(aad), len(ciphertext))
    )
    return _poly1305(otk, mac_data)


def aead_encrypt(key: bytes, nonce: bytes, plaintext: bytes, aad: bytes) -> bytes:
    """ChaCha20-Poly1305 encryption: ``ciphertext || tag``."""
    if _LibChaCha is not None:
        return _LibChaCha(key).encrypt(nonce, plaintext, aad)
    ciphertext = _chacha20_xor(key, 1, nonce, plaintext)
    return ciphertext + _pure_tag(key, nonce, aad, ciphertext)


def aead_decrypt(key: bytes, nonce: bytes, sealed: bytes, aad: bytes) -> bytes:
    """ChaCha20-Poly1305 decryption; raises :class:`ContentSealError`."""
    if len(sealed) < TAG_LENGTH:
        raise ContentSealError("malformed")
    if _LibChaCha is not None:
        try:
            return _LibChaCha(key).decrypt(nonce, sealed, aad)
        except _LibInvalidTag as error:  # type: ignore[misc]
            raise ContentSealError("does_not_open") from error
    ciphertext, tag = sealed[:-TAG_LENGTH], sealed[-TAG_LENGTH:]
    if not hmac.compare_digest(_pure_tag(key, nonce, aad, ciphertext), tag):
        raise ContentSealError("does_not_open")
    return _chacha20_xor(key, 1, nonce, ciphertext)


# ── contexts, ciphers, blind indexes ────────────────────────────────────────


@dataclass(frozen=True)
class ContentContext:
    """Where a value lives: a purpose plus labelled stable ids (never a path)."""

    purpose: str
    fields: tuple[tuple[str, str], ...] = ()

    def with_field(self, label: str, value: object) -> "ContentContext":
        return ContentContext(self.purpose, (*self.fields, (label, str(value))))

    def associated_data(self, store: str) -> bytes:
        parts = [_AAD_LABEL, b"store", store.encode(), b"purpose", self.purpose.encode()]
        for label, value in self.fields:
            parts.extend((label.encode(), value.encode()))
        return _length_prefixed(parts)


def context(purpose: str, *pairs: tuple[str, object]) -> ContentContext:
    result = ContentContext(purpose)
    for label, value in pairs:
        result = result.with_field(label, value)
    return result


@dataclass(frozen=True)
class Opened:
    value: object
    needs_reseal: bool


class ContentCipher:
    """The content cipher of one registered store (the name is KDF input)."""

    def __init__(
        self, store: str, key: bytes, root_kcv: str, retired: tuple[tuple[bytes, str], ...] = ()
    ) -> None:
        self.store = store
        self._key = key
        self.kcv = root_kcv
        # (key, root kcv) of the same store under retired roots: they open, never seal.
        self._retired = retired

    @classmethod
    def from_root(cls, root: bytes, store: str) -> "ContentCipher":
        if len(root) != KEY_LENGTH:
            raise SealError("a root key must be 32 bytes")
        if not store or any(not 0x21 <= ord(ch) < 0x7F for ch in store):
            raise ValueError("store names are non-empty printable ASCII without spaces")
        return cls(store, hkdf32(root, _CONTENT_SALT, _CONTENT_INFO + store.encode()), kcv(root))

    def with_retired_roots(self, roots: Iterable[bytes]) -> "ContentCipher":
        """The same cipher, also able to open values sealed under ``roots``."""
        retired = list(self._retired)
        for root in roots:
            root_kcv = kcv(root)
            if root_kcv == self.kcv or any(seen == root_kcv for _, seen in retired):
                continue
            retired.append(
                (hkdf32(root, _CONTENT_SALT, _CONTENT_INFO + self.store.encode()), root_kcv)
            )
        return ContentCipher(self.store, self._key, self.kcv, tuple(retired))

    def subcipher(self, label: str) -> "ContentCipher":
        info = _SUB_INFO + label.encode()
        return ContentCipher(
            self.store,
            hkdf32(self._key, _SUB_SALT, info),
            self.kcv,
            tuple((hkdf32(key, _SUB_SALT, info), root_kcv) for key, root_kcv in self._retired),
        )

    def _seal(self, plaintext: bytes, ctx: ContentContext, nonce: bytes | None = None) -> bytes:
        nonce = nonce if nonce is not None else secrets.token_bytes(NONCE_LENGTH)
        return nonce + aead_encrypt(self._key, nonce, plaintext, ctx.associated_data(self.store))

    def _open(self, blob: bytes, ctx: ContentContext) -> tuple[bytes, bool]:
        """The plaintext and whether a retired key opened it (reseal it then)."""
        if len(blob) < NONCE_LENGTH + TAG_LENGTH:
            raise ContentSealError("malformed")
        aad = ctx.associated_data(self.store)
        nonce, sealed = blob[:NONCE_LENGTH], blob[NONCE_LENGTH:]
        try:
            return aead_decrypt(self._key, nonce, sealed, aad), False
        except ContentSealError:
            for key, _ in self._retired:
                try:
                    return aead_decrypt(key, nonce, sealed, aad), True
                except ContentSealError:
                    continue
            raise

    def seal_text(self, plaintext: str, ctx: ContentContext, *, nonce: bytes | None = None) -> str:
        sealed = self._seal(plaintext.encode("utf-8"), ctx, nonce)
        return TEXT_PREFIX + base64.b64encode(sealed).decode("ascii")

    def open_text(self, stored: str, ctx: ContentContext) -> Opened:
        if not stored.startswith(TEXT_PREFIX):
            if _VERSIONED_TEXT.match(stored):
                raise ContentSealError("unknown_version")
            return Opened(stored, stored != "")
        try:
            blob = base64.b64decode(stored[len(TEXT_PREFIX) :], validate=True)
        except (ValueError, TypeError) as error:
            raise ContentSealError("malformed") from error
        plaintext, retired = self._open(blob, ctx)
        try:
            return Opened(plaintext.decode("utf-8"), retired)
        except UnicodeDecodeError as error:
            raise ContentSealError("malformed") from error

    def seal_bytes(self, plaintext: bytes, ctx: ContentContext, *, nonce: bytes | None = None) -> bytes:
        if any(plaintext.startswith(magic) for magic in RYU_MAGICS):
            raise ContentSealError("already_a_ryu_envelope")
        return BYTES_MAGIC + self._seal(plaintext, ctx, nonce)

    def open_bytes(self, stored: bytes, ctx: ContentContext) -> Opened:
        if stored.startswith(BYTES_MAGIC):
            plaintext, retired = self._open(stored[len(BYTES_MAGIC) :], ctx)
            return Opened(plaintext, retired)
        if len(stored) >= 8 and _RYU_MAGIC.match(stored[:8]):
            raise ContentSealError("unknown_version")
        return Opened(bytes(stored), len(stored) > 0)


def normalize(text: str) -> str:
    """``NFKC(lowercase(NFKC(text)))`` - the shared blind-index normalization."""
    return unicodedata.normalize("NFKC", unicodedata.normalize("NFKC", text).lower())


def tokenize(text: str) -> list[str]:
    """The shared full-text tokenizer (split on non-L/M/N, 2..64 chars, stopwords)."""
    terms: list[str] = []
    current: list[str] = []
    for ch in normalize(text):
        if unicodedata.category(ch)[0] in "LMN":
            current.append(ch)
        elif current:
            terms.append("".join(current))
            current = []
    if current:
        terms.append("".join(current))
    out = {term[:64] for term in terms if len(term) >= 2}
    out = {term for term in out if term not in STOPWORDS}
    return sorted(out, key=lambda term: term.encode("utf-8"))


class BlindIndex:
    """Deterministic keyed tokens for one ``(store, field)``."""

    def __init__(self, key: bytes) -> None:
        self._key = key

    @classmethod
    def from_root(cls, root: bytes, store: str, field_name: str) -> "BlindIndex":
        info = _BLIND_INFO + store.encode() + b"\x00" + field_name.encode()
        return cls(hkdf32(root, _BLIND_SALT, info))

    def token(self, normalized: str) -> str:
        return "bi1:" + self.term(normalized)

    def term(self, term: str) -> str:
        return hmac.new(self._key, term.encode("utf-8"), hashlib.sha256).digest()[:16].hex()


# ── process key custody ─────────────────────────────────────────────────────


def _pop_key_env(environ: "os._Environ[str] | dict[str, str]") -> dict[str, str]:
    captured: dict[str, str] = {}
    for name in SEAL_ENV_VARS:
        value = environ.pop(name, None)
        if value is not None and value.strip():
            captured[name] = value.strip()
    return captured


# Removed from the environment at import so no child process inherits a key.
_PROCESS_KEY_ENV = _pop_key_env(os.environ)


def scrubbed_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """A copy of ``environ`` (default ``os.environ``) without key-carrying variables."""
    source = os.environ if environ is None else environ
    return {name: value for name, value in source.items() if name not in SEAL_ENV_VARS}


def _decode_key(encoded: str, name: str) -> bytes:
    try:
        key = base64.b64decode(encoded.strip(), validate=True)
    except (ValueError, TypeError) as error:
        raise SealError(f"{name} must be base64 of exactly 32 bytes") from error
    if len(key) != KEY_LENGTH:
        raise SealError(f"{name} must be base64 of exactly 32 bytes")
    return key


def _write_private(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, text.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _pin_standalone_kcv(secrets_dir: Path, root_kcv: str, source: str) -> None:
    path = secrets_dir / STANDALONE_KCV_FILE
    if path.is_file():
        try:
            recorded = json.loads(path.read_text(encoding="utf-8")).get("kcv")
        except (OSError, ValueError, AttributeError) as error:
            raise SealError(f"{path} is unreadable; refusing to guess which key sealed this data") from error
        if recorded != root_kcv:
            raise SealError(
                f"this app's standalone data was sealed under a different key (kcv {recorded});"
                f" restore that {MASTER_KEY_ENV} or {STANDALONE_ROOT_FILE}"
            )
        return
    secrets_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    _write_private(path, json.dumps({"v": 1, "kcv": root_kcv, "source": source}) + "\n")


def _load_or_create_standalone_root(secrets_dir: Path) -> bytes:
    path = secrets_dir / STANDALONE_ROOT_FILE
    if path.is_file():
        return _decode_key(path.read_text(encoding="utf-8"), STANDALONE_ROOT_FILE)
    if (secrets_dir / STANDALONE_KCV_FILE).is_file():
        raise SealError(
            f"this app's standalone data was sealed under {MASTER_KEY_ENV}; set it again"
            " instead of generating a new key"
        )
    secrets_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(secrets_dir, 0o700)
    root = secrets.token_bytes(KEY_LENGTH)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, base64.b64encode(root) + b"\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return root


@dataclass(frozen=True)
class StoreKeyStatus:
    """How one store relates to the process key (design section 11)."""

    state: str  # record | current | retired | mismatch | compat | no_key
    writable: bool
    sealing: bool
    kcv: str | None = None
    recorded: str | None = None


@dataclass
class PluginKeys:
    """A satellite's keys, resolved once per process."""

    plugin_id: str
    root: bytes | None
    source: str | None  # delivered | master | standalone | None (compat)
    # The same scope's subkeys under retired master keys (``RYU_SEAL_RETIRED_KEYS``):
    # stores sealed under them still open and are resealed under ``root``.
    retired_roots: tuple[bytes, ...] = ()
    kcv: str | None = field(init=False)

    def __post_init__(self) -> None:
        self.kcv = kcv(self.root) if self.root is not None else None

    @property
    def can_seal(self) -> bool:
        return self.root is not None

    def content_cipher(self, store: str) -> ContentCipher:
        if self.root is None:
            raise SealError(f"no sealing key for plugin:{self.plugin_id} ({COMPAT_HEALTH})")
        cipher = ContentCipher.from_root(self.root, store)
        return cipher.with_retired_roots(self.retired_roots) if self.retired_roots else cipher

    def blind_index(self, store: str, field_name: str) -> BlindIndex:
        if self.root is None:
            raise SealError(f"no sealing key for plugin:{self.plugin_id} ({COMPAT_HEALTH})")
        return BlindIndex.from_root(self.root, store, field_name)

    def store_key_state(self, recorded_kcv: str | None, holds_sealed: bool) -> StoreKeyStatus:
        if self.root is None:
            if recorded_kcv is not None or holds_sealed:
                return StoreKeyStatus("no_key", False, False)
            return StoreKeyStatus("compat", True, False)
        if recorded_kcv is None:
            return StoreKeyStatus("record", True, True, kcv=self.kcv)
        if recorded_kcv == self.kcv:
            return StoreKeyStatus("current", True, True, kcv=self.kcv)
        if any(kcv(retired) == recorded_kcv for retired in self.retired_roots):
            # A backup import or `keys adopt` retired the key that sealed it:
            # values open, are resealed, and the current KCV is recorded once
            # none is left unreadable.
            return StoreKeyStatus("retired", True, True, kcv=self.kcv, recorded=recorded_kcv)
        return StoreKeyStatus("mismatch", False, False, recorded=recorded_kcv)


def capture_plugin_keys(
    plugin_id: str,
    key_env: Mapping[str, str],
    *,
    under_core: bool,
    data_dir: Path,
) -> PluginKeys:
    """Resolve a satellite's root from captured key variables (see module doc)."""
    delivered = key_env.get(SEAL_KEY_ENV)
    if delivered:
        scope = key_env.get(SEAL_SCOPE_ENV, "")
        if scope != f"plugin:{plugin_id}":
            raise SealError(f"{SEAL_KEY_ENV} was delivered for {scope or 'no scope'!r}, not plugin:{plugin_id}")
        retired = tuple(
            _decode_key(item, SEAL_RETIRED_KEYS_ENV)
            for item in key_env.get(SEAL_RETIRED_KEYS_ENV, "").split(",")
            if item.strip()
        )
        return PluginKeys(plugin_id, _decode_key(delivered, SEAL_KEY_ENV), "delivered", retired)
    secrets_dir = data_dir / "secrets"
    master = key_env.get(MASTER_KEY_ENV)
    if master:
        root = derive_plugin_key(_decode_key(master, MASTER_KEY_ENV), plugin_id)
        if not under_core:
            if (secrets_dir / STANDALONE_ROOT_FILE).is_file():
                raise SealError(
                    f"this app's standalone data was sealed under {STANDALONE_ROOT_FILE};"
                    f" unset {MASTER_KEY_ENV} or move that data aside"
                )
            _pin_standalone_kcv(secrets_dir, kcv(root), "master")
        return PluginKeys(plugin_id, root, "master")
    if under_core:
        return PluginKeys(plugin_id, None, None)
    root = _load_or_create_standalone_root(secrets_dir)
    _pin_standalone_kcv(secrets_dir, kcv(root), "standalone")
    return PluginKeys(plugin_id, root, "standalone")


_PROCESS_KEYS: dict[tuple[str, str], PluginKeys] = {}
_PROCESS_KEYS_LOCK = threading.Lock()


def process_keys(plugin_id: str, data_dir: Path) -> PluginKeys:
    """This process's keys for ``plugin_id``, resolved once per data dir."""
    cache_key = (plugin_id, str(data_dir))
    with _PROCESS_KEYS_LOCK:
        keys = _PROCESS_KEYS.get(cache_key)
        if keys is None:
            keys = capture_plugin_keys(
                plugin_id,
                _PROCESS_KEY_ENV,
                under_core=bool(os.environ.get(PLUGIN_DATA_DIR_ENV, "").strip()),
                data_dir=data_dir,
            )
            _PROCESS_KEYS[cache_key] = keys
        return keys


def reset_process_keys_for_tests(key_env: Mapping[str, str] | None = None) -> None:
    """Forget resolved keys (tests); optionally replace the captured key env."""
    with _PROCESS_KEYS_LOCK:
        _PROCESS_KEYS.clear()
        if key_env is not None:
            _PROCESS_KEY_ENV.clear()
            _PROCESS_KEY_ENV.update(key_env)


# ── SQLite stores ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SealedColumn:
    """One sealed TEXT column: AAD binds ``bind`` (row columns, also the labels)."""

    table: str
    column: str
    order_key: str
    bind: tuple[str, ...]

    @property
    def purpose(self) -> str:
        return f"{self.table}.{self.column}"

    def context(self, row: Mapping[str, object]) -> ContentContext:
        ctx = ContentContext(self.purpose)
        for name in self.bind:
            value = row[name]
            ctx = ctx.with_field(name, "" if value is None else value)
        return ctx


_LOG = logging.getLogger("ryu_research.content_seal")
_STATE_DDL = f"""CREATE TABLE IF NOT EXISTS ryu_content_seal (
    store TEXT NOT NULL, item TEXT NOT NULL, state TEXT NOT NULL, cursor TEXT,
    kcv TEXT, min_reader INTEGER NOT NULL DEFAULT {SEAL_MIN_READER},
    updated_at INTEGER NOT NULL, PRIMARY KEY(store, item))"""
_STORE_ITEM = "__store__"
_BATCH = 500
# Largest zero-filled row written at once while scrubbing freed pages. Each
# chunk is its own transaction followed by a WAL truncate, so the WAL never
# grows past about one chunk.
_SCRUB_CHUNK = 64 * 1024 * 1024
# Free disk space kept beyond one scrub chunk before the scrub runs.
_SCRUB_HEADROOM = 256 * 1024 * 1024
_VACUUM_HEADROOM = 256 * 1024 * 1024
# `__store__` state: free pages were zero-filled; only the VACUUM is pending.
_SCRUBBED = "scrubbed"
_VACUUM_PENDING = "vacuum_pending"


def _free_space(directory: Path) -> int:
    """Free bytes on the volume holding ``directory``."""
    return shutil.disk_usage(directory).free


def _read_state(connection: sqlite3.Connection, store: str, item: str):
    return connection.execute(
        "SELECT state, cursor, kcv FROM ryu_content_seal WHERE store = ? AND item = ?",
        (store, item),
    ).fetchone()


def _write_state(
    connection: sqlite3.Connection, store: str, item: str, state: str, cursor: str | None, root_kcv: str | None
) -> None:
    connection.execute(
        f"""INSERT INTO ryu_content_seal(store, item, state, cursor, kcv, min_reader, updated_at)
        VALUES (?, ?, ?, ?, ?, {SEAL_MIN_READER}, ?)
        ON CONFLICT(store, item) DO UPDATE SET state = excluded.state,
            cursor = excluded.cursor, kcv = excluded.kcv,
            min_reader = MAX(min_reader, excluded.min_reader), updated_at = excluded.updated_at""",
        (store, item, state, cursor, root_kcv, int(time.time() * 1000)),
    )


class StoreSeal:
    """Per-value sealing of one SQLite store (open on every connection)."""

    def __init__(
        self,
        store: str,
        keys: PluginKeys | None,
        columns: Sequence[SealedColumn],
        status: StoreKeyStatus | None,
    ) -> None:
        self.store = store
        self.columns = tuple(columns)
        self.status = status
        self._cipher = (
            keys.content_cipher(store) if keys is not None and status is not None and status.sealing else None
        )
        self.unreadable_rows = 0
        # Sealed under a retired key: every column is rescanned and resealed;
        # the current KCV is recorded only once nothing is left unreadable.
        self.retired_left = status is not None and status.state == "retired"
        self.vacuum_pending = False
        # Free pages already zero-filled since the last reseal; VACUUM may still be pending.
        self.scrubbed = False

    @classmethod
    def open(
        cls,
        connection: sqlite3.Connection,
        path: Path | None,
        store: str,
        keys: PluginKeys | None,
        columns: Sequence[SealedColumn],
    ) -> "StoreSeal":
        """Resolve the store's key state, migrate legacy plaintext, scrub."""
        connection.execute("PRAGMA secure_delete = ON")
        connection.execute(_STATE_DDL)
        connection.commit()
        _refuse_newer_format(connection, store)
        if keys is None:
            return cls(store, None, columns, None)
        recorded = _read_state(connection, store, _STORE_ITEM)
        status = keys.store_key_state(
            recorded[2] if recorded is not None else None,
            _holds_sealed(connection, store),
        )
        seal = cls(store, keys, columns, status)
        # The first key for this store (or an unfinished scrub): free pages and
        # WAL frames may still hold plaintext written before sealing, or rows
        # deleted under an older version without secure_delete.
        recorded_state = recorded[0] if recorded is not None else None
        seal.vacuum_pending = recorded_state in (_VACUUM_PENDING, _SCRUBBED) or status.state == "record"
        seal.scrubbed = seal.vacuum_pending and recorded_state == _SCRUBBED
        if status.sealing:
            seal.migrate(connection, path)
        return seal

    @property
    def sealing(self) -> bool:
        return self._cipher is not None

    def health(self) -> dict:
        state = self.status.state if self.status is not None else "unmanaged"
        sealing = {
            "compat": COMPAT_HEALTH,
            "mismatch": "key mismatch",
            "no_key": "no key",
            "unmanaged": "unmanaged",
        }.get(state, "ok")
        return {
            "store": self.store,
            "state": state,
            "sealing": sealing,
            "unreadable_rows": self.unreadable_rows,
            "vacuum_pending": self.vacuum_pending,
        }

    def assert_writable(self) -> None:
        """Raise :class:`SealUnavailableError` unless writes may proceed."""
        if self._cipher is not None or self.status is None or self.status.state == "compat":
            return
        if self.status.state == "mismatch":
            raise SealUnavailableError(
                f"{self.store} was sealed under a different key; writes are refused until that key is restored"
            )
        raise SealUnavailableError(
            f"{self.store} holds sealed data but no sealing key was delivered; writes are refused"
        )

    def seal_text(self, column: SealedColumn, row: Mapping[str, object], value: str) -> str:
        """Seal for writing. Empty strings stay empty (only emptiness is visible)."""
        self.assert_writable()
        if value == "" or self._cipher is None:
            # Compat-mode plaintext that starts like an envelope would later
            # read as sealed data (locking the store once no key is held, and
            # never being sealed when one arrives), so it is refused.
            if (
                self.status is not None
                and self.status.state == "compat"
                and _VERSIONED_TEXT.match(value)
            ):
                raise ContentSealError("already_a_ryu_envelope")
            return value
        return self._cipher.seal_text(value, column.context(row))

    def open_text(self, column: SealedColumn, row: Mapping[str, object], stored: object) -> str:
        """Open a stored value; raises :class:`ContentSealError` if it does not open."""
        if stored is None:
            return ""
        text = str(stored)
        if not text.startswith(TEXT_PREFIX):
            if _VERSIONED_TEXT.match(text):
                raise ContentSealError("unknown_version")
            return text
        if self._cipher is None:
            raise ContentSealError("does_not_open")
        return str(self._cipher.open_text(text, column.context(row)).value)

    def migrate(self, connection: sqlite3.Connection, path: Path | None) -> dict:
        """Seal legacy plaintext: cursor-paginated, compare-and-swap, resumable."""
        report = {"resealed": 0, "unreadable": 0}
        cipher = self._cipher
        if cipher is None:
            return report
        for column in self.columns:
            item = column.purpose
            state = _read_state(connection, self.store, item)
            if state is not None and state[0] == "sealed" and not self.retired_left:
                # An older build run after sealing (a rollback) writes
                # plaintext; a column holding any is scanned again.
                if not _holds_plaintext(connection, column):
                    continue
                _LOG.warning(
                    "%s: %s holds plaintext written after it was sealed (an older version ran); sealing it again",
                    self.store,
                    item,
                )
            cursor = json.loads(state[1]) if state is not None and state[1] else None
            names = list(dict.fromkeys([column.order_key, *column.bind, column.column]))
            selected = ", ".join(f'"{name}"' for name in names)
            while True:
                rows = connection.execute(
                    f'SELECT {selected} FROM "{column.table}" WHERE ? IS NULL OR "{column.order_key}" > ?'
                    f' ORDER BY "{column.order_key}" LIMIT {_BATCH}',
                    (cursor, cursor),
                ).fetchall()
                if not rows:
                    break
                with connection:
                    for values in rows:
                        row = dict(zip(names, values))
                        stored = row[column.column]
                        if stored is None or stored == "":
                            continue
                        ctx = column.context(row)
                        try:
                            opened = cipher.open_text(str(stored), ctx)
                        except ContentSealError:
                            report["unreadable"] += 1
                            continue
                        if opened.needs_reseal:
                            connection.execute(
                                f'UPDATE "{column.table}" SET "{column.column}" = ?'
                                f' WHERE "{column.order_key}" = ? AND "{column.column}" IS ?',
                                (cipher.seal_text(str(opened.value), ctx), row[column.order_key], stored),
                            )
                            report["resealed"] += 1
                    cursor = rows[-1][0]
                    _write_state(connection, self.store, item, "sealing", json.dumps(cursor), cipher.kcv)
            with connection:
                _write_state(connection, self.store, item, "sealed", None, cipher.kcv)
        self.unreadable_rows += report["unreadable"]
        if self.retired_left and report["unreadable"] == 0:
            _LOG.info("%s: resealed from a retired key under the current key", self.store)
            self.retired_left = False
        if report["resealed"]:
            # The reseal freed new plaintext pages: scrub them again.
            self.vacuum_pending = True
            self.scrubbed = False
        self._write_store_state(connection)
        if self.vacuum_pending:
            self._finish(connection, path)
        return report

    def _write_store_state(self, connection: sqlite3.Connection) -> None:
        state = "sealed"
        if self.vacuum_pending:
            state = _SCRUBBED if self.scrubbed else _VACUUM_PENDING
        root_kcv = self._cipher.kcv if self._cipher else None
        if self.retired_left and self.status is not None and self.status.state == "retired":
            # Values still do not open: keep the retired KCV so the next start rescans.
            root_kcv = self.status.recorded
        with connection:
            _write_state(connection, self.store, _STORE_ITEM, state, None, root_kcv)

    def _finish(self, connection: sqlite3.Connection, path: Path | None) -> None:
        """Checkpoint, zero every free page, then VACUUM; best-effort, never raises.

        The zero fill runs only with one chunk plus headroom free (it passes
        through the WAL one chunk at a time) and is recorded as done
        (``scrubbed``) so a deferred VACUUM does not repeat it; the VACUUM runs
        only with 2x the database + 256 MiB free. Anything skipped or failed
        keeps the store pending and is retried at the next start.
        """
        try:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error as error:
            _LOG.warning("%s: could not truncate the write-ahead log (%s)", self.store, error)
            return
        if not self.scrubbed and not self._scrub(connection, path):
            return
        if not _has_room(path, lambda size: 2 * size + _VACUUM_HEADROOM):
            _LOG.warning(
                "%s: not enough free disk space to compact; free pages are zero-filled"
                " and the compaction is retried at a later start",
                self.store,
            )
            return
        try:
            connection.execute("VACUUM")
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error as error:
            _LOG.warning("%s: compaction failed; it is retried at a later start (%s)", self.store, error)
            return
        self.vacuum_pending = False
        self.scrubbed = False
        self._write_store_state(connection)

    def _scrub(self, connection: sqlite3.Connection, path: Path | None) -> bool:
        """Zero-fill free pages; True when done (and recorded as ``scrubbed``)."""
        if not _has_room(path, lambda _size: _SCRUB_CHUNK + _SCRUB_HEADROOM):
            _LOG.warning(
                "%s: not enough free disk space to zero-fill freed pages; retried at a later start",
                self.store,
            )
            return False
        try:
            _scrub_free_pages(connection)
        except sqlite3.Error as error:
            _drop_scrub_table(connection)
            _LOG.warning(
                "%s: zero-filling freed pages failed; it is retried at a later start (%s)", self.store, error
            )
            return False
        self.scrubbed = True
        self._write_store_state(connection)
        return True


def _has_room(path: Path | None, needed: Callable[[int], int]) -> bool:
    """Whether the store's volume has ``needed(database size)`` bytes free (True for ``:memory:``)."""
    if path is None:
        return True
    try:
        return _free_space(path.parent) >= needed(path.stat().st_size)
    except OSError:
        return False


def _refuse_newer_format(connection: sqlite3.Connection, store: str) -> None:
    """Refuse a store a newer build sealed under an unsupported reader format."""
    row = connection.execute(
        "SELECT MAX(min_reader) FROM ryu_content_seal WHERE store = ?", (store,)
    ).fetchone()
    reader = row[0] if row is not None else None
    if reader is not None and int(reader) > SEAL_MIN_READER:
        raise NewerStoreFormatError(
            f"{store} was written by a newer Ryu (needs reader format {reader}, "
            f"this build reads {SEAL_MIN_READER}); update the app"
        )


def _holds_plaintext(connection: sqlite3.Connection, column: SealedColumn) -> bool:
    """Whether a column marked sealed holds a non-empty value with no envelope prefix.

    Values with an unknown envelope version count as sealed, so they do not
    force a rescan on every start.
    """
    name = f'"{column.column}"'
    hit = connection.execute(
        f'SELECT 1 FROM "{column.table}" WHERE length({name}) > 0'
        f" AND substr(CAST({name} AS BLOB), 1, 6) <> CAST('encb:v' AS BLOB) LIMIT 1"
    ).fetchone()
    return hit is not None


def _holds_sealed(connection: sqlite3.Connection, store: str) -> bool:
    """Whether this store was ever sealed: a key check value recorded for it.

    Values that merely look sealed do not count: compat-mode plaintext could
    start with an envelope prefix and would otherwise lock the store.
    """
    hit = connection.execute(
        "SELECT 1 FROM ryu_content_seal WHERE store = ? AND kcv IS NOT NULL LIMIT 1",
        (store,),
    ).fetchone()
    return hit is not None


def _scrub_free_pages(connection: sqlite3.Connection) -> None:
    """Overwrite every freelist page with zeros, one WAL-truncated chunk at a time.

    Zero-filled scratch rows take the free pages, each chunk in its own
    transaction followed by a WAL truncate (so the WAL stays near one chunk);
    then the scratch table is dropped. Its pages already hold zeros, so the
    drop runs without ``secure_delete`` and does not rewrite them.
    """
    # A scratch table left by an interrupted scrub holds only zeros.
    _drop_scrub_table(connection)
    free = int(connection.execute("PRAGMA freelist_count").fetchone()[0])
    if free <= 0:
        return
    remaining = free * int(connection.execute("PRAGMA page_size").fetchone()[0])
    with connection:
        connection.execute("CREATE TABLE ryu_content_seal_scrub (fill BLOB)")
    while remaining > 0:
        size = min(remaining, _SCRUB_CHUNK)
        with connection:
            connection.execute("INSERT INTO ryu_content_seal_scrub(fill) VALUES (zeroblob(?))", (size,))
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        remaining -= size
    _drop_scrub_table(connection)


def _drop_scrub_table(connection: sqlite3.Connection) -> None:
    """Drop the zero-filled scratch table without rewriting its pages (best-effort)."""
    try:
        connection.execute("PRAGMA secure_delete = OFF")
        with connection:
            connection.execute("DROP TABLE IF EXISTS ryu_content_seal_scrub")
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.Error:
        pass  # A leftover table holds only zeros and is dropped next time.
    finally:
        connection.execute("PRAGMA secure_delete = ON")
