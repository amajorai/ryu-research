"""The vendored content sealer against the shared phase-2 vectors."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ryu_research import ryu_content_seal as seal

VECTORS = json.loads(
    (Path(__file__).parent / "fixtures" / "content-v1.json").read_text(encoding="utf-8")
)
ROOT = base64.b64decode(VECTORS["root_key_b64"])
STORE = VECTORS["store"]


def _vector_context(spec: dict) -> seal.ContentContext:
    return seal.context(spec["purpose"], *(tuple(pair) for pair in spec["fields"]))


class VectorTests(unittest.TestCase):
    def _check_both_aeads(self, check) -> None:
        check()
        with patch.object(seal, "_LibChaCha", None):
            check()

    def test_fixture_is_the_same_json_as_the_rust_crates_vectors(self) -> None:
        # The canonical vectors live in the Rust crypto crate. This satellite
        # vendors a copy so it can ship standalone; when the monorepo is present
        # the copy must not drift (the TypeScript copy has the same guard).
        rust = (
            Path(__file__).resolve().parents[4]
            / "crates"
            / "core"
            / "crypto"
            / "tests"
            / "vectors"
            / "content-v1.json"
        )
        if not rust.is_file():
            self.skipTest("standalone satellite mirror has no Rust crate")
        self.assertEqual(json.loads(rust.read_text(encoding="utf-8")), VECTORS)

    def test_content_key_aad_and_kcv(self) -> None:
        cipher = seal.ContentCipher.from_root(ROOT, STORE)
        self.assertEqual(base64.b64encode(cipher._key).decode(), VECTORS["content_key_b64"])
        ctx = _vector_context(VECTORS["text"])
        self.assertEqual(ctx.associated_data(STORE).hex(), VECTORS["aad_hex"])
        self.assertEqual(seal.kcv(ROOT), VECTORS["kcv"]["kcv_hex"])
        sub = cipher.subcipher(VECTORS["subkey"]["label"])
        self.assertEqual(base64.b64encode(sub._key).decode(), VECTORS["subkey"]["key_b64"])

    def test_text_and_bytes_envelopes(self) -> None:
        cipher = seal.ContentCipher.from_root(ROOT, STORE)
        text = VECTORS["text"]
        ctx = _vector_context(text)
        nonce = bytes.fromhex(text["nonce_hex"])

        def check() -> None:
            self.assertEqual(cipher.seal_text(text["plaintext"], ctx, nonce=nonce), text["sealed"])
            self.assertEqual(cipher.open_text(text["sealed"], ctx).value, text["plaintext"])
            sealed = cipher.seal_bytes(bytes.fromhex(VECTORS["bytes"]["plaintext_hex"]), ctx, nonce=nonce)
            self.assertEqual(sealed.hex(), VECTORS["bytes"]["sealed_hex"])
            self.assertEqual(cipher.open_bytes(sealed, ctx).value.hex(), VECTORS["bytes"]["plaintext_hex"])
            with self.assertRaises(seal.ContentSealError):
                cipher.open_text(text["sealed"], ctx.with_field("extra", "1"))

        self._check_both_aeads(check)

    def test_pure_python_aead_round_trips_long_values(self) -> None:
        key = bytes(range(32))
        nonce = bytes(range(12))
        message = os.urandom(1000)
        with patch.object(seal, "_LibChaCha", None):
            pure = seal.aead_encrypt(key, nonce, message, b"aad")
        if seal._LibChaCha is not None:
            self.assertEqual(pure, seal.aead_encrypt(key, nonce, message, b"aad"))
        with patch.object(seal, "_LibChaCha", None):
            self.assertEqual(seal.aead_decrypt(key, nonce, pure, b"aad"), message)

    def test_blind_index_and_tokenizer(self) -> None:
        bi = VECTORS["blind_index"]
        self.assertEqual(seal.BlindIndex.from_root(ROOT, STORE, bi["field"]).token(bi["normalized"]), bi["token"])
        fts = VECTORS["fts_term"]
        self.assertEqual(seal.BlindIndex.from_root(ROOT, STORE, fts["field"]).term(fts["term"]), fts["token"])
        spec = VECTORS["tokenizer"]
        index = seal.BlindIndex.from_root(ROOT, spec["store"], spec["field"])
        for case in spec["cases"]:
            terms = seal.tokenize(case["text"])
            self.assertEqual(terms, case["terms"])
            self.assertEqual([index.term(term) for term in terms], case["tokens"])

    def test_legacy_and_unknown_versions(self) -> None:
        cipher = seal.ContentCipher.from_root(ROOT, STORE)
        ctx = seal.context("t.c", ("id", "1"))
        self.assertEqual(cipher.open_text("plain", ctx), seal.Opened("plain", True))
        self.assertEqual(cipher.open_text("", ctx), seal.Opened("", False))
        with self.assertRaises(seal.ContentSealError):
            cipher.open_text("encb:v2:AAAA", ctx)
        with self.assertRaises(seal.ContentSealError):
            cipher.seal_bytes(b"RYU-SF01 nope", ctx)


class KeyCustodyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_master_key_derives_the_core_plugin_subkey(self) -> None:
        master = base64.b64encode(bytes([7]) * 32).decode()
        keys = seal.capture_plugin_keys(
            "@ryu/forms", {seal.MASTER_KEY_ENV: master}, under_core=True, data_dir=self.dir
        )
        # Shared with crates/core/crypto process_keys.rs and the TS seal tests.
        self.assertEqual(base64.b64encode(keys.root).decode(), "9mkpBp6aF4R2Ik7g9laiJVWhDFYQUz6E5u9jB+Lak9w=")

    def test_delivered_key_scope_is_checked(self) -> None:
        key = base64.b64encode(bytes([1]) * 32).decode()
        with self.assertRaises(seal.SealError):
            seal.capture_plugin_keys(
                "@ryu/research",
                {seal.SEAL_KEY_ENV: key, seal.SEAL_SCOPE_ENV: "plugin:@ryu/mail"},
                under_core=True,
                data_dir=self.dir,
            )
        keys = seal.capture_plugin_keys(
            "@ryu/research",
            {seal.SEAL_KEY_ENV: key, seal.SEAL_SCOPE_ENV: "plugin:@ryu/research"},
            under_core=True,
            data_dir=self.dir,
        )
        self.assertEqual(keys.source, "delivered")

    def test_compat_under_core_and_standalone_root_file(self) -> None:
        compat = seal.capture_plugin_keys("@ryu/research", {}, under_core=True, data_dir=self.dir)
        self.assertFalse(compat.can_seal)
        self.assertEqual(compat.store_key_state(None, False).state, "compat")
        self.assertEqual(compat.store_key_state(None, True).state, "no_key")
        first = seal.capture_plugin_keys("@ryu/research", {}, under_core=False, data_dir=self.dir)
        root_file = self.dir / "secrets" / seal.STANDALONE_ROOT_FILE
        self.assertEqual(root_file.stat().st_mode & 0o777, 0o600)
        again = seal.capture_plugin_keys("@ryu/research", {}, under_core=False, data_dir=self.dir)
        self.assertEqual(first.kcv, again.kcv)
        master = base64.b64encode(bytes([9]) * 32).decode()
        with self.assertRaises(seal.SealError):
            seal.capture_plugin_keys(
                "@ryu/research", {seal.MASTER_KEY_ENV: master}, under_core=False, data_dir=self.dir
            )

    def test_key_variables_are_removed_at_import(self) -> None:
        code = (
            "import os, sys; sys.path.insert(0, sys.argv[1]);"
            "from ryu_research import ryu_content_seal as s;"
            "print(sorted(n for n in s.SEAL_ENV_VARS if n in os.environ), bool(s._PROCESS_KEY_ENV))"
        )
        env = {
            **os.environ,
            seal.SEAL_KEY_ENV: base64.b64encode(bytes(32)).decode(),
            seal.SEAL_SCOPE_ENV: "plugin:@ryu/research",
        }
        output = subprocess.run(
            [sys.executable, "-c", code, str(Path(__file__).resolve().parents[1])],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        self.assertEqual(output, "[] True")


class StoreSealTests(unittest.TestCase):
    COLUMN = seal.SealedColumn("notes", "body", "id", ("id",))

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "t.db"
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.execute("CREATE TABLE notes (id TEXT PRIMARY KEY, body TEXT NOT NULL DEFAULT '')")
        self.db.commit()

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def _keys(self, fill: int) -> seal.PluginKeys:
        return seal.PluginKeys("@ryu/test", bytes([fill]) * 32, "delivered")

    def test_migration_is_idempotent_and_scrubs_plaintext(self) -> None:
        self.db.execute("INSERT INTO notes VALUES ('a', 'legacy secret body'), ('b', '')")
        self.db.commit()
        store = seal.StoreSeal.open(self.db, self.path, "t.db", self._keys(3), [self.COLUMN])
        body = self.db.execute("SELECT body FROM notes WHERE id = 'a'").fetchone()[0]
        self.assertTrue(body.startswith(seal.TEXT_PREFIX))
        self.assertEqual(store.open_text(self.COLUMN, {"id": "a"}, body), "legacy secret body")
        self.assertEqual(self.db.execute("SELECT body FROM notes WHERE id = 'b'").fetchone()[0], "")
        self.assertEqual(store.migrate(self.db, self.path), {"resealed": 0, "unreadable": 0})
        self.db.close()
        raw = self.path.read_bytes()
        wal = Path(f"{self.path}-wal")
        if wal.exists():
            raw += wal.read_bytes()
        self.assertNotIn(b"legacy secret body", raw)
        self.db = sqlite3.connect(self.path)

    def test_mismatch_refuses_writes_and_aad_swap_fails(self) -> None:
        store = seal.StoreSeal.open(self.db, self.path, "t.db", self._keys(3), [self.COLUMN])
        sealed = store.seal_text(self.COLUMN, {"id": "a"}, "bound")
        with self.assertRaises(seal.ContentSealError):
            store.open_text(self.COLUMN, {"id": "b"}, sealed)
        other = seal.StoreSeal.open(self.db, self.path, "t.db", self._keys(4), [self.COLUMN])
        self.assertEqual(other.health()["sealing"], "key mismatch")
        with self.assertRaises(seal.SealUnavailableError):
            other.seal_text(self.COLUMN, {"id": "a"}, "x")

    def test_a_store_sealed_under_a_retired_key_opens_and_is_resealed(self) -> None:
        old = seal.StoreSeal.open(self.db, self.path, "t.db", self._keys(3), [self.COLUMN])
        self.db.execute(
            "INSERT INTO notes VALUES ('a', ?)", (old.seal_text(self.COLUMN, {"id": "a"}, "old body"),)
        )
        self.db.commit()
        # A backup import or `keys adopt` retired key 3; Core delivers key 4
        # with key 3's subkey in RYU_SEAL_RETIRED_KEYS.
        keys = seal.capture_plugin_keys(
            "@ryu/test",
            {
                seal.SEAL_KEY_ENV: base64.b64encode(bytes([4]) * 32).decode(),
                seal.SEAL_SCOPE_ENV: "plugin:@ryu/test",
                seal.SEAL_RETIRED_KEYS_ENV: ",".join(
                    base64.b64encode(bytes([fill]) * 32).decode() for fill in (9, 3)
                ),
            },
            under_core=True,
            data_dir=Path(self.tmp.name),
        )
        store = seal.StoreSeal.open(self.db, self.path, "t.db", keys, [self.COLUMN])
        self.assertEqual(store.status.state, "retired")
        self.assertEqual(store.health()["sealing"], "ok")
        body = self.db.execute("SELECT body FROM notes WHERE id = 'a'").fetchone()[0]
        self.assertEqual(store.open_text(self.COLUMN, {"id": "a"}, body), "old body")
        store.seal_text(self.COLUMN, {"id": "b"}, "new body")
        # Resealed under key 4, which is now recorded: no retired key needed.
        current = seal.StoreSeal.open(self.db, self.path, "t.db", self._keys(4), [self.COLUMN])
        self.assertEqual(current.status.state, "current")
        body = self.db.execute("SELECT body FROM notes WHERE id = 'a'").fetchone()[0]
        self.assertEqual(current.open_text(self.COLUMN, {"id": "a"}, body), "old body")

    def test_a_retired_store_with_unreadable_values_keeps_the_retired_kcv(self) -> None:
        old = seal.StoreSeal.open(self.db, self.path, "t.db", self._keys(3), [self.COLUMN])
        self.db.execute(
            "INSERT INTO notes VALUES ('a', ?), ('b', ?)",
            (
                old.seal_text(self.COLUMN, {"id": "a"}, "opens"),
                old.seal_text(self.COLUMN, {"id": "elsewhere"}, "does not open"),
            ),
        )
        self.db.commit()
        rotated = seal.PluginKeys("@ryu/test", bytes([4]) * 32, "delivered", (bytes([3]) * 32,))
        store = seal.StoreSeal.open(self.db, self.path, "t.db", rotated, [self.COLUMN])
        self.assertEqual(store.unreadable_rows, 1)
        without = seal.StoreSeal.open(self.db, self.path, "t.db", self._keys(4), [self.COLUMN])
        self.assertEqual(without.status.state, "mismatch")

    def test_plaintext_written_after_a_rollback_is_sealed_on_the_next_open(self) -> None:
        self.db.execute("INSERT INTO notes VALUES ('a', 'first')")
        self.db.commit()
        seal.StoreSeal.open(self.db, self.path, "t.db", self._keys(3), [self.COLUMN])
        # The older build writes plaintext and ignores the seal table.
        self.db.execute("INSERT INTO notes VALUES ('b', 'SECRET-DOWNGRADE'), ('c', '')")
        self.db.commit()
        store = seal.StoreSeal.open(self.db, self.path, "t.db", self._keys(3), [self.COLUMN])
        body = self.db.execute("SELECT body FROM notes WHERE id = 'b'").fetchone()[0]
        self.assertTrue(body.startswith(seal.TEXT_PREFIX))
        self.assertEqual(store.open_text(self.COLUMN, {"id": "b"}, body), "SECRET-DOWNGRADE")
        self.db.close()
        self.assertNotIn(b"SECRET-DOWNGRADE", self._raw())
        self.db = sqlite3.connect(self.path)

    def test_a_newer_min_reader_is_refused_on_open(self) -> None:
        seal.StoreSeal.open(self.db, self.path, "t.db", self._keys(3), [self.COLUMN])
        self.db.execute("UPDATE ryu_content_seal SET min_reader = 3 WHERE store = 't.db'")
        self.db.commit()
        with self.assertRaises(seal.NewerStoreFormatError):
            seal.StoreSeal.open(self.db, self.path, "t.db", self._keys(3), [self.COLUMN])
        with self.assertRaisesRegex(seal.NewerStoreFormatError, "written by a newer Ryu"):
            seal.StoreSeal.open(self.db, self.path, "t.db", None, [self.COLUMN])

    def test_value_that_looks_sealed_is_counted_not_resealed(self) -> None:
        self.db.execute("INSERT INTO notes VALUES ('a', 'encb:v1:AAAA')")
        self.db.commit()
        store = seal.StoreSeal.open(self.db, self.path, "t.db", self._keys(3), [self.COLUMN])
        self.assertEqual(store.unreadable_rows, 1)
        self.assertEqual(self.db.execute("SELECT body FROM notes").fetchone()[0], "encb:v1:AAAA")


    def _raw(self) -> bytes:
        raw = self.path.read_bytes()
        wal = Path(f"{self.path}-wal")
        if wal.exists():
            raw += wal.read_bytes()
        return raw

    def test_rows_deleted_before_the_first_key_are_scrubbed_with_nothing_to_reseal(self) -> None:
        # An older version: no secure_delete, so the deleted bytes stay.
        self.db.execute("PRAGMA secure_delete = OFF")
        for index in range(40):
            self.db.execute(
                "INSERT INTO notes VALUES (?, ?)",
                (f"gone-{index}", f"deleted-plaintext-{index}-" + "z" * 900),
            )
        self.db.commit()
        self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.db.execute("DELETE FROM notes")
        self.db.commit()
        self.assertIn(b"deleted-plaintext-7-", self._raw())
        store = seal.StoreSeal.open(self.db, self.path, "t.db", self._keys(3), [self.COLUMN])
        self.assertFalse(store.health()["vacuum_pending"])
        self.db.close()
        self.assertNotIn(b"deleted-plaintext-", self._raw())
        self.db = sqlite3.connect(self.path)

    def test_free_pages_are_zeroed_even_when_vacuum_is_skipped(self) -> None:
        self.db.execute("PRAGMA secure_delete = OFF")
        # Long enough to spill into overflow pages, which a delete frees.
        # (Fragments inside pages still in use wait for the VACUUM.)
        self.db.execute(
            "INSERT INTO notes VALUES ('a', ?)", ("y" * 4000 + "freed-secret-" + "y" * 20000,)
        )
        self.db.commit()
        self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.db.execute("DELETE FROM notes")
        self.db.commit()
        with patch.object(seal, "_VACUUM_HEADROOM", 1 << 62):
            store = seal.StoreSeal.open(self.db, self.path, "t.db", self._keys(3), [self.COLUMN])
        self.assertTrue(store.health()["vacuum_pending"])
        self.db.close()
        self.assertNotIn(b"freed-secret-", self._raw())
        self.db = sqlite3.connect(self.path)

    def _store_state(self) -> str | None:
        row = self.db.execute(
            "SELECT state FROM ryu_content_seal WHERE store = 't.db' AND item = '__store__'"
        ).fetchone()
        return row[0] if row is not None else None

    def _free_plaintext(self, marker: str) -> int:
        """Delete overflow-sized plaintext without secure_delete; return the page count."""
        self.db.execute("PRAGMA secure_delete = OFF")
        self.db.execute("INSERT INTO notes VALUES ('gone', ?)", ("y" * 4000 + marker + "y" * 40000,))
        self.db.commit()
        self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.db.execute("DELETE FROM notes")
        self.db.commit()
        self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return int(self.db.execute("PRAGMA page_count").fetchone()[0])

    def test_a_failing_scrub_never_fails_the_open_and_stays_pending(self) -> None:
        pages = self._free_plaintext("scrub-fails-")
        # The zero fill needs a few pages more than the freelist holds.
        self.db.execute(f"PRAGMA max_page_count = {pages}")
        store = seal.StoreSeal.open(self.db, self.path, "t.db", self._keys(3), [self.COLUMN])
        self.assertTrue(store.health()["vacuum_pending"])
        self.assertEqual(self._store_state(), "vacuum_pending")
        self.assertIsNone(
            self.db.execute("SELECT 1 FROM sqlite_master WHERE name = 'ryu_content_seal_scrub'").fetchone()
        )
        self.db.execute("PRAGMA max_page_count = 1073741823")
        retried = seal.StoreSeal.open(self.db, self.path, "t.db", self._keys(3), [self.COLUMN])
        self.assertFalse(retried.health()["vacuum_pending"])
        self.assertEqual(self._store_state(), "sealed")
        self.db.close()
        self.assertNotIn(b"scrub-fails-", self._raw())
        self.db = sqlite3.connect(self.path)

    def test_scrub_waits_for_space_and_is_not_repeated_while_vacuum_is_deferred(self) -> None:
        self._free_plaintext("scrub-space-")
        with patch.object(seal, "_free_space", lambda _dir: 0):
            store = seal.StoreSeal.open(self.db, self.path, "t.db", self._keys(3), [self.COLUMN])
        self.assertTrue(store.health()["vacuum_pending"])
        self.assertEqual(self._store_state(), "vacuum_pending")
        self.assertIn(b"scrub-space-", self._raw())

        # Room for the scrub (first check) but not for the VACUUM (second).
        answers = [1 << 62, 0]
        with patch.object(seal, "_free_space", lambda _dir: answers.pop(0) if answers else 0):
            seal.StoreSeal.open(self.db, self.path, "t.db", self._keys(3), [self.COLUMN])
        self.assertEqual(self._store_state(), "scrubbed")
        self.assertNotIn(b"scrub-space-", self._raw())

        # A later start with the VACUUM still deferred only checks for VACUUM room.
        checks: list[Path] = []
        with patch.object(seal, "_free_space", lambda directory: checks.append(directory) or 0):
            deferred = seal.StoreSeal.open(self.db, self.path, "t.db", self._keys(3), [self.COLUMN])
        self.assertEqual(len(checks), 1)
        self.assertTrue(deferred.health()["vacuum_pending"])
        self.assertEqual(self._store_state(), "scrubbed")

        with patch.object(seal, "_free_space", lambda _dir: 1 << 62):
            done = seal.StoreSeal.open(self.db, self.path, "t.db", self._keys(3), [self.COLUMN])
        self.assertFalse(done.health()["vacuum_pending"])
        self.assertEqual(self._store_state(), "sealed")

    def test_compat_refuses_lookalikes_and_a_stored_lookalike_does_not_lock(self) -> None:
        compat_keys = seal.capture_plugin_keys(
            "@ryu/test", {}, under_core=True, data_dir=Path(self.tmp.name)
        )
        store = seal.StoreSeal.open(self.db, self.path, "t.db", compat_keys, [self.COLUMN])
        with self.assertRaises(seal.ContentSealError) as raised:
            store.seal_text(self.COLUMN, {"id": "a"}, "encb:v1:not-really-sealed")
        self.assertEqual(raised.exception.code, "already_a_ryu_envelope")
        self.db.execute("INSERT INTO notes VALUES ('old', 'encb:v1:from-an-older-version')")
        self.db.commit()
        reopened = seal.StoreSeal.open(self.db, self.path, "t.db", compat_keys, [self.COLUMN])
        self.assertEqual(reopened.status.state, "compat")
        self.assertEqual(reopened.seal_text(self.COLUMN, {"id": "b"}, "plain"), "plain")


if __name__ == "__main__":
    unittest.main()
