"""Fail-closed OS confinement for agent-authored experiment code.

An experiment attempt executes a file the agent wrote. Approval authorizes one
bounded attempt; it does not make that code trusted. Every attempt therefore runs
behind an OS boundary that exposes only:

* a private, per-attempt run directory (the bound checkout plus its temp dir),
  read-write;
* a Research-private runtime directory for the workspace (its ``HOME``, XDG
  dirs and the uv/pip caches, tool and managed-Python dirs), read-write. Nothing
  outside a confined attempt ever reads or executes from it;
* the interpreter/runtime and system library locations, read-only;
* paths the (non-agent-writable) ``experiment.toml`` explicitly declares in
  ``sandbox_paths``, read-only, and never under the home directory or Ryu data.

Write access is never granted to a location that code outside the sandbox uses:
the user's shared package caches and tool dirs (for example ``~/.cache/uv`` or
``~/.local/share/uv``, which Core and other plugins execute from), the
workspaces root (whose git hooks the engine runs), the engine's own code and
interpreter, or system locations. Otherwise a confined attempt could plant code
that later runs unconfined and reads the secrets the sandbox hides.

The Research data directory, the workspaces root (engine token and campaign
database), the user's home directory and Ryu's data directory are never granted.
A seccomp filter denies every ``AF_UNIX`` socket (so the session bus, keyring
and local relays are unreachable) and ``io_uring``; network sockets are denied
too unless the experiment declares ``sandbox_network``. Even with network,
``bind()`` and ``listen()`` are refused, so an attempt can only open outgoing
connections and never squat a loopback port that Core, the Research backend or
a sidecar later sends credentials to. ``setsid()`` and ``setpgid()`` are refused
too, so every process an attempt starts stays in the attempt's process group
and the engine's ``killpg`` ends all of them when the attempt finishes. That
filter does not depend on the Landlock ABI version.

Platforms: Linux only. Landlock (unprivileged, kernel 5.13+) plus seccomp on
x86_64 or aarch64. This file doubles as the tiny trusted launcher: it is
executed with ``python -I`` before any agent code, applies the restrictions to
itself, then ``exec``s the experiment command, which inherits them. Every other
platform, architecture or kernel refuses the attempt; the module never falls
back to running the command unconfined.

This file must stay standard-library only and free of package-relative imports,
because the launcher runs it as a standalone script.
"""

from __future__ import annotations

import ctypes
import errno
import glob
import json
import os
import shutil
import stat
import sys
from pathlib import Path

LAUNCHER_FLAG = "--ryu-research-landlock"

# Folder, beside the workspaces root inside the Research data dir, that holds
# one private runtime directory per workspace.
RUNTIME_DIR_NAME = "sandbox-runtime"

# Landlock ABI constants (include/uapi/linux/landlock.h). The syscall numbers are
# shared by every architecture that implements Landlock.
_SYS_LANDLOCK_CREATE_RULESET = 444
_SYS_LANDLOCK_ADD_RULE = 445
_SYS_LANDLOCK_RESTRICT_SELF = 446
_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1
_PR_SET_NO_NEW_PRIVS = 38

_FS_EXECUTE = 1 << 0
_FS_WRITE_FILE = 1 << 1
_FS_READ_FILE = 1 << 2
_FS_READ_DIR = 1 << 3
_FS_REMOVE_DIR = 1 << 4
_FS_REMOVE_FILE = 1 << 5
_FS_MAKE_CHAR = 1 << 6
_FS_MAKE_DIR = 1 << 7
_FS_MAKE_REG = 1 << 8
_FS_MAKE_SOCK = 1 << 9
_FS_MAKE_FIFO = 1 << 10
_FS_MAKE_BLOCK = 1 << 11
_FS_MAKE_SYM = 1 << 12
_FS_REFER = 1 << 13  # ABI 2
_FS_TRUNCATE = 1 << 14  # ABI 3
_FS_IOCTL_DEV = 1 << 15  # ABI 5
_NET_BIND_TCP = 1 << 0  # ABI 4
_NET_CONNECT_TCP = 1 << 1  # ABI 4
_SCOPE_ABSTRACT_UNIX_SOCKET = 1 << 0  # ABI 6
_SCOPE_SIGNAL = 1 << 1  # ABI 6

_FILE_ONLY_RIGHTS = _FS_EXECUTE | _FS_WRITE_FILE | _FS_READ_FILE | _FS_TRUNCATE | _FS_IOCTL_DEV
_READ_RIGHTS = _FS_EXECUTE | _FS_READ_FILE | _FS_READ_DIR

# seccomp (include/uapi/linux/seccomp.h, filter.h, audit.h).
_PR_GET_SECCOMP = 21
_PR_SET_SECCOMP = 22
_SECCOMP_MODE_FILTER = 2
_SECCOMP_RET_ALLOW = 0x7FFF0000
_SECCOMP_RET_ERRNO = 0x00050000
_BPF_LD_W_ABS = 0x20
_BPF_JEQ_K = 0x15
_BPF_JGE_K = 0x35
_BPF_RET_K = 0x06
_SECCOMP_DATA_NR = 0
_SECCOMP_DATA_ARCH = 4
_SECCOMP_DATA_ARG0_LOW = 16  # little-endian low word of args[0]
_X32_SYSCALL_BIT = 0x40000000
_SYS_IO_URING_SETUP = 425  # same number on every architecture
_AF_INET = 2
_AF_INET6 = 10
_AF_NETLINK = 16
# machine -> AUDIT_ARCH_*, the syscall numbers the filter inspects, and whether
# the architecture also has an x32 ABI (arch/x86/entry/syscalls/syscall_64.tbl,
# include/uapi/asm-generic/unistd.h).
_SECCOMP_ARCHES = {
    "x86_64": {
        "audit_arch": 0xC000003E,
        "socket": 41,
        "bind": 49,
        "listen": 50,
        "setpgid": 109,
        "setsid": 112,
        "x32": True,
    },
    "aarch64": {
        "audit_arch": 0xC00000B7,
        "socket": 198,
        "bind": 200,
        "listen": 201,
        "setpgid": 154,
        "setsid": 157,
        "x32": False,
    },
}

# System locations an interpreter needs read-only. Missing entries are skipped.
_SYSTEM_READ_ROOTS = (
    "/usr",
    "/lib",
    "/lib32",
    "/lib64",
    "/libx32",
    "/bin",
    "/sbin",
    "/etc",
    "/sys",
    "/nix/store",
)
# Never all of /proc: another same-user process's /proc/<pid>/environ (Core's or
# the engine's) is readable through it. Only the confined process's own entry
# and a few global, credential-free files. "/proc/self" is resolved by the
# launcher itself, so it names the experiment's own pid.
_PROC_READ = (
    "/proc/self",
    "/proc/cpuinfo",
    "/proc/meminfo",
    "/proc/stat",
    "/proc/loadavg",
    "/proc/uptime",
    "/proc/filesystems",
    "/proc/sys/kernel",
    "/proc/sys/vm",
    "/proc/driver/nvidia",
)
_DEVICE_FILES = ("/dev/null", "/dev/zero", "/dev/full", "/dev/random", "/dev/urandom")
# GPU device nodes only. /dev/shm is deliberately absent: it is shared with
# every same-user process.
_DEVICE_GLOBS = ("/dev/nvidia*", "/dev/dri")

# The engine package (and this launcher) execute unconfined; never writable.
_ENGINE_CODE_DIR = Path(__file__).resolve().parent.parent


class SandboxUnavailableError(RuntimeError):
    """Raised when confinement cannot be established; the attempt must not run."""


# ── policy ───────────────────────────────────────────────────────────────────


def _resolve(path: str | os.PathLike) -> Path:
    return Path(os.path.abspath(os.path.realpath(os.path.expanduser(str(path)))))


def _is_ancestor_or_same(candidate: Path, target: Path) -> bool:
    return candidate == target or candidate in target.parents


def _overlaps(first: Path, second: Path) -> bool:
    return _is_ancestor_or_same(first, second) or _is_ancestor_or_same(second, first)


def protected_paths(workspaces_root: Path) -> list[Path]:
    """Locations an experiment must never be able to read.

    Covers the workspaces root (engine bearer and campaign database), the
    Research plugin data dir, Ryu's data dir (Core's node bearer and keys) and the
    user's home directory as a whole.
    """
    candidates: list[Path] = [workspaces_root, Path.home(), Path.home() / ".ryu"]
    plugin_data = os.environ.get("RYU_PLUGIN_DATA_DIR")
    if plugin_data:
        candidates.append(Path(plugin_data))
        if Path(plugin_data).parent.name == "plugin-data":
            candidates.append(Path(plugin_data).parent.parent)
    ryu_dir = os.environ.get("RYU_DIR")
    if ryu_dir:
        candidates.append(Path(ryu_dir))
    parent = workspaces_root.parent
    if parent.parent.name == "plugin-data":
        candidates.extend([parent, parent.parent.parent])
    out: list[Path] = []
    for candidate in candidates:
        resolved = _resolve(candidate)
        if resolved not in out:
            out.append(resolved)
    return out


def runtime_root(workspaces_root: Path) -> Path:
    """Research-owned folder holding every workspace's private runtime dir.

    It sits beside the workspaces root in the Research data dir, never inside
    it: the engine runs git (and its hooks) unconfined in the workspaces.
    """
    return _resolve(workspaces_root).parent / RUNTIME_DIR_NAME


def workspace_runtime_dir(workspaces_root: Path, workspace_id: str) -> Path:
    """The private, persistent runtime dir of one workspace (one campaign)."""
    if workspace_id in {"", ".", ".."} or "/" in workspace_id or "\\" in workspace_id:
        raise SandboxUnavailableError(f"invalid workspace id {workspace_id!r}")
    return runtime_root(workspaces_root) / workspace_id


def _runtime_layout(runtime: Path) -> dict[str, Path]:
    home = runtime / "home"
    cache = runtime / "cache"
    data = runtime / "data"
    return {
        "HOME": home,
        "XDG_CACHE_HOME": cache,
        "XDG_DATA_HOME": data,
        "XDG_CONFIG_HOME": runtime / "config",
        "XDG_STATE_HOME": runtime / "state",
        "UV_CACHE_DIR": cache / "uv",
        "UV_TOOL_DIR": data / "uv" / "tools",
        "UV_TOOL_BIN_DIR": home / ".local" / "bin",
        "UV_PYTHON_INSTALL_DIR": data / "uv" / "python",
        "UV_PYTHON_BIN_DIR": home / ".local" / "bin",
        "PIP_CACHE_DIR": cache / "pip",
        "PYTHONUSERBASE": home / ".local",
    }


_DIR_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(
    os, "O_CLOEXEC", 0
)


def _private_subdir(parent_fd: int, name: str) -> int:
    """Return an fd for ``name`` under ``parent_fd``: a real directory, mode 0700.

    Runs unconfined, on a tree that earlier confined attempts could write. It
    never follows a symlink: an existing symlink or other non-directory entry is
    removed and replaced by a fresh directory, and the final open uses
    ``O_NOFOLLOW`` so a racing swap fails instead of escaping.
    """
    if name in {"", ".", ".."} or "/" in name:
        raise SandboxUnavailableError(f"invalid runtime path component {name!r}")
    try:
        existing = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        existing = None
    if existing is not None and not stat.S_ISDIR(existing.st_mode):
        os.unlink(name, dir_fd=parent_fd)
        existing = None
    if existing is None:
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
    try:
        fd = os.open(name, _DIR_OPEN_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        raise SandboxUnavailableError(
            f"runtime directory component {name!r} is not a private directory: {exc}"
        ) from exc
    os.fchmod(fd, 0o700)
    return fd


def prepare_runtime_dir(runtime: Path) -> dict[str, str]:
    """Create the workspace's private runtime dir and return its environment.

    The returned variables point the experiment's home, XDG dirs and every
    uv/pip cache, tool and managed-Python location into the private dir, so a
    package manager never touches the user's shared caches. The confinement
    makes that binding, not the environment: shared locations stay unwritable
    even if the experiment ignores these variables.

    Earlier confined attempts of the same campaign could write this tree, so it
    is built relative to directory fds without following any symlink they
    planted: this unconfined step never creates anything outside it.
    """
    runtime = Path(os.path.abspath(runtime))
    root = runtime.parent
    # The runtime root is Research-owned and never writable by an attempt.
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        root_fd = os.open(str(root), _DIR_OPEN_FLAGS)
    except OSError as exc:
        raise SandboxUnavailableError(f"runtime root {str(root)!r} is unusable: {exc}") from exc
    try:
        runtime_fd = _private_subdir(root_fd, runtime.name)
    finally:
        os.close(root_fd)
    layout = _runtime_layout(runtime)
    try:
        for path in layout.values():
            fd = os.dup(runtime_fd)
            try:
                for component in path.relative_to(runtime).parts:
                    child = _private_subdir(fd, component)
                    os.close(fd)
                    fd = child
            finally:
                os.close(fd)
    finally:
        os.close(runtime_fd)
    return {name: str(path) for name, path in layout.items()}


def _runtime_read_paths() -> list[Path]:
    """Interpreter prefixes and import roots the experiment needs read-only."""
    out: list[Path] = []
    for value in (
        sys.prefix,
        sys.base_prefix,
        sys.exec_prefix,
        sys.base_exec_prefix,
        os.path.dirname(sys.executable),
    ):
        if value:
            out.append(Path(value))
    for entry in sys.path:
        if entry and os.path.isabs(entry) and "site-packages" in entry:
            out.append(Path(entry))
    return out


def _never_writable(workspaces_root: Path) -> list[Path]:
    """Locations whose contents run, or are trusted, outside the sandbox."""
    out = [
        _resolve(workspaces_root),
        _ENGINE_CODE_DIR,
        Path.home() / ".local",
        Path.home() / ".cache",
    ]
    out.extend(_resolve(path) for path in _runtime_read_paths())
    out.extend(_resolve(root) for root in _SYSTEM_READ_ROOTS)
    return [_resolve(path) for path in out]


def build_policy(
    run_root: Path,
    argv: list[str],
    *,
    workspaces_root: Path,
    runtime_dir: Path | None = None,
    declared_paths: list[str] | None = None,
    network: bool = False,
) -> dict:
    """Compute the explicit grant set for one attempt.

    Only ``run_root`` and the workspace's Research-private ``runtime_dir`` are
    writable. Raises :class:`SandboxUnavailableError` when a grant would expose
    a protected location or make shared, unconfined code writable, so the caller
    refuses the attempt instead of widening.
    """
    protected = protected_paths(workspaces_root)
    never_writable = _never_writable(workspaces_root)

    def checked(path: Path, label: str) -> Path:
        resolved = _resolve(path)
        for secret in protected:
            if _is_ancestor_or_same(resolved, secret):
                raise SandboxUnavailableError(
                    f"{label} {str(path)!r} would expose protected data to the experiment"
                )
        return resolved

    def checked_write(path: Path, label: str) -> Path:
        resolved = checked(path, label)
        for shared in never_writable:
            if _overlaps(resolved, shared):
                raise SandboxUnavailableError(
                    f"{label} {str(path)!r} overlaps {str(shared)!r}, which runs outside "
                    "the sandbox; refusing to make it writable"
                )
        return resolved

    write: list[str] = [str(checked_write(run_root, "attempt run directory"))]
    if runtime_dir is not None:
        resolved_runtime = checked_write(runtime_dir, "runtime directory")
        if runtime_root(workspaces_root) not in resolved_runtime.parents:
            raise SandboxUnavailableError(
                "the runtime directory must be a Research-private workspace runtime dir"
            )
        write.append(str(resolved_runtime))

    read: list[str] = []
    for root in _SYSTEM_READ_ROOTS:
        if os.path.exists(root):
            read.append(str(checked(Path(root), "system path")))
    for root in _runtime_read_paths():
        if root.exists():
            try:
                read.append(str(checked(root, "runtime path")))
            except SandboxUnavailableError:
                # A site-packages dir colocated with protected data is skipped,
                # never granted; the interpreter prefix itself is checked below.
                continue
    prefix = checked(Path(sys.base_prefix), "interpreter prefix")
    if str(prefix) not in read:
        read.append(str(prefix))

    # Declared paths are read-only and may not reach into home, Ryu data, the
    # workspaces or another workspace's runtime dir, in either direction.
    no_declared_reach = [*protected, _resolve(workspaces_root), runtime_root(workspaces_root)]
    for declared in declared_paths or []:
        path = Path(os.path.expanduser(str(declared)))
        if not path.is_absolute():
            raise SandboxUnavailableError(
                f"sandbox path {declared!r} must be absolute or start with ~"
            )
        resolved = _resolve(path)
        for location in no_declared_reach:
            if _overlaps(resolved, location):
                raise SandboxUnavailableError(
                    f"declared sandbox path {declared!r} overlaps {str(location)!r}; "
                    "declared paths may not reach the home directory or Ryu data"
                )
        if resolved.exists():
            read.append(str(resolved))

    executables: list[str] = []
    if argv:
        found = shutil.which(argv[0]) or argv[0]
        if os.path.isabs(found) and os.path.exists(found):
            executables.append(str(checked(Path(found), "experiment executable")))
            # Keep the unresolved (symlink) path reachable too.
            if os.path.abspath(found) != executables[-1]:
                executables.append(os.path.abspath(found))

    devices: list[str] = [device for device in _DEVICE_FILES if os.path.exists(device)]
    for pattern in _DEVICE_GLOBS:
        devices.extend(sorted(glob.glob(pattern)))

    return {
        "read": _dedupe(read),
        "exec_files": _dedupe(executables),
        "write": _dedupe(write),
        "devices": _dedupe(devices),
        "proc": [path for path in _PROC_READ if os.path.exists(path)],
        "network": bool(network),
    }


def _dedupe(values: list[str]) -> list[str]:
    seen: list[str] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


# ── platform availability + command construction ────────────────────────────


def _libc() -> ctypes.CDLL:
    return ctypes.CDLL(None, use_errno=True)


def landlock_abi() -> int:
    """Return the kernel's Landlock ABI version, or 0 when unavailable."""
    if not sys.platform.startswith("linux"):
        return 0
    try:
        libc = _libc()
        libc.syscall.restype = ctypes.c_long
        version = libc.syscall(
            ctypes.c_long(_SYS_LANDLOCK_CREATE_RULESET),
            None,
            ctypes.c_size_t(0),
            ctypes.c_uint32(_LANDLOCK_CREATE_RULESET_VERSION),
        )
    except (AttributeError, OSError):
        return 0
    return int(version) if version > 0 else 0


def _seccomp_arch() -> dict | None:
    return _SECCOMP_ARCHES.get(os.uname().machine) if hasattr(os, "uname") else None


def seccomp_supported() -> bool:
    """Whether the socket filter can be installed on this kernel and architecture."""
    if not sys.platform.startswith("linux") or _seccomp_arch() is None:
        return False
    try:
        libc = _libc()
        return libc.prctl(ctypes.c_int(_PR_GET_SECCOMP), 0, 0, 0, 0) >= 0
    except (AttributeError, OSError):
        return False


def supported() -> bool:
    """Whether attempts can run confined on this host."""
    return landlock_abi() >= 1 and seccomp_supported()


def ensure_available() -> None:
    """Raise :class:`SandboxUnavailableError` unless confinement can be applied."""
    if not sys.platform.startswith("linux"):
        raise SandboxUnavailableError(
            f"Research experiments can only be confined on Linux, not {sys.platform!r}; "
            "refusing to run agent-authored code unconfined"
        )
    if landlock_abi() < 1:
        raise SandboxUnavailableError(
            "Research experiments require Linux Landlock (kernel 5.13 or newer "
            "with Landlock enabled); refusing to run agent-authored code unconfined"
        )
    if not seccomp_supported():
        raise SandboxUnavailableError(
            "Research experiments require seccomp on x86_64 or aarch64; refusing to "
            "run agent-authored code unconfined"
        )


def confined_command(argv: list[str], policy: dict) -> list[str]:
    """Wrap ``argv`` so it starts only inside the confinement described by ``policy``."""
    ensure_available()
    return [
        sys.executable,
        "-I",
        str(Path(__file__).resolve()),
        LAUNCHER_FLAG,
        json.dumps(policy, separators=(",", ":")),
        "--",
        *argv,
    ]


# ── Linux launcher (runs as a standalone script before any agent code) ───────


class _RulesetAttr(ctypes.Structure):
    _fields_ = [
        ("handled_access_fs", ctypes.c_uint64),
        ("handled_access_net", ctypes.c_uint64),
        ("scoped", ctypes.c_uint64),
    ]


class _PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


class _SockFilter(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint16),
        ("jt", ctypes.c_uint8),
        ("jf", ctypes.c_uint8),
        ("k", ctypes.c_uint32),
    ]


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(_SockFilter))]


def _handled_fs(abi: int) -> int:
    handled = (1 << 13) - 1  # ABI 1: EXECUTE .. MAKE_SYM
    if abi >= 2:
        handled |= _FS_REFER
    if abi >= 3:
        handled |= _FS_TRUNCATE
    if abi >= 5:
        handled |= _FS_IOCTL_DEV
    return handled


def _add_path_rule(libc: ctypes.CDLL, ruleset_fd: int, path: str, access: int, handled: int) -> None:
    try:
        fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
    except OSError:
        return
    try:
        if not os.path.isdir(path):
            access &= _FILE_ONLY_RIGHTS
        access &= handled
        if access == 0:
            return
        attr = _PathBeneathAttr(allowed_access=access, parent_fd=fd)
        result = libc.syscall(
            ctypes.c_long(_SYS_LANDLOCK_ADD_RULE),
            ctypes.c_int(ruleset_fd),
            ctypes.c_int(_LANDLOCK_RULE_PATH_BENEATH),
            ctypes.byref(attr),
            ctypes.c_uint32(0),
        )
        if result != 0:
            err = ctypes.get_errno()
            raise OSError(err, f"landlock_add_rule({path}): {os.strerror(err)}")
    finally:
        os.close(fd)


def apply_landlock(policy: dict) -> None:
    """Restrict the current process (and every future child) to ``policy``."""
    abi = landlock_abi()
    if abi < 1:
        raise SandboxUnavailableError("Landlock is not available")
    libc = _libc()
    libc.syscall.restype = ctypes.c_long
    handled = _handled_fs(abi)
    attr = _RulesetAttr(handled_access_fs=handled, handled_access_net=0, scoped=0)
    size = 8
    if abi >= 4:
        # No net rule is ever added: binding a TCP port is always denied (an
        # attempt must not squat a loopback port that trusted clients send
        # credentials to), and outgoing connections are denied unless the
        # experiment declares network. The seccomp filter enforces the same
        # on kernels without Landlock networking.
        attr.handled_access_net = _NET_BIND_TCP
        if not policy.get("network"):
            attr.handled_access_net |= _NET_CONNECT_TCP
        size = 16
    if abi >= 6:
        attr.scoped = _SCOPE_ABSTRACT_UNIX_SOCKET | _SCOPE_SIGNAL
        size = 24
    ruleset_fd = libc.syscall(
        ctypes.c_long(_SYS_LANDLOCK_CREATE_RULESET),
        ctypes.byref(attr),
        ctypes.c_size_t(size),
        ctypes.c_uint32(0),
    )
    if ruleset_fd < 0:
        err = ctypes.get_errno()
        raise OSError(err, f"landlock_create_ruleset: {os.strerror(err)}")
    try:
        for path in [*policy.get("read", []), *policy.get("proc", [])]:
            _add_path_rule(libc, ruleset_fd, path, _READ_RIGHTS, handled)
        for path in policy.get("exec_files", []):
            _add_path_rule(libc, ruleset_fd, path, _FS_EXECUTE | _FS_READ_FILE, handled)
        for path in policy.get("devices", []):
            _add_path_rule(
                libc,
                ruleset_fd,
                path,
                _FS_READ_FILE | _FS_WRITE_FILE | _FS_READ_DIR | _FS_IOCTL_DEV | _FS_TRUNCATE,
                handled,
            )
        for path in policy.get("write", []):
            _add_path_rule(libc, ruleset_fd, path, handled, handled)
        if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            err = ctypes.get_errno()
            raise OSError(err, f"prctl(NO_NEW_PRIVS): {os.strerror(err)}")
        if (
            libc.syscall(
                ctypes.c_long(_SYS_LANDLOCK_RESTRICT_SELF),
                ctypes.c_int(ruleset_fd),
                ctypes.c_uint32(0),
            )
            != 0
        ):
            err = ctypes.get_errno()
            raise OSError(err, f"landlock_restrict_self: {os.strerror(err)}")
    finally:
        os.close(ruleset_fd)


def _bpf(code: int, k: int, jt: int = 0, jf: int = 0) -> tuple[int, int, int, int]:
    return (code, jt, jf, k)


def seccomp_program(network: bool, machine: str | None = None) -> list[tuple[int, int, int, int]]:
    """The socket filter as ``(code, jt, jf, k)`` BPF instructions.

    * any syscall from a foreign ABI (i386, x32, arm32) fails with EPERM, so the
      socket rule cannot be sidestepped through a compat syscall table;
    * ``io_uring_setup`` fails with ENOSYS (io_uring can open sockets itself);
    * ``bind()`` and ``listen()`` fail with EACCES on every socket, network or
      not, so an attempt can never accept a connection: it cannot squat the
      loopback port of the Research engine or a sidecar and capture the bearer
      a trusted client sends there. Outgoing ``connect()`` still works;
    * ``setsid()`` and ``setpgid()`` fail with EPERM, so no descendant can leave
      the attempt's process group and outlive the ``killpg`` that ends it;
    * ``socket()`` is allowed only for ``AF_NETLINK``, plus ``AF_INET``/
      ``AF_INET6`` when the experiment declares network. ``AF_UNIX`` is always
      refused with EACCES; ``socketpair()`` (anonymous) stays available.
    """
    arch = _SECCOMP_ARCHES.get(machine or os.uname().machine)
    if arch is None:
        raise SandboxUnavailableError("seccomp is unsupported on this architecture")
    audit_arch = arch["audit_arch"]
    socket_nr = arch["socket"]
    has_x32 = arch["x32"]
    allowed_families = [_AF_NETLINK]
    if network:
        allowed_families.extend([_AF_INET, _AF_INET6])
    eperm = _SECCOMP_RET_ERRNO | errno.EPERM
    program = [
        _bpf(_BPF_LD_W_ABS, _SECCOMP_DATA_ARCH),
        _bpf(_BPF_JEQ_K, audit_arch, jt=1, jf=0),
        _bpf(_BPF_RET_K, eperm),
        _bpf(_BPF_LD_W_ABS, _SECCOMP_DATA_NR),
    ]
    if has_x32:
        program += [
            _bpf(_BPF_JGE_K, _X32_SYSCALL_BIT, jt=0, jf=1),
            _bpf(_BPF_RET_K, eperm),
        ]
    denied = [
        (_SYS_IO_URING_SETUP, errno.ENOSYS),
        (arch["bind"], errno.EACCES),
        (arch["listen"], errno.EACCES),
        (arch["setsid"], errno.EPERM),
        (arch["setpgid"], errno.EPERM),
    ]
    for number, error in denied:
        program += [
            _bpf(_BPF_JEQ_K, number, jt=0, jf=1),
            _bpf(_BPF_RET_K, _SECCOMP_RET_ERRNO | error),
        ]
    program += [
        _bpf(_BPF_JEQ_K, socket_nr, jt=1, jf=0),
        _bpf(_BPF_RET_K, _SECCOMP_RET_ALLOW),
        _bpf(_BPF_LD_W_ABS, _SECCOMP_DATA_ARG0_LOW),
    ]
    for family in allowed_families:
        program += [
            _bpf(_BPF_JEQ_K, family, jt=0, jf=1),
            _bpf(_BPF_RET_K, _SECCOMP_RET_ALLOW),
        ]
    program.append(_bpf(_BPF_RET_K, _SECCOMP_RET_ERRNO | errno.EACCES))
    return program


def apply_seccomp(policy: dict) -> None:
    """Install the syscall filter on the current process (inherited across exec)."""
    program = seccomp_program(bool(policy.get("network")))
    filters = (_SockFilter * len(program))(
        *(_SockFilter(code=code, jt=jt, jf=jf, k=k) for code, jt, jf, k in program)
    )
    fprog = _SockFprog(len=len(program), filter=ctypes.cast(filters, ctypes.POINTER(_SockFilter)))
    libc = _libc()
    # NO_NEW_PRIVS is already set by apply_landlock; set it again so this
    # function is safe on its own.
    if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"prctl(NO_NEW_PRIVS): {os.strerror(err)}")
    result = libc.prctl(
        ctypes.c_int(_PR_SET_SECCOMP),
        ctypes.c_ulong(_SECCOMP_MODE_FILTER),
        ctypes.byref(fprog),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
    )
    if result != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"prctl(PR_SET_SECCOMP): {os.strerror(err)}")


def _launch(args: list[str]) -> int:
    if len(args) < 3 or args[0] != LAUNCHER_FLAG or "--" not in args:
        sys.stderr.write("ryu-research sandbox: invalid launcher arguments\n")
        return 126
    separator = args.index("--")
    try:
        policy = json.loads(args[1])
    except json.JSONDecodeError:
        sys.stderr.write("ryu-research sandbox: invalid policy\n")
        return 126
    command = args[separator + 1 :]
    if not command:
        sys.stderr.write("ryu-research sandbox: missing command\n")
        return 126
    try:
        apply_landlock(policy)
        apply_seccomp(policy)
    except (OSError, SandboxUnavailableError) as exc:
        sys.stderr.write(f"ryu-research sandbox: refusing to run unconfined: {exc}\n")
        return 126
    try:
        os.execvp(command[0], command)
    except OSError as exc:
        sys.stderr.write(f"ryu-research sandbox: failed to launch {command[0]!r}: {exc}\n")
        return 127
    return 127


if __name__ == "__main__":
    sys.exit(_launch(sys.argv[1:]))
