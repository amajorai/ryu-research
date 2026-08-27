"""Git-versioned experiment workspaces + the run/ledger machinery.

A workspace is a copy of one experiment kind's files under
``~/.ryu/research-workspaces/<uuid>/``, made a git repo so every attempt is a
commit an agent can keep (advance) or discard (reset). Running an attempt commits
the current state, executes the experiment's ``run_cmd`` under a wall-clock
budget, samples peak memory, and parses the single scalar metric from stdout.
"""

from __future__ import annotations

import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import TextIO

from . import campaigns
from .experiments import (
    ExperimentConfig,
    experiment_config_from_raw,
    load_experiment,
    load_toml,
)

# The ledger file kept inside each workspace. Tab-separated with a header row so
# it is both git-diffable and trivially parsed.
LEDGER_FILE = "results.tsv"
LEDGER_HEADER = ["commit", "score", "memory_gb", "status", "description"]

# Files copied into a workspace but never treated as experiment inputs.
_SKIP_NAMES = {".git", "__pycache__", LEDGER_FILE}
_RUN_LOCK = threading.Lock()
_PRIVATE_ENGINE_ENV = {
    "RYU_CORE_BEARER_TOKEN",
    "RYU_CORE_TOKEN",
    "RYU_EXT_TOKEN",
    "RYU_GATEWAY_TOKEN",
    "RYU_MASTER_KEY",
    "RYU_RESEARCH_ENGINE_TOKEN",
}
_STREAM_TAIL_CHARS = 64 * 1024


class _BoundedTextBuffer:
    """Thread-safe text tail whose memory use never exceeds ``max_chars``."""

    def __init__(self, max_chars: int) -> None:
        self._max_chars = max(1, int(max_chars))
        self._chunks: deque[str] = deque()
        self._length = 0
        self._lock = threading.Lock()

    def append(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            if len(text) >= self._max_chars:
                self._chunks.clear()
                tail = text[-self._max_chars :]
                self._chunks.append(tail)
                self._length = len(tail)
                return
            self._chunks.append(text)
            self._length += len(text)
            while self._length > self._max_chars and self._chunks:
                overflow = self._length - self._max_chars
                head = self._chunks[0]
                if len(head) <= overflow:
                    self._chunks.popleft()
                    self._length -= len(head)
                else:
                    self._chunks[0] = head[overflow:]
                    self._length -= overflow

    def text(self) -> str:
        with self._lock:
            return "".join(self._chunks)


class _StreamingMetric:
    """Keep the latest finite metric without retaining unbounded stdout."""

    def __init__(self, pattern: str) -> None:
        self._pattern = pattern
        self._window = _BoundedTextBuffer(8 * 1024)
        self._value: float | None = None
        self._lock = threading.Lock()

    def append(self, text: str) -> None:
        self._window.append(text)
        candidate = _parse_metric(self._pattern, self._window.text())
        if candidate is not None:
            with self._lock:
                self._value = candidate

    def value(self) -> float | None:
        with self._lock:
            return self._value


def workspaces_root() -> Path:
    """Root holding every workspace. Overridable via ``RESEARCH_WORKSPACES``
    (``RYU_RESEARCH_WORKSPACES`` honored too); defaults to
    ``~/.ryu/research-workspaces``."""
    override = os.environ.get("RESEARCH_WORKSPACES") or os.environ.get(
        "RYU_RESEARCH_WORKSPACES"
    )
    if override:
        return Path(override)
    ryu = os.environ.get("RYU_DIR")
    base = Path(ryu) if ryu else Path.home() / ".ryu"
    return base / "research-workspaces"


def _safe_rel(path: str) -> Path:
    """Resolve a caller-supplied relative path, rejecting traversal/absolute."""
    p = Path(path)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"unsafe path: {path!r}")
    return p


def workspace_dir(workspace_id: str) -> Path:
    # workspace ids are minted uuids; still guard against traversal.
    if not re.fullmatch(r"[A-Za-z0-9._-]+", workspace_id or ""):
        raise ValueError(f"invalid workspace id: {workspace_id!r}")
    return workspaces_root() / workspace_id


def _git(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=check,
        capture_output=True,
        text=True,
    )


def _git_head(cwd: Path) -> str:
    try:
        return _git(["rev-parse", "HEAD"], cwd).stdout.strip()
    except subprocess.CalledProcessError:
        return ""


# ── init ─────────────────────────────────────────────────────────────────────


def init_workspace(
    experiment: str,
    name: str = "",
    goal: str = "",
    reasoning_threshold_chars: int | None = None,
) -> dict:
    """Copy an experiment kind's files into a fresh workspace, git-init, and make
    the initial commit. Returns the init envelope."""
    cfg = load_experiment(experiment)
    workspace_id = uuid.uuid4().hex
    dest = workspace_dir(workspace_id)
    dest.mkdir(parents=True, exist_ok=True)

    # Copy every experiment file (skipping vcs/cache) into the workspace.
    for item in cfg.root.iterdir():
        if item.name in _SKIP_NAMES:
            continue
        target = dest / item.name
        if item.is_dir():
            shutil.copytree(item, target, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(item, target)

    # Seed the ledger with its header so appends stay well-formed.
    ledger = dest / LEDGER_FILE
    if not ledger.exists():
        ledger.write_text("\t".join(LEDGER_HEADER) + "\n", encoding="utf-8")

    # git init + deterministic identity + initial commit.
    _git(["init"], dest)
    _git(["config", "user.email", "research@ryu.local"], dest)
    _git(["config", "user.name", "Ryu Research"], dest)
    _exclude_runtime_history(dest)
    _git(["add", "-A"], dest)
    _git(["commit", "-m", f"init: {experiment}", "--allow-empty"], dest)

    campaign = campaigns.create_campaign(
        workspace_id,
        experiment,
        name=name.strip() or cfg.name,
        goal=goal,
        reasoning_threshold_chars=reasoning_threshold_chars,
    )

    return {
        "workspace_id": workspace_id,
        "campaign_id": workspace_id,
        "mutable_files": list(cfg.mutable_files),
        "program_md": cfg.program_md,
        "experiment": experiment,
        "campaign": campaign,
    }


def _experiment_for_workspace(dest: Path) -> ExperimentConfig | None:
    """Recover the experiment config for a workspace by matching its files back
    to a known kind (the workspace stores no explicit kind marker; we match on
    the presence of the kind's ``experiment.toml``)."""
    toml_path = dest / "experiment.toml"
    if not toml_path.is_file():
        return None
    # The copied experiment.toml is authoritative for run_cmd/metric/budget.
    raw = load_toml(toml_path)
    program_md = ""
    md = dest / "program.md"
    if md.is_file():
        program_md = md.read_text(encoding="utf-8")
    return experiment_config_from_raw(dest.name, raw, dest, program_md)


# ── files ────────────────────────────────────────────────────────────────────


def read_file(workspace_id: str, path: str) -> dict:
    dest = workspace_dir(workspace_id)
    rel = _safe_rel(path)
    content = (dest / rel).read_text(encoding="utf-8")
    return {"path": path, "content": content}


def _declared_mutable_files(dest: Path) -> set[Path]:
    """The workspace's declared ``mutable_files`` (from its copied
    ``experiment.toml``) as safe relative paths. Declared entries that are
    absolute or contain ``..`` are ignored rather than honored."""
    cfg = _experiment_for_workspace(dest)
    allowed: set[Path] = set()
    for name in cfg.mutable_files if cfg else []:
        try:
            allowed.add(_safe_rel(str(name)))
        except ValueError:
            continue
    return allowed


def write_file(workspace_id: str, path: str, content: str) -> dict:
    dest = workspace_dir(workspace_id)
    rel = _safe_rel(path)
    # SECURITY: only the experiment's DECLARED mutable_files are writable.
    # Anything else — above all `experiment.toml`, whose `run_cmd` is executed
    # verbatim by POST /workspace/{id}/run — is rejected (403 at the server),
    # closing the rewrite-run_cmd-then-run arbitrary-command path. Fail-closed:
    # a workspace with no recoverable config accepts no writes at all.
    if rel not in _declared_mutable_files(dest):
        raise PermissionError(
            f"path {path!r} is not in this workspace's declared mutable_files"
        )
    target = dest / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {"ok": True}


# ── memory sampling ──────────────────────────────────────────────────────────


def _process_tree_rss_kb(root_pid: int) -> int:
    """Return RSS for ``root_pid`` and every current descendant, in KiB."""
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,rss="],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return 0

    rows: dict[int, tuple[int, int]] = {}
    for line in result.stdout.splitlines():
        columns = line.split()
        if len(columns) < 3:
            continue
        try:
            pid, parent_pid, rss_kb = map(int, columns[:3])
        except ValueError:
            continue
        rows[pid] = (parent_pid, max(0, rss_kb))

    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (parent_pid, _) in rows.items():
            if pid not in descendants and parent_pid in descendants:
                descendants.add(pid)
                changed = True
    return sum(rows[pid][1] for pid in descendants if pid in rows)


def _sample_peak_rss_gb(pid: int, stop: threading.Event) -> float:
    """Poll the whole experiment process tree and return its peak RSS in GiB."""
    peak_kb = 0
    while not stop.is_set():
        peak_kb = max(peak_kb, _process_tree_rss_kb(pid))
        stop.wait(0.25)
    return peak_kb / 1024.0 / 1024.0


def _drain_stream(
    stream: TextIO,
    buffer: _BoundedTextBuffer,
    metric: _StreamingMetric | None = None,
) -> None:
    """Drain one child pipe continuously into bounded state."""
    try:
        while True:
            chunk = stream.read(4096)
            if not chunk:
                return
            buffer.append(chunk)
            if metric is not None:
                metric.append(chunk)
    finally:
        stream.close()


def _process_group_exists(group_id: int) -> bool:
    if os.name == "nt":
        return False
    try:
        os.killpg(group_id, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _terminate_process_tree(proc: subprocess.Popen[str], grace_s: float = 2.0) -> None:
    """Terminate every process launched for one experiment, then force-kill."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
        return

    group_id = proc.pid  # ``start_new_session=True`` makes pid == process-group id.
    try:
        os.killpg(group_id, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    deadline = time.monotonic() + max(0.0, grace_s)
    while time.monotonic() < deadline:
        if not _process_group_exists(group_id):
            return
        time.sleep(0.05)
    try:
        os.killpg(group_id, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        return


def _experiment_environment() -> dict[str, str]:
    """Inherit normal runtime settings without leaking Ryu host credentials."""
    return {
        name: value
        for name, value in os.environ.items()
        if name.upper() not in _PRIVATE_ENGINE_ENV
    }


# ── run ──────────────────────────────────────────────────────────────────────


def run_experiment(
    workspace_id: str,
    budget_s: int | None = None,
    hypothesis: str = "",
    description: str = "",
    proposal_id: int | None = None,
    analysis_id: int | None = None,
) -> dict:
    """Commit the current state, execute ``run_cmd`` under the budget, parse the
    metric, and return ``{score, memory_gb, status, commit, logs_tail}``."""
    if not _RUN_LOCK.acquire(blocking=False):
        raise campaigns.CampaignBusyError("another research attempt is already running")

    try:
        dest = workspace_dir(workspace_id)
        cfg = _experiment_for_workspace(dest)
        if cfg is None or not cfg.run_cmd:
            raise ValueError(f"workspace {workspace_id} has no runnable experiment config")

        attempt = campaigns.start_attempt(
            workspace_id,
            hypothesis,
            proposal_id=proposal_id,
            analysis_id=analysis_id,
            budget_s=budget_s,
        )
        budget = int(attempt["budget_s"])
        started = time.monotonic()
        attempt_commit_created = False
        finalized: dict | None = None

        try:
            # Commit the current state so this attempt is a distinct,
            # resettable commit.
            _exclude_runtime_history(dest)
            _git(["add", "-A"], dest)
            _git(
                ["commit", "-m", "run attempt", "--allow-empty"],
                dest,
            )
            attempt_commit_created = True
            commit = _git_head(dest)

            argv = shlex.split(cfg.run_cmd)
            # A leading bare python uses the exact interpreter running this sidecar.
            if argv and argv[0] in ("python", "python3"):
                argv[0] = sys.executable
            status = "ok"
            stdout = ""
            stderr = ""
            memory_gb = 0.0
            streamed_score: float | None = None

            try:
                if os.name == "nt":
                    proc = subprocess.Popen(
                        argv,
                        cwd=str(dest),
                        env=_experiment_environment(),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    )
                else:
                    proc = subprocess.Popen(
                        argv,
                        cwd=str(dest),
                        env=_experiment_environment(),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        start_new_session=True,
                    )
            except OSError as exc:
                status = "crash"
                stderr = f"failed to launch run_cmd {cfg.run_cmd!r}: {exc}"
            else:
                stop = threading.Event()
                holder: list[float] = []
                sampler = threading.Thread(
                    target=lambda: holder.append(_sample_peak_rss_gb(proc.pid, stop)),
                    daemon=True,
                )
                stdout_buffer = _BoundedTextBuffer(_STREAM_TAIL_CHARS)
                stderr_buffer = _BoundedTextBuffer(_STREAM_TAIL_CHARS)
                metric = _StreamingMetric(cfg.metric_regex)
                assert proc.stdout is not None
                assert proc.stderr is not None
                stdout_reader = threading.Thread(
                    target=_drain_stream,
                    args=(proc.stdout, stdout_buffer, metric),
                    daemon=True,
                )
                stderr_reader = threading.Thread(
                    target=_drain_stream,
                    args=(proc.stderr, stderr_buffer),
                    daemon=True,
                )
                sampler.start()
                stdout_reader.start()
                stderr_reader.start()
                try:
                    return_code = proc.wait(timeout=budget)
                    if return_code != 0:
                        status = "crash"
                except subprocess.TimeoutExpired:
                    status = "timeout"
                finally:
                    _terminate_process_tree(proc)
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                    stdout_reader.join(timeout=5)
                    stderr_reader.join(timeout=5)
                    stop.set()
                    sampler.join(timeout=3)
                    stdout = stdout_buffer.text()
                    stderr = stderr_buffer.text()
                    streamed_score = metric.value()
                    if holder:
                        memory_gb = round(holder[0], 4)

            score = streamed_score
            if score is None:
                score = _parse_metric(cfg.metric_regex, stdout)
            combined = (stdout or "") + (
                "\n--- stderr ---\n" + stderr if stderr else ""
            )
            logs_tail = combined[-4000:]
            elapsed_ms = round((time.monotonic() - started) * 1000)
            finalized = campaigns.finish_attempt(
                int(attempt["id"]),
                status=status,
                score=score,
                memory_gb=memory_gb,
                elapsed_ms=elapsed_ms,
                commit_sha=commit,
                logs_tail=logs_tail,
                description=description,
            )

            if finalized["decision"] == "rejected":
                _reset_preserving_ledger(dest)

            append_ledger(
                workspace_id,
                commit,
                score,
                memory_gb,
                status,
                description or hypothesis or finalized["decision"],
            )
            return {
                "score": score,
                "memory_gb": memory_gb,
                "status": status,
                "commit": commit,
                "logs_tail": logs_tail,
                "attempt_id": finalized["id"],
                "ordinal": finalized["ordinal"],
                "decision": finalized["decision"],
                "improved": finalized["improved"],
                "best_score": finalized["best_score"],
                "elapsed_ms": finalized["elapsed_ms"],
                "budget_s": finalized["budget_s"],
                "campaign_status": finalized["campaign_status"],
                "stop_reason": finalized["stop_reason"],
            }
        except Exception as exc:
            if finalized is None:
                elapsed_ms = round((time.monotonic() - started) * 1000)
                finalized = campaigns.finish_attempt(
                    int(attempt["id"]),
                    status="crash",
                    score=None,
                    memory_gb=0,
                    elapsed_ms=elapsed_ms,
                    commit_sha=_git_head(dest) if attempt_commit_created else "",
                    logs_tail=str(exc)[-4000:],
                    description=description,
                )
                if attempt_commit_created:
                    _reset_preserving_ledger(dest)
            raise
    finally:
        _RUN_LOCK.release()


def _parse_metric(pattern: str, text: str) -> float | None:
    """Capture the last match of ``pattern`` group 1 as a float, else None."""
    if not pattern:
        return None
    try:
        rx = re.compile(pattern)
    except re.error:
        return None
    last: float | None = None
    for m in rx.finditer(text or ""):
        try:
            candidate = float(m.group(1))
            last = candidate if math.isfinite(candidate) else None
        except (IndexError, ValueError):
            continue
    return last


# ── git advance / reset ──────────────────────────────────────────────────────


def _reset_preserving_ledger(dest: Path) -> None:
    """Reset the attempt commit without rolling back the append-only TSV."""
    ledger = dest / LEDGER_FILE
    existed = ledger.is_file()
    contents = ledger.read_bytes() if existed else b""
    try:
        _git(["reset", "--hard", "HEAD~1"], dest)
    finally:
        if existed:
            ledger.write_bytes(contents)
        elif ledger.exists():
            ledger.unlink()


def _exclude_runtime_history(dest: Path) -> None:
    """Keep the SQLite projection out of experiment git commits."""
    info = dest / ".git" / "info"
    if not info.is_dir():
        return
    exclude = info / "exclude"
    existing = exclude.read_text(encoding="utf-8") if exclude.is_file() else ""
    entry = "/history.jsonl"
    if entry not in existing.splitlines():
        separator = "" if not existing or existing.endswith("\n") else "\n"
        exclude.write_text(existing + separator + entry + "\n", encoding="utf-8")


def recover_interrupted_workspace(workspace_id: str) -> None:
    """Best-effort reset for an attempt marked interrupted during startup."""
    dest = workspace_dir(workspace_id)
    if not dest.is_dir() or not (dest / ".git").is_dir():
        return
    try:
        _reset_preserving_ledger(dest)
    except (OSError, subprocess.SubprocessError):
        return


def git_action(workspace_id: str, action: str) -> dict:
    """``advance`` keeps the current commit (no-op); ``reset`` discards the last
    experiment commit (``git reset --hard HEAD~1``). Returns the new HEAD."""
    dest = workspace_dir(workspace_id)
    if action not in {"advance", "reset"}:
        raise ValueError(f"unknown git action: {action!r} (expected 'advance' or 'reset')")

    # New runners decide atomically from the metric. Legacy keep/reset calls are
    # compatibility acknowledgements and may not override that decision.
    latest = campaigns.latest_attempt(workspace_id)
    if latest is not None and latest["decision"] in {"kept", "rejected"}:
        enforced_action = "advance" if latest["decision"] == "kept" else "reset"
        return {
            "ok": True,
            "head": _git_head(dest),
            "decision": latest["decision"],
            "enforced_action": enforced_action,
            "requested_action": action,
        }
    if action == "advance":
        return {"ok": True, "head": _git_head(dest)}
    if action == "reset":
        _reset_preserving_ledger(dest)
        return {"ok": True, "head": _git_head(dest)}
    raise ValueError(f"unknown git action: {action!r} (expected 'advance' or 'reset')")


# ── ledger ───────────────────────────────────────────────────────────────────


def read_ledger(workspace_id: str) -> dict:
    dest = workspace_dir(workspace_id)
    ledger = dest / LEDGER_FILE
    rows: list[dict] = []
    if ledger.is_file():
        lines = ledger.read_text(encoding="utf-8").splitlines()
        for line in lines[1:]:  # skip header
            if not line.strip():
                continue
            cols = line.split("\t")
            row = {LEDGER_HEADER[i]: (cols[i] if i < len(cols) else "") for i in range(len(LEDGER_HEADER))}
            rows.append(row)
    return {"rows": rows}


def append_ledger(
    workspace_id: str,
    commit: str,
    score,
    memory_gb,
    status: str,
    description: str,
) -> dict:
    dest = workspace_dir(workspace_id)
    ledger = dest / LEDGER_FILE
    if not ledger.exists():
        ledger.write_text("\t".join(LEDGER_HEADER) + "\n", encoding="utf-8")
    campaigns.update_attempt_description(str(commit), description)

    # A run now appends automatically. The old explicit ledger tool remains an
    # idempotent compatibility shim, keyed by the attempt commit.
    if commit:
        for row in read_ledger(workspace_id)["rows"]:
            if row["commit"] == str(commit):
                campaigns.refresh_history(workspace_id)
                return {"ok": True, "duplicate": True}
    # Sanitize cell values so a stray tab/newline can't break the TSV.
    def cell(v) -> str:
        return str(v if v is not None else "").replace("\t", " ").replace("\n", " ")

    row = "\t".join(
        [cell(commit), cell(score), cell(memory_gb), cell(status), cell(description)]
    )
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write(row + "\n")
    campaigns.refresh_history(workspace_id)
    return {"ok": True, "duplicate": False}
