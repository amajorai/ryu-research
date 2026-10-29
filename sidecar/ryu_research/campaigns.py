"""Durable campaign, attempt, and one-shot proposal history for autoresearch."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import ryu_content_seal as content_seal
from .experiments import (
    AttemptOutcome,
    GLOBAL_BUDGET_MAX_S,
    GLOBAL_BUDGET_MIN_S,
    evaluate_stop,
    list_experiments,
    load_experiment,
    load_toml,
    score_improved,
)

DATABASE_FILE = "experiments.db"
SCHEMA_VERSION = 5
HISTORY_DIGEST_PREFIX = "sha256:"
DEFAULT_REASONING_THRESHOLD_CHARS = 12_000
RECENT_ATTEMPT_LIMIT = 8
RECENT_ANALYSIS_LIMIT = 5
RECENT_LOG_CHARS = 400
RECENT_REASONING_CHARS = 1_000
PROPOSAL_KINDS = {"legacy_unverified", "freeform", "rlm_verified"}
_IMPROVEMENT_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_TERMINAL_CAMPAIGN_STATUSES = {"completed", "failed", "cancelled"}
_CAMPAIGN_STATUSES = {"active", "running", *_TERMINAL_CAMPAIGN_STATUSES}
_INITIALIZE_LOCK = threading.Lock()
_HISTORY_LOCK = threading.Lock()
_INITIALIZED_DATABASES: set[Path] = set()

PLUGIN_ID = "@ryu/research"
# Registered content-store names (KDF input; never renamed).
CONTENT_STORE = "experiments.db"
HISTORY_STORE = "workspace-history"

# Sealed columns (``encb:v1:``, AAD bound to stable ids). Attempt ids are
# AUTOINCREMENT, so attempts bind their unique ``(campaign_id, ordinal)``;
# analyses are inserted with empty placeholders and sealed for their id in the
# same transaction. Names, statuses, scores, digests, commit SHAs, timestamps
# and policies stay plaintext (lists, budgets and the history digest use them).
_GOAL = content_seal.SealedColumn("campaigns", "goal", "id", ("id",))
_ATTEMPT_SEALED = {
    name: content_seal.SealedColumn("attempts", name, "id", ("campaign_id", "ordinal"))
    for name in ("hypothesis", "description", "logs_tail")
}
_ANALYSIS_SEALED = {
    name: content_seal.SealedColumn("analyses", name, "id", ("id",))
    for name in ("query", "response", "reasoning", "error", "recommendation", "citations_json")
}
_SEALED_BY_TABLE = {
    "campaigns": {"goal": _GOAL},
    "attempts": _ATTEMPT_SEALED,
    "analyses": _ANALYSIS_SEALED,
}
_STORE_SEALS: dict[Path, content_seal.StoreSeal] = {}


class CampaignBusyError(RuntimeError):
    """Raised when another attempt is already running."""


class ReasoningRequiredError(RuntimeError):
    """Raised when an attempt lacks a current proposal lease."""


class ProposalMigrationError(ValueError):
    """Raised when the removed self-attested reasoning API is used."""


# Owner kinds stored on every campaign. ``legacy`` marks rows created before
# ownership existed; they stay quarantined to the node operator.
OWNER_NODE = "node"
OWNER_USER = "user"
OWNER_LEGACY = "legacy"
_MAX_PRINCIPAL_ID_CHARS = 256


@dataclass(frozen=True)
class CallerScope:
    """The verified principal a request acts for.

    ``node`` is the node operator on a personal (unbound) node, or Core's own
    node-token caller: it sees every campaign. ``user`` is a verified user (and
    organization, possibly empty) on a shared node: it sees only campaigns it
    created. The scope arrives from the trusted Research backend, which derives
    it from Core-stamped metadata, never from tool arguments.
    """

    kind: str
    user_id: str = ""
    org_id: str = ""

    def __post_init__(self) -> None:
        if self.kind == OWNER_NODE:
            if self.user_id or self.org_id:
                raise ValueError("a node scope carries no user or organization")
            return
        if self.kind != OWNER_USER:
            raise ValueError(f"unknown caller scope kind: {self.kind!r}")
        if not self.user_id or len(self.user_id) > _MAX_PRINCIPAL_ID_CHARS:
            raise ValueError("a user scope requires a bounded user id")
        if len(self.org_id) > _MAX_PRINCIPAL_ID_CHARS:
            raise ValueError("organization id is too long")

    @property
    def unrestricted(self) -> bool:
        return self.kind == OWNER_NODE


def _scope_predicate(scope: CallerScope | None, alias: str = "c") -> tuple[str, tuple]:
    """SQL predicate selecting only campaigns visible to ``scope``.

    ``None`` is reserved for engine-internal bookkeeping (admission counts,
    recovery) and never reaches a request handler.
    """
    if scope is None or scope.unrestricted:
        return "", ()
    return (
        f"{alias}.owner_kind = ? AND {alias}.owner_user_id = ? AND {alias}.owner_org_id = ?",
        (OWNER_USER, scope.user_id, scope.org_id),
    )


def authorize_campaign(campaign_id: str, scope: CallerScope) -> None:
    """Raise ``FileNotFoundError`` unless ``scope`` may act on ``campaign_id``.

    Unknown and foreign campaigns are indistinguishable to the caller.
    """
    predicate, parameters = _scope_predicate(scope)
    where = "c.id = ?" + (f" AND {predicate}" if predicate else "")
    connection = _connect()
    try:
        row = connection.execute(
            f"SELECT 1 FROM campaigns AS c WHERE {where}",  # noqa: S608 - static fragments
            (campaign_id, *parameters),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise FileNotFoundError(f"unknown campaign: {campaign_id}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _workspaces_root() -> Path:
    """Root holding every workspace, the campaign database and the engine token.

    ``RESEARCH_WORKSPACES`` (or ``RYU_RESEARCH_WORKSPACES``) wins; otherwise the
    ``workspaces/`` folder of the app's data dir — ``RYU_PLUGIN_DATA_DIR`` (set by
    Core), else ``<RYU_DIR or ~/.ryu>/plugin-data/@ryu+research``.
    """
    override = os.environ.get("RESEARCH_WORKSPACES") or os.environ.get(
        "RYU_RESEARCH_WORKSPACES"
    )
    if override:
        return Path(override)
    plugin_data = os.environ.get("RYU_PLUGIN_DATA_DIR")
    if plugin_data:
        return Path(plugin_data) / "workspaces"
    ryu_dir = os.environ.get("RYU_DIR")
    base = Path(ryu_dir) if ryu_dir else Path.home() / ".ryu"
    return base / "plugin-data" / "@ryu+research" / "workspaces"


def database_path() -> Path:
    return _workspaces_root() / DATABASE_FILE


def _data_dir() -> Path:
    """The app data dir that holds ``secrets/`` for a standalone key.

    ``RYU_PLUGIN_DATA_DIR`` (Core) or ``RYU_RESEARCH_DATA_DIR`` (the Rust
    sidecar passes its data dir), else the parent of the default workspaces
    root, else the explicit workspaces root itself.
    """
    for name in (content_seal.PLUGIN_DATA_DIR_ENV, "RYU_RESEARCH_DATA_DIR"):
        value = os.environ.get(name, "").strip()
        if value:
            return Path(value)
    if os.environ.get("RESEARCH_WORKSPACES") or os.environ.get("RYU_RESEARCH_WORKSPACES"):
        return _workspaces_root()
    return _workspaces_root().parent


def plugin_keys() -> content_seal.PluginKeys:
    """This engine's content keys (captured from the environment at import)."""
    return content_seal.process_keys(PLUGIN_ID, _data_dir())


def _store_seal() -> content_seal.StoreSeal:
    seal = _STORE_SEALS.get(database_path().resolve())
    if seal is None:
        raise RuntimeError("the campaign store is not initialized")
    return seal


def sealing_health() -> dict:
    """The campaign store's sealing state (for ``/health``)."""
    initialize_store()
    return _store_seal().health()


def _opened(table: str, row) -> dict | None:
    """A row as a dict with its sealed columns opened (raises if one does not open)."""
    if row is None:
        return None
    data = {key: row[key] for key in row.keys()}
    seal = _store_seal()
    for name, column in _SEALED_BY_TABLE[table].items():
        if name in data:
            data[name] = seal.open_text(column, data, data[name])
    return data


def _opened_all(table: str, rows) -> list[dict]:
    return [_opened(table, row) for row in rows]


def _seal(table: str, name: str, ids: dict, value: str) -> str:
    return _store_seal().seal_text(_SEALED_BY_TABLE[table][name], ids, value)


def _open_connection() -> sqlite3.Connection:
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=5.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    connection.execute("PRAGMA journal_mode = WAL")
    # Sealed store: freed pages are zeroed on every connection.
    connection.execute("PRAGMA secure_delete = ON")
    return connection


def _legacy_policy() -> dict:
    return {
        "metric_direction": "minimize",
        "budget_s": 300,
        "budget_min_s": GLOBAL_BUDGET_MIN_S,
        "budget_max_s": GLOBAL_BUDGET_MAX_S,
        "early_stop_max_attempts": 0,
        "early_stop_patience": 0,
        "early_stop_min_delta": 0.0,
        "early_stop_max_failures": 0,
        "early_stop_target": None,
    }


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _policy_json(policy: dict) -> str:
    return _canonical_json(policy)


def _policy_from_row(row: sqlite3.Row) -> dict:
    try:
        stored = json.loads(row["policy_json"] or "{}")
    except (TypeError, json.JSONDecodeError):
        stored = {}
    policy = _legacy_policy()
    if isinstance(stored, dict):
        policy.update(stored)
    return policy


def _clamp_budget(policy: dict, requested_budget_s: object | None) -> int:
    if requested_budget_s is None:
        requested = int(policy["budget_s"])
    else:
        try:
            requested = int(requested_budget_s)
        except (TypeError, ValueError) as exc:
            raise ValueError("budget_s must be an integer") from exc
    return min(
        int(policy["budget_max_s"]),
        max(int(policy["budget_min_s"]), requested),
    )


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS campaigns (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, experiment TEXT NOT NULL,
            goal TEXT NOT NULL DEFAULT '', status TEXT NOT NULL,
            created_at TEXT NOT NULL, started_at TEXT, updated_at TEXT NOT NULL,
            finished_at TEXT, best_score REAL, best_attempt_id INTEGER,
            current_attempt_id INTEGER,
            reasoning_threshold_chars INTEGER NOT NULL DEFAULT 12000,
            policy_json TEXT NOT NULL DEFAULT '{}', stop_reason TEXT,
            improvement_run_id TEXT NOT NULL DEFAULT '',
            owner_kind TEXT NOT NULL DEFAULT 'legacy',
            owner_user_id TEXT NOT NULL DEFAULT '',
            owner_org_id TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            campaign_id TEXT NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL, status TEXT NOT NULL, decision TEXT NOT NULL,
            score REAL, memory_gb REAL, elapsed_ms INTEGER,
            started_at TEXT NOT NULL, finished_at TEXT,
            hypothesis TEXT NOT NULL DEFAULT '', description TEXT NOT NULL DEFAULT '',
            commit_sha TEXT NOT NULL DEFAULT '', logs_tail TEXT NOT NULL DEFAULT '',
            analysis_id INTEGER, proposal_id INTEGER,
            budget_s INTEGER NOT NULL DEFAULT 0,
            workspace_tree TEXT NOT NULL DEFAULT '', UNIQUE(campaign_id, ordinal)
        );
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            campaign_id TEXT NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
            status TEXT NOT NULL, created_at TEXT NOT NULL, finished_at TEXT,
            query TEXT NOT NULL DEFAULT '', response TEXT NOT NULL DEFAULT '',
            reasoning TEXT NOT NULL DEFAULT '', error TEXT NOT NULL DEFAULT '',
            history_digest TEXT NOT NULL DEFAULT '',
            after_attempt_ordinal INTEGER NOT NULL DEFAULT 0,
            rlm_context_id TEXT NOT NULL DEFAULT '', rlm_run_id TEXT NOT NULL DEFAULT '',
            recommendation TEXT NOT NULL DEFAULT '',
            kind TEXT NOT NULL DEFAULT 'legacy_unverified',
            citations_json TEXT NOT NULL DEFAULT '[]',
            successful_recursions INTEGER NOT NULL DEFAULT 0,
            consumed_by_attempt_id INTEGER
        );
        CREATE INDEX IF NOT EXISTS attempts_campaign_idx ON attempts(campaign_id, ordinal);
        CREATE INDEX IF NOT EXISTS campaigns_updated_idx ON campaigns(updated_at DESC);
        CREATE INDEX IF NOT EXISTS analyses_campaign_idx ON analyses(campaign_id, id);
        CREATE TABLE IF NOT EXISTS history_refreshes (
            campaign_id TEXT PRIMARY KEY REFERENCES campaigns(id) ON DELETE CASCADE,
            refreshed_at TEXT NOT NULL
        );
        """
    )
    additions = (
        ("campaigns", "reasoning_threshold_chars", "INTEGER NOT NULL DEFAULT 12000"),
        ("campaigns", "policy_json", "TEXT NOT NULL DEFAULT '{}'"),
        ("campaigns", "stop_reason", "TEXT"),
        ("campaigns", "improvement_run_id", "TEXT NOT NULL DEFAULT ''"),
        # Rows that predate ownership default to the quarantined legacy scope.
        ("campaigns", "owner_kind", "TEXT NOT NULL DEFAULT 'legacy'"),
        ("campaigns", "owner_user_id", "TEXT NOT NULL DEFAULT ''"),
        ("campaigns", "owner_org_id", "TEXT NOT NULL DEFAULT ''"),
        ("attempts", "workspace_tree", "TEXT NOT NULL DEFAULT ''"),
        ("attempts", "analysis_id", "INTEGER"),
        ("attempts", "proposal_id", "INTEGER"),
        ("attempts", "budget_s", "INTEGER NOT NULL DEFAULT 0"),
        ("analyses", "after_attempt_ordinal", "INTEGER NOT NULL DEFAULT 0"),
        ("analyses", "rlm_context_id", "TEXT NOT NULL DEFAULT ''"),
        ("analyses", "rlm_run_id", "TEXT NOT NULL DEFAULT ''"),
        ("analyses", "recommendation", "TEXT NOT NULL DEFAULT ''"),
        ("analyses", "kind", "TEXT NOT NULL DEFAULT 'legacy_unverified'"),
        ("analyses", "citations_json", "TEXT NOT NULL DEFAULT '[]'"),
        ("analyses", "successful_recursions", "INTEGER NOT NULL DEFAULT 0"),
        ("analyses", "consumed_by_attempt_id", "INTEGER"),
    )
    for table, column, declaration in additions:
        _ensure_column(connection, table, column, declaration)
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def _ensure_column(
    connection: sqlite3.Connection, table: str, column: str, declaration: str
) -> None:
    # SAFETY: these identifiers come only from the static migration table above.
    columns = {
        str(row["name"])
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")


def _float_or_none(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _legacy_experiment_id(workspace: Path, known_configs: dict[str, str]) -> str:
    config = workspace / "experiment.toml"
    if config.is_file():
        try:
            digest = hashlib.sha256(config.read_bytes()).hexdigest()
            if digest in known_configs:
                return known_configs[digest]
        except OSError:
            pass
    return "legacy"


def _legacy_timestamp(workspace: Path) -> str:
    try:
        modified = datetime.fromtimestamp(workspace.stat().st_mtime, timezone.utc)
    except OSError:
        modified = datetime.now(timezone.utc)
    return modified.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _migrate_legacy_workspaces(connection: sqlite3.Connection) -> None:
    seal = _store_seal()
    root = _workspaces_root()
    known_configs: dict[str, str] = {}
    policies: dict[str, dict] = {}
    for config in list_experiments():
        policies[config.id] = config.policy_snapshot()
        try:
            digest = hashlib.sha256((config.root / "experiment.toml").read_bytes()).hexdigest()
            known_configs[digest] = config.id
        except OSError:
            continue
    for workspace in sorted(root.iterdir() if root.is_dir() else []):
        if not workspace.is_dir() or not (workspace / "experiment.toml").is_file():
            continue
        campaign_id = workspace.name
        if connection.execute(
            "SELECT 1 FROM campaigns WHERE id = ?", (campaign_id,)
        ).fetchone():
            continue
        raw = load_toml(workspace / "experiment.toml")
        experiment = _legacy_experiment_id(workspace, known_configs)
        policy = policies.get(experiment, _legacy_policy())
        timestamp = _legacy_timestamp(workspace)
        connection.execute(
            """INSERT INTO campaigns(
                id, name, experiment, goal, status, created_at, updated_at, policy_json
            ) VALUES (?, ?, ?, '', 'active', ?, ?, ?)""",
            (
                campaign_id,
                str(raw.get("name", experiment)),
                experiment,
                timestamp,
                timestamp,
                _policy_json(policy),
            ),
        )
        ledger = workspace / "results.tsv"
        lines = (
            [
                open_ledger_line(campaign_id, line)
                for line in ledger.read_text(encoding="utf-8").splitlines()[1:]
            ]
            if ledger.is_file()
            else []
        )
        best_score = None
        best_attempt_id = None
        ordinal = 0
        for line in lines:
            if not line.strip():
                continue
            ordinal += 1
            columns = line.split("\t")
            columns.extend([""] * (5 - len(columns)))
            commit_sha, score_text, memory_text, status, description = columns[:5]
            score = _float_or_none(score_text)
            kept = status == "ok" and score_improved(
                score,
                best_score,
                policy["metric_direction"],
                float(policy["early_stop_min_delta"]),
            )
            cursor = connection.execute(
                """INSERT INTO attempts(
                    campaign_id, ordinal, status, decision, score, memory_gb,
                    elapsed_ms, started_at, finished_at, hypothesis, description,
                    commit_sha, logs_tail, budget_s
                ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, '', ?)""",
                (
                    campaign_id,
                    ordinal,
                    status or "unknown",
                    "kept" if kept else "rejected",
                    score,
                    _float_or_none(memory_text),
                    timestamp,
                    timestamp,
                    seal.seal_text(
                        _ATTEMPT_SEALED["hypothesis"],
                        {"campaign_id": campaign_id, "ordinal": ordinal},
                        description,
                    ),
                    seal.seal_text(
                        _ATTEMPT_SEALED["description"],
                        {"campaign_id": campaign_id, "ordinal": ordinal},
                        description,
                    ),
                    commit_sha,
                    int(policy["budget_s"]),
                ),
            )
            if kept:
                best_score = score
                best_attempt_id = cursor.lastrowid
        connection.execute(
            """UPDATE campaigns SET best_score = ?, best_attempt_id = ?, updated_at = ?
            WHERE id = ?""",
            (best_score, best_attempt_id, timestamp, campaign_id),
        )


def _backfill_policy_snapshots(connection: sqlite3.Connection) -> None:
    for row in connection.execute(
        "SELECT id, experiment, policy_json FROM campaigns"
    ).fetchall():
        try:
            stored = json.loads(row["policy_json"] or "{}")
        except json.JSONDecodeError:
            stored = {}
        if isinstance(stored, dict) and stored.get("metric_direction"):
            policy = _legacy_policy()
            policy.update(stored)
        else:
            try:
                policy = load_experiment(str(row["experiment"])).policy_snapshot()
            except (KeyError, OSError, ValueError):
                policy = _legacy_policy()
        connection.execute(
            "UPDATE campaigns SET policy_json = ? WHERE id = ?",
            (_policy_json(policy), row["id"]),
        )
        connection.execute(
            "UPDATE attempts SET budget_s = ? WHERE campaign_id = ? AND budget_s <= 0",
            (int(policy["budget_s"]), row["id"]),
        )


def initialize_store() -> list[str]:
    """Migrate the store and reject attempts left running by a crash."""
    path = database_path().resolve()
    with _INITIALIZE_LOCK:
        if path in _INITIALIZED_DATABASES and path.is_file():
            return []
        _INITIALIZED_DATABASES.discard(path)
        connection = _open_connection()
        try:
            with connection:
                _create_schema(connection)
            # Seal legacy plaintext (resumable, idempotent) before anything new
            # is written, then scrub freed pages.
            _STORE_SEALS[path] = content_seal.StoreSeal.open(
                connection,
                path,
                CONTENT_STORE,
                plugin_keys(),
                [_GOAL, *_ATTEMPT_SEALED.values(), *_ANALYSIS_SEALED.values()],
            )
            with connection:
                _migrate_legacy_workspaces(connection)
                _backfill_policy_snapshots(connection)
                stale = connection.execute(
                    "SELECT DISTINCT campaign_id FROM attempts WHERE status = 'running'"
                ).fetchall()
                timestamp = _now()
                connection.execute(
                    """UPDATE attempts SET status = 'interrupted', decision = 'rejected',
                    finished_at = ?, elapsed_ms = CASE WHEN elapsed_ms IS NULL THEN 0
                    ELSE elapsed_ms END WHERE status = 'running'""",
                    (timestamp,),
                )
                connection.execute(
                    """UPDATE campaigns SET status = 'active', current_attempt_id = NULL,
                    updated_at = ? WHERE status = 'running'""",
                    (timestamp,),
                )
            stale_ids = {str(row["campaign_id"]) for row in stale}
            campaign_ids = [
                str(row["id"])
                for row in connection.execute(
                    """SELECT c.id, c.updated_at, r.refreshed_at FROM campaigns c
                    LEFT JOIN history_refreshes r ON r.campaign_id = c.id"""
                ).fetchall()
                if _history_needs_refresh(row, stale_ids)
            ]
            ledger_ids = [
                str(row["id"])
                for row in connection.execute("SELECT id FROM campaigns").fetchall()
            ]
        finally:
            connection.close()
        _INITIALIZED_DATABASES.add(path)
        for campaign_id in ledger_ids:
            if campaign_id in campaign_ids:
                continue
            try:
                seal_legacy_ledger(campaign_id)
            except (content_seal.ContentSealError, content_seal.SealUnavailableError) as error:
                _STORE_SEALS[path].unreadable_rows += 1
                print(
                    f"[ryu-research] campaign {campaign_id} ledger was not sealed: {error}",
                    file=sys.stderr,
                )
        for campaign_id in campaign_ids:
            # Rewrites history.jsonl sealed, and seals legacy ledger rows. A
            # campaign that does not open (or a store that refuses writes) is
            # skipped and counted; it never blocks the others.
            try:
                refresh_history(campaign_id)
                seal_legacy_ledger(campaign_id)
            except (content_seal.ContentSealError, content_seal.SealUnavailableError) as error:
                _STORE_SEALS[path].unreadable_rows += 1
                print(
                    f"[ryu-research] campaign {campaign_id} was not refreshed: {error}",
                    file=sys.stderr,
                )
        return [str(row["campaign_id"]) for row in stale]


def _history_needs_refresh(row: sqlite3.Row, stale_ids: set[str]) -> bool:
    """Whether startup must rewrite a campaign's ``history.jsonl``.

    Every write refreshes the file and records when; only a campaign changed
    after its last refresh (a crash in between, a startup repair), one never
    refreshed, or a file missing or still plaintext under a key is rebuilt.
    Rebuilding opens every attempt, which a large store cannot afford for
    every campaign on every start.
    """
    campaign_id = str(row["id"])
    if campaign_id in stale_ids or row["refreshed_at"] is None:
        return True
    if str(row["updated_at"]) >= str(row["refreshed_at"]):
        return True
    path = _history_path(campaign_id)
    try:
        with path.open("rb") as handle:
            head = handle.read(len(content_seal.TEXT_PREFIX))
    except OSError:
        return True
    return plugin_keys().can_seal and head != content_seal.TEXT_PREFIX.encode("ascii")


def _connect() -> sqlite3.Connection:
    initialize_store()
    return _open_connection()


def has_running_attempt() -> bool:
    """Whether the store records an attempt as running (plaintext column)."""
    if not database_path().is_file():
        return False
    connection = _open_connection()
    try:
        row = connection.execute(
            "SELECT 1 FROM attempts WHERE status = 'running' LIMIT 1"
        ).fetchone()
    except sqlite3.OperationalError:
        return False
    finally:
        connection.close()
    return row is not None


def _history_path(campaign_id: str) -> Path:
    return _workspaces_root() / campaign_id / "history.jsonl"


def _attempt_projection(row: sqlite3.Row, logs_limit: int | None = None) -> dict:
    logs_tail = row["logs_tail"]
    if logs_limit is not None:
        logs_tail = logs_tail[-logs_limit:]
    proposal_id = row["proposal_id"] or row["analysis_id"]
    projection = {
        "id": row["id"], "ordinal": row["ordinal"], "status": row["status"],
        "decision": row["decision"], "score": row["score"],
        "memory_gb": row["memory_gb"], "elapsed_ms": row["elapsed_ms"],
        "started_at": row["started_at"], "finished_at": row["finished_at"],
        "hypothesis": row["hypothesis"], "description": row["description"],
        "commit_sha": row["commit_sha"], "logs_tail": logs_tail,
        "budget_s": row["budget_s"], "proposal_id": proposal_id,
        "analysis_id": proposal_id,
    }
    # The bound tree joins the projection only when recorded, so histories
    # (and their digests) written before tree binding stay byte-identical.
    workspace_tree = row["workspace_tree"] if "workspace_tree" in row.keys() else ""
    if workspace_tree:
        projection["workspace_tree"] = workspace_tree
    return projection


def _citations_from_row(row: sqlite3.Row) -> list:
    try:
        citations = json.loads(row["citations_json"] or "[]")
    except json.JSONDecodeError:
        return []
    return citations if isinstance(citations, list) else []


def _analysis_projection(row: sqlite3.Row, text_limit: int | None = None) -> dict:
    response = row["response"]
    reasoning = row["reasoning"] or response
    recommendation = row["recommendation"]
    if text_limit is not None:
        response = response[-text_limit:]
        reasoning = reasoning[-text_limit:]
        recommendation = recommendation[-text_limit:]
    return {
        "id": row["id"], "proposal_id": row["id"], "kind": row["kind"],
        "status": row["status"], "created_at": row["created_at"],
        "finished_at": row["finished_at"], "query": row["query"],
        "response": response, "reasoning": reasoning, "error": row["error"],
        "history_digest": row["history_digest"],
        "after_attempt_ordinal": row["after_attempt_ordinal"],
        "trigger_attempt_count": row["after_attempt_ordinal"],
        "rlm_context_id": row["rlm_context_id"], "rlm_run_id": row["rlm_run_id"],
        "recommendation": recommendation, "citations": _citations_from_row(row),
        "successful_recursions": row["successful_recursions"],
        "consumed_by_attempt_id": row["consumed_by_attempt_id"],
        "summary": reasoning or "No analysis summary recorded.",
    }


def _canonical_attempt(row: sqlite3.Row) -> dict:
    projection = _attempt_projection(row)
    projection.pop("analysis_id", None)
    projection.pop("proposal_id", None)
    return {"type": "attempt", **projection}


def _opened_attempts(connection: sqlite3.Connection, campaign_id: str) -> list[dict]:
    """Every attempt of a campaign, opened once (in ordinal order)."""
    return _opened_all("attempts", connection.execute(
        "SELECT * FROM attempts WHERE campaign_id = ? ORDER BY ordinal", (campaign_id,)
    ))


def _canonical_history_jsonl(campaign: dict, attempts: list[dict]) -> str:
    records = [{
        "type": "campaign", "experiment": campaign["experiment"],
        "goal": campaign["goal"],
        "improvement_run_id": campaign["improvement_run_id"] or None,
        "policy": _policy_from_row(campaign),
    }]
    records.extend(_canonical_attempt(row) for row in attempts)
    return "".join(_canonical_json(record) + "\n" for record in records)


def _document_history_jsonl(
    connection: sqlite3.Connection, campaign: dict, attempts: list[dict]
) -> str:
    records = [{
        "type": "campaign", "id": campaign["id"], "name": campaign["name"],
        "experiment": campaign["experiment"], "goal": campaign["goal"],
        "status": campaign["status"], "created_at": campaign["created_at"],
        "improvement_run_id": campaign["improvement_run_id"] or None,
        "best_score": campaign["best_score"], "policy": _policy_from_row(campaign),
        "stop_reason": campaign["stop_reason"],
    }]
    for row in attempts:
        records.append({"type": "attempt", **_attempt_projection(row)})
    for row in _opened_all("analyses", connection.execute(
        "SELECT * FROM analyses WHERE campaign_id = ? ORDER BY id", (campaign["id"],)
    )):
        records.append({"type": "proposal", **_analysis_projection(row)})
    return "".join(_canonical_json(record) + "\n" for record in records)


def _proposal_is_valid(row: sqlite3.Row, required_kind: str) -> bool:
    if (
        row["kind"] != required_kind
        or row["status"] != "completed"
        or row["consumed_by_attempt_id"] is not None
    ):
        return False
    if required_kind == "freeform":
        return not str(row["recommendation"]).strip()
    return bool(
        str(row["rlm_context_id"]).strip()
        and str(row["rlm_run_id"]).strip()
        and str(row["recommendation"]).strip()
        and _citations_from_row(row)
        and int(row["successful_recursions"]) > 0
    )


def _history_state(
    connection: sqlite3.Connection, campaign_id: str, *, include_document: bool = True
) -> dict:
    """The campaign's canonical history digest and proposal state.

    Each attempt row is opened once and feeds both projections. Writers pass
    ``include_document=False``: they need the digest, not the RLM document.
    """
    campaign = _opened("campaigns", connection.execute(
        "SELECT * FROM campaigns WHERE id = ?", (campaign_id,)
    ).fetchone())
    if campaign is None:
        raise FileNotFoundError(f"unknown campaign: {campaign_id}")
    attempts = _opened_attempts(connection, campaign_id)
    attempt_count = len(attempts)
    canonical = _canonical_history_jsonl(campaign, attempts)
    document = (
        _document_history_jsonl(connection, campaign, attempts) if include_document else None
    )
    digest = (
        HISTORY_DIGEST_PREFIX
        + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    )
    threshold = int(campaign["reasoning_threshold_chars"])
    required_kind = "rlm_verified" if len(canonical) >= threshold else "freeform"
    proposal = None
    for row in _opened_all("analyses", connection.execute(
        """SELECT * FROM analyses WHERE campaign_id = ? AND after_attempt_ordinal = ?
        AND history_digest = ? AND consumed_by_attempt_id IS NULL ORDER BY id DESC""",
        (campaign_id, attempt_count, digest),
    ).fetchall()):
        if _proposal_is_valid(row, required_kind):
            proposal = row
            break
    return {
        "history_path": str(_history_path(campaign_id).resolve()),
        "history_chars": len(canonical),
        "document_chars": len(document) if document is not None else None,
        "history_digest": digest, "reasoning_threshold_chars": threshold,
        "reasoning_required": required_kind == "rlm_verified" and proposal is None,
        "proposal_required": proposal is None, "required_proposal_kind": required_kind,
        "current_proposal_id": int(proposal["id"]) if proposal is not None else None,
        "current_analysis_id": (
            int(proposal["id"])
            if proposal is not None and required_kind == "rlm_verified" else None
        ),
        "attempt_count": attempt_count,
        "canonical_document": canonical,
        "document": document,
    }


def _begin_with_history(connection: sqlite3.Connection, campaign_id: str) -> dict:
    """``BEGIN IMMEDIATE`` with the campaign's history state already computed.

    Opening every attempt can take seconds (the pure-Python AEAD), so the state
    is computed before the write lock is taken. ``PRAGMA data_version`` changes
    only when another connection commits; if one did meanwhile, the state is
    computed again under the lock, so a writer never acts on a stale digest.
    """
    version = connection.execute("PRAGMA data_version").fetchone()[0]
    state = _history_state(connection, campaign_id, include_document=False)
    connection.execute("BEGIN IMMEDIATE")
    if connection.execute("PRAGMA data_version").fetchone()[0] != version:
        state = _history_state(connection, campaign_id, include_document=False)
    return state


def refresh_history(campaign_id: str) -> dict:
    """Atomically refresh the workspace's RLM-readable JSONL projection."""
    with _HISTORY_LOCK:
        # Taken before the read: a write that lands meanwhile carries a later
        # (or equal) updated_at, so startup still sees this campaign as stale.
        refreshed_at = _now()
        connection = _open_connection()
        try:
            state = _history_state(connection, campaign_id)
        finally:
            connection.close()
        path = _history_path(campaign_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        # The digest was computed over the plaintext above; the file on disk
        # is sealed (plaintext only in compat mode).
        temporary_path.write_text(
            seal_history_document(campaign_id, state["document"]), encoding="utf-8"
        )
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
        _record_history_refresh(campaign_id, refreshed_at)
        return state


def _record_history_refresh(campaign_id: str, refreshed_at: str) -> None:
    connection = _open_connection()
    try:
        with connection:
            connection.execute(
                """INSERT INTO history_refreshes(campaign_id, refreshed_at)
                SELECT id, ? FROM campaigns WHERE id = ?
                ON CONFLICT(campaign_id) DO UPDATE SET refreshed_at = excluded.refreshed_at""",
                (refreshed_at, campaign_id),
            )
    finally:
        connection.close()


def create_campaign(
    campaign_id: str,
    experiment: str,
    name: str = "",
    goal: str = "",
    reasoning_threshold_chars: int | None = None,
    improvement_run_id: str = "",
    owner: CallerScope | None = None,
) -> dict:
    """Create (or refresh the metadata of) one campaign.

    ``owner`` is stamped once at creation and never changed afterwards; a
    missing owner records the quarantined legacy scope.
    """
    if owner is None:
        owner_row = (OWNER_LEGACY, "", "")
    else:
        owner_row = (owner.kind, owner.user_id, owner.org_id)
    threshold = (
        DEFAULT_REASONING_THRESHOLD_CHARS
        if reasoning_threshold_chars is None else int(reasoning_threshold_chars)
    )
    if threshold < 1:
        raise ValueError("reasoning_threshold_chars must be at least 1")
    improvement_run_id = str(improvement_run_id or "").strip()
    if improvement_run_id and not _IMPROVEMENT_RUN_ID_RE.fullmatch(improvement_run_id):
        raise ValueError("improvement_run_id must match [A-Za-z0-9._-] and be at most 128 characters")
    policy = load_experiment(experiment).policy_snapshot()
    timestamp = _now()
    connection = _connect()
    try:
        with connection:
            connection.execute(
                """INSERT INTO campaigns(
                    id, name, experiment, goal, status, created_at, updated_at,
                    reasoning_threshold_chars, policy_json, improvement_run_id,
                    owner_kind, owner_user_id, owner_org_id
                ) VALUES (?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET name = excluded.name,
                    experiment = excluded.experiment, goal = excluded.goal,
                    updated_at = excluded.updated_at,
                    reasoning_threshold_chars = excluded.reasoning_threshold_chars,
                    improvement_run_id = CASE WHEN excluded.improvement_run_id != ''
                        THEN excluded.improvement_run_id ELSE campaigns.improvement_run_id END,
                    policy_json = CASE WHEN campaigns.policy_json = '{}'
                    THEN excluded.policy_json ELSE campaigns.policy_json END""",
                (
                    campaign_id, name.strip() or experiment, experiment,
                    _seal("campaigns", "goal", {"id": campaign_id}, goal.strip()),
                    timestamp, timestamp, threshold, _policy_json(policy), improvement_run_id,
                    *owner_row,
                ),
            )
    finally:
        connection.close()
    refresh_history(campaign_id)
    return get_campaign(campaign_id)["campaign"]


def _summary_sql(where: str = "") -> str:
    return f"""SELECT c.*, COUNT(a.id) AS attempt_count,
        (SELECT first_attempt.score FROM attempts AS first_attempt
         WHERE first_attempt.campaign_id = c.id AND first_attempt.status = 'ok'
           AND first_attempt.score IS NOT NULL ORDER BY first_attempt.ordinal ASC LIMIT 1
        ) AS baseline_score
        FROM campaigns AS c LEFT JOIN attempts AS a ON a.campaign_id = c.id
        {where} GROUP BY c.id"""


def _campaign_summary(row: sqlite3.Row) -> dict:
    policy = _policy_from_row(row)
    return {
        "id": row["id"], "name": row["name"], "experiment": row["experiment"],
        "status": row["status"], "created_at": row["created_at"],
        "started_at": row["started_at"], "updated_at": row["updated_at"],
        "finished_at": row["finished_at"], "best_score": row["best_score"],
        "baseline_score": row["baseline_score"],
        "attempt_count": int(row["attempt_count"]),
        "current_attempt_id": row["current_attempt_id"],
        "metric_direction": policy["metric_direction"], "policy": policy,
        "should_stop": row["stop_reason"] is not None,
        "stop_reason": row["stop_reason"],
        "improvement_run_id": row["improvement_run_id"] or None,
    }


def list_campaigns(status: str | None = None, scope: CallerScope | None = None) -> dict:
    """List campaigns visible to ``scope`` (``None`` = engine-internal, all)."""
    connection = _connect()
    try:
        clauses: list[str] = []
        parameters: tuple = ()
        if status:
            if status not in _CAMPAIGN_STATUSES:
                raise ValueError(f"unknown campaign status: {status!r}")
            clauses.append("c.status = ?")
            parameters = (status,)
        predicate, scope_parameters = _scope_predicate(scope)
        if predicate:
            clauses.append(predicate)
            parameters = (*parameters, *scope_parameters)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = connection.execute(
            _summary_sql(where) + " ORDER BY c.updated_at DESC", parameters
        ).fetchall()
        return {"campaigns": [_campaign_summary(row) for row in rows]}
    finally:
        connection.close()


def get_campaign(campaign_id: str, include_full: bool = False) -> dict:
    connection = _connect()
    try:
        row = _opened("campaigns", connection.execute(
            _summary_sql("WHERE c.id = ?"), (campaign_id,)
        ).fetchone())
        if row is None:
            raise FileNotFoundError(f"unknown campaign: {campaign_id}")
        recent_attempt_rows = list(reversed(_opened_all("attempts", connection.execute(
            "SELECT * FROM attempts WHERE campaign_id = ? ORDER BY ordinal DESC LIMIT ?",
            (campaign_id, RECENT_ATTEMPT_LIMIT),
        ).fetchall())))
        recent_analysis_rows = list(reversed(_opened_all("analyses", connection.execute(
            "SELECT * FROM analyses WHERE campaign_id = ? ORDER BY id DESC LIMIT ?",
            (campaign_id, RECENT_ANALYSIS_LIMIT),
        ).fetchall())))
        if include_full:
            attempt_rows = _opened_all("attempts", connection.execute(
                "SELECT * FROM attempts WHERE campaign_id = ? ORDER BY ordinal",
                (campaign_id,),
            ).fetchall())
            analysis_rows = _opened_all("analyses", connection.execute(
                "SELECT * FROM analyses WHERE campaign_id = ? ORDER BY id",
                (campaign_id,),
            ).fetchall())
        else:
            attempt_rows = recent_attempt_rows
            analysis_rows = recent_analysis_rows
        history = _history_state(connection, campaign_id)
        campaign = _campaign_summary(row)
        campaign["goal"] = row["goal"]
        for key in (
            "history_path", "history_chars", "document_chars", "history_digest",
            "reasoning_threshold_chars", "reasoning_required", "proposal_required",
            "required_proposal_kind", "current_proposal_id", "current_analysis_id",
        ):
            campaign[key] = history[key]
        recent_attempts = [
            _attempt_projection(attempt, RECENT_LOG_CHARS) for attempt in recent_attempt_rows
        ]
        campaign["recent_attempts"] = recent_attempts
        campaign["attempts"] = (
            [_attempt_projection(attempt) for attempt in attempt_rows]
            if include_full else recent_attempts
        )
        proposals = [
            _analysis_projection(analysis, None if include_full else RECENT_REASONING_CHARS)
            for analysis in analysis_rows
        ]
        campaign["proposals"] = proposals
        campaign["reasoning"] = proposals
        campaign["analyses"] = proposals
        return {"campaign": campaign}
    finally:
        connection.close()


def set_campaign_status(campaign_id: str, status: str) -> dict:
    if status not in _CAMPAIGN_STATUSES - {"running"}:
        raise ValueError(f"unsupported campaign status: {status!r}")
    connection = _connect()
    try:
        with connection:
            campaign = connection.execute(
                "SELECT current_attempt_id FROM campaigns WHERE id = ?", (campaign_id,)
            ).fetchone()
            if campaign is None:
                raise FileNotFoundError(f"unknown campaign: {campaign_id}")
            if campaign["current_attempt_id"] is not None:
                raise CampaignBusyError("cannot change status while an attempt is running")
            timestamp = _now()
            terminal = status in _TERMINAL_CAMPAIGN_STATUSES
            connection.execute(
                """UPDATE campaigns SET status = ?, updated_at = ?, finished_at = ?,
                stop_reason = ? WHERE id = ?""",
                (
                    status, timestamp, timestamp if terminal else None,
                    f"manual_{status}" if terminal else None, campaign_id,
                ),
            )
    finally:
        connection.close()
    refresh_history(campaign_id)
    return get_campaign(campaign_id)


def _normalize_hypothesis(value: object) -> str:
    return " ".join(str(value or "").split())


def start_attempt(
    campaign_id: str,
    hypothesis: str = "",
    proposal_id: int | None = None,
    analysis_id: int | None = None,
    budget_s: object | None = None,
    workspace_tree: str = "",
) -> dict:
    if proposal_id is not None and analysis_id is not None:
        if int(proposal_id) != int(analysis_id):
            raise ValueError("proposal_id and legacy analysis_id must match")
    requested_proposal_id = proposal_id if proposal_id is not None else analysis_id
    if requested_proposal_id is None:
        raise ReasoningRequiredError("a current proposal_id is required for every attempt")
    try:
        selected_proposal_id = int(requested_proposal_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("proposal_id must be an integer") from exc

    connection = _connect()
    try:
        history = _begin_with_history(connection, campaign_id)
        running = connection.execute(
            "SELECT id FROM attempts WHERE status = 'running' LIMIT 1"
        ).fetchone()
        if running is not None:
            # Never name the running campaign: it may belong to another user.
            raise CampaignBusyError("another research attempt is already running")
        campaign = connection.execute(
            "SELECT * FROM campaigns WHERE id = ?", (campaign_id,)
        ).fetchone()
        if campaign is None:
            raise FileNotFoundError(f"unknown campaign: {campaign_id}")
        if campaign["status"] in _TERMINAL_CAMPAIGN_STATUSES:
            raise ValueError(
                f"campaign {campaign_id} is {campaign['status']} and cannot accept attempts"
            )
        proposal = _opened("analyses", connection.execute(
            "SELECT * FROM analyses WHERE id = ? AND campaign_id = ?",
            (selected_proposal_id, campaign_id),
        ).fetchone())
        if proposal is None:
            raise ReasoningRequiredError("proposal_id is not a proposal for this campaign")
        if int(proposal["after_attempt_ordinal"]) != history["attempt_count"]:
            raise ReasoningRequiredError("proposal is stale for the current attempt ordinal")
        if proposal["history_digest"] != history["history_digest"]:
            raise ReasoningRequiredError("proposal is stale for the current history digest")
        required_kind = history["required_proposal_kind"]
        if not _proposal_is_valid(proposal, required_kind):
            raise ReasoningRequiredError(
                f"the next attempt requires an unconsumed {required_kind} proposal"
            )
        supplied_hypothesis = _normalize_hypothesis(hypothesis)
        if required_kind == "rlm_verified":
            recommendation = _normalize_hypothesis(proposal["recommendation"])
            if supplied_hypothesis and supplied_hypothesis != recommendation:
                raise ReasoningRequiredError(
                    "hypothesis must exactly match the verified recommendation"
                )
            bound_hypothesis = recommendation
        else:
            if not supplied_hypothesis:
                raise ValueError("freeform proposals require a non-empty hypothesis")
            bound_hypothesis = supplied_hypothesis
        effective_budget = _clamp_budget(_policy_from_row(campaign), budget_s)
        ordinal = history["attempt_count"] + 1
        timestamp = _now()
        cursor = connection.execute(
            """INSERT INTO attempts(
                campaign_id, ordinal, status, decision, started_at, hypothesis,
                analysis_id, proposal_id, budget_s, workspace_tree
            ) VALUES (?, ?, 'running', 'pending', ?, ?, ?, ?, ?, ?)""",
            (
                campaign_id, ordinal, timestamp,
                _seal(
                    "attempts", "hypothesis",
                    {"campaign_id": campaign_id, "ordinal": ordinal}, bound_hypothesis,
                ),
                selected_proposal_id, selected_proposal_id, effective_budget,
                str(workspace_tree or ""),
            ),
        )
        attempt_id = int(cursor.lastrowid)
        consumed = connection.execute(
            """UPDATE analyses SET consumed_by_attempt_id = ?
            WHERE id = ? AND consumed_by_attempt_id IS NULL""",
            (attempt_id, selected_proposal_id),
        )
        if consumed.rowcount != 1:
            raise ReasoningRequiredError("proposal was already consumed")
        connection.execute(
            """UPDATE campaigns SET status = 'running',
            started_at = COALESCE(started_at, ?), updated_at = ?,
            current_attempt_id = ?, finished_at = NULL, stop_reason = NULL WHERE id = ?""",
            (timestamp, timestamp, attempt_id, campaign_id),
        )
        connection.commit()
        row = _opened("attempts", connection.execute(
            "SELECT * FROM attempts WHERE id = ?", (attempt_id,)
        ).fetchone())
        result = _attempt_projection(row)
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    refresh_history(campaign_id)
    return result


def _stop_evaluation(
    connection: sqlite3.Connection,
    campaign_id: str,
    policy: dict,
    best_score: float | None,
):
    rows = connection.execute(
        "SELECT status, decision, score FROM attempts WHERE campaign_id = ? ORDER BY ordinal",
        (campaign_id,),
    ).fetchall()
    return evaluate_stop(
        attempts=[
            AttemptOutcome(str(row["status"]), str(row["decision"]), row["score"])
            for row in rows
        ],
        best_score=best_score,
        metric_direction=policy["metric_direction"],
        early_stop_max_attempts=int(policy["early_stop_max_attempts"]),
        early_stop_patience=int(policy["early_stop_patience"]),
        early_stop_max_failures=int(policy["early_stop_max_failures"]),
        early_stop_target=policy["early_stop_target"],
    )


def finish_attempt(
    attempt_id: int,
    *,
    status: str,
    score,
    memory_gb,
    elapsed_ms: int,
    commit_sha: str,
    logs_tail: str,
    description: str = "",
) -> dict:
    campaign_id = ""
    connection = _connect()
    try:
        connection.execute("BEGIN IMMEDIATE")
        attempt = connection.execute(
            "SELECT campaign_id, ordinal FROM attempts WHERE id = ? AND status = 'running'",
            (attempt_id,),
        ).fetchone()
        if attempt is None:
            raise ValueError(f"attempt {attempt_id} is not running")
        campaign_id = str(attempt["campaign_id"])
        campaign = connection.execute(
            "SELECT * FROM campaigns WHERE id = ?", (campaign_id,)
        ).fetchone()
        policy = _policy_from_row(campaign)
        numeric_score = _float_or_none(score)
        incumbent = _float_or_none(campaign["best_score"])
        improved = status == "ok" and score_improved(
            numeric_score, incumbent, policy["metric_direction"],
            float(policy["early_stop_min_delta"]),
        )
        decision = "kept" if improved else "rejected"
        timestamp = _now()
        connection.execute(
            """UPDATE attempts SET status = ?, decision = ?, score = ?, memory_gb = ?,
            elapsed_ms = ?, finished_at = ?, commit_sha = ?, logs_tail = ?,
            description = ? WHERE id = ?""",
            (
                status, decision, numeric_score, memory_gb, max(0, int(elapsed_ms)),
                timestamp, commit_sha,
                _seal("attempts", "logs_tail", dict(attempt), logs_tail),
                _seal("attempts", "description", dict(attempt), description.strip()),
                attempt_id,
            ),
        )
        best_score = numeric_score if improved else incumbent
        stop = _stop_evaluation(connection, campaign_id, policy, best_score)
        campaign_status = stop.terminal_status if stop.should_stop else "active"
        connection.execute(
            """UPDATE campaigns SET status = ?, updated_at = ?, current_attempt_id = NULL,
            best_score = ?, best_attempt_id = CASE WHEN ? THEN ? ELSE best_attempt_id END,
            finished_at = ?, stop_reason = ? WHERE id = ?""",
            (
                campaign_status, timestamp, best_score, improved, attempt_id,
                timestamp if stop.should_stop else None, stop.stop_reason, campaign_id,
            ),
        )
        connection.commit()
        row = _opened("attempts", connection.execute(
            "SELECT * FROM attempts WHERE id = ?", (attempt_id,)
        ).fetchone())
        result = _attempt_projection(row)
        result.update({
            "improved": improved, "best_score": best_score,
            "campaign_status": campaign_status, "should_stop": stop.should_stop,
            "stop_reason": stop.stop_reason,
        })
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    refresh_history(campaign_id)
    return result


def latest_attempt(campaign_id: str) -> dict | None:
    connection = _connect()
    try:
        row = _opened("attempts", connection.execute(
            "SELECT * FROM attempts WHERE campaign_id = ? ORDER BY ordinal DESC LIMIT 1",
            (campaign_id,),
        ).fetchone())
        return _attempt_projection(row) if row is not None else None
    finally:
        connection.close()


def update_attempt_description(commit_sha: str, description: str) -> None:
    if not commit_sha or not description.strip():
        return
    connection = _connect()
    campaign_id = None
    try:
        with connection:
            # Empty descriptions are stored as '' (never sealed), so the
            # "only fill an empty description" rule still runs in SQL.
            for target in connection.execute(
                """SELECT id, campaign_id, ordinal FROM attempts
                WHERE commit_sha = ? AND description = ''""",
                (commit_sha,),
            ).fetchall():
                connection.execute(
                    "UPDATE attempts SET description = ? WHERE id = ? AND description = ''",
                    (
                        _seal("attempts", "description", dict(target), description.strip()),
                        target["id"],
                    ),
                )
                # The description is part of the history: mark it changed.
                connection.execute(
                    "UPDATE campaigns SET updated_at = ? WHERE id = ?",
                    (_now(), target["campaign_id"]),
                )
            row = connection.execute(
                "SELECT campaign_id FROM attempts WHERE commit_sha = ? LIMIT 1",
                (commit_sha,),
            ).fetchone()
            campaign_id = str(row["campaign_id"]) if row is not None else None
    finally:
        connection.close()
    if campaign_id:
        refresh_history(campaign_id)


def _write_analysis_text(
    connection: sqlite3.Connection, analysis_id: int, values: dict[str, str]
) -> None:
    """Seal an analysis row's text for its id (inserted with '' placeholders
    in the same transaction, because the AUTOINCREMENT id is only known then)."""
    assignments = ", ".join(f"{name} = ?" for name in values)
    connection.execute(
        f"UPDATE analyses SET {assignments} WHERE id = ?",  # names: static tuple
        (
            *(
                _seal("analyses", name, {"id": analysis_id}, value)
                for name, value in values.items()
            ),
            analysis_id,
        ),
    )


def record_analysis(
    campaign_id: str,
    *,
    status: str,
    query: str = "",
    response: str = "",
    reasoning: str = "",
    error: str = "",
    history_digest: str = "",
) -> dict:
    """Record a compatibility analysis that can never authorize an attempt."""
    del history_digest
    connection = _connect()
    try:
        state = _begin_with_history(connection, campaign_id)
        timestamp = _now()
        cursor = connection.execute(
            """INSERT INTO analyses(
                campaign_id, status, created_at, finished_at, query, response,
                reasoning, error, history_digest, after_attempt_ordinal, kind
            ) VALUES (?, ?, ?, ?, '', '', '', '', ?, ?, 'legacy_unverified')""",
            (
                campaign_id, status, timestamp,
                None if status == "running" else timestamp,
                state["history_digest"], state["attempt_count"],
            ),
        )
        _write_analysis_text(
            connection,
            int(cursor.lastrowid),
            {"query": query.strip(), "response": response, "reasoning": reasoning, "error": error},
        )
        connection.execute(
            "UPDATE campaigns SET updated_at = ? WHERE id = ?",
            (timestamp, campaign_id),
        )
        connection.commit()
        row = _opened("analyses", connection.execute(
            "SELECT * FROM analyses WHERE id = ?", (cursor.lastrowid,)
        ).fetchone())
        result = {"analysis": _analysis_projection(row)}
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    refresh_history(campaign_id)
    return result


def _evidence_text(value: object | None) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else _canonical_json(value)


def create_proposal(
    campaign_id: str,
    *,
    history_digest: str,
    kind: str,
    recommendation: str = "",
    query: str = "",
    response: str = "",
    reasoning: str = "",
    error: str = "",
    evidence: object | None = None,
    rlm_context_id: str = "",
    rlm_run_id: str = "",
    citations: object | None = None,
    successful_recursions: object = 0,
) -> dict:
    """Create an idempotent proposal lease for the current history snapshot."""
    proposal_kind = str(kind).strip()
    if proposal_kind not in PROPOSAL_KINDS:
        raise ValueError(f"kind must be one of {sorted(PROPOSAL_KINDS)}")
    if proposal_kind == "legacy_unverified":
        raise ValueError(
            "legacy_unverified proposals must use the analyses compatibility API"
        )
    next_variant = recommendation.strip()
    context_id = rlm_context_id.strip()
    run_id = rlm_run_id.strip()
    if citations is None:
        citation_rows: list = []
    elif isinstance(citations, list):
        citation_rows = citations
    else:
        raise ValueError("citations must be a JSON array")
    citations_json = _canonical_json(citation_rows)
    try:
        recursions = int(successful_recursions)
    except (TypeError, ValueError) as exc:
        raise ValueError("successful_recursions must be an integer") from exc
    if recursions < 0:
        raise ValueError("successful_recursions must be non-negative")
    if proposal_kind == "rlm_verified" and not (
        context_id and run_id and next_variant and citation_rows and recursions > 0
    ):
        raise ValueError(
            "rlm_verified requires context, run, recommendation, citations, and a successful recursion"
        )
    if proposal_kind == "freeform" and next_variant:
        raise ValueError("freeform proposals cannot include a recommendation")

    reasoning_value = reasoning or _evidence_text(evidence)
    connection = _connect()
    try:
        state = _begin_with_history(connection, campaign_id)
        supplied_digest = history_digest.strip()
        if not supplied_digest or supplied_digest != state["history_digest"]:
            raise ReasoningRequiredError(
                "proposal history_digest is stale; analyze the current campaign history"
            )
        sealed_text = {
            "query": query.strip(), "response": response, "reasoning": reasoning_value,
            "error": error, "recommendation": next_variant, "citations_json": citations_json,
        }
        # The text columns are sealed (randomized), so the idempotency match
        # selects on the plaintext columns and compares opened text here.
        existing = None
        for candidate in _opened_all("analyses", connection.execute(
            """SELECT * FROM analyses WHERE campaign_id = ? AND kind = ?
            AND after_attempt_ordinal = ? AND history_digest = ? AND rlm_context_id = ?
            AND rlm_run_id = ? AND successful_recursions = ? ORDER BY id DESC""",
            (
                campaign_id, proposal_kind, state["attempt_count"],
                state["history_digest"], context_id, run_id, recursions,
            ),
        ).fetchall()):
            if all(candidate[name] == value for name, value in sealed_text.items()):
                existing = candidate
                break
        if existing is None:
            timestamp = _now()
            cursor = connection.execute(
                """INSERT INTO analyses(
                    campaign_id, kind, status, created_at, finished_at,
                    after_attempt_ordinal, history_digest, query, response,
                    reasoning, error, rlm_context_id, rlm_run_id,
                    recommendation, citations_json, successful_recursions
                ) VALUES (?, ?, 'completed', ?, ?, ?, ?, '', '', '', '', ?, ?, '', '', ?)""",
                (
                    campaign_id, proposal_kind, timestamp, timestamp,
                    state["attempt_count"], state["history_digest"],
                    context_id, run_id, recursions,
                ),
            )
            _write_analysis_text(connection, int(cursor.lastrowid), sealed_text)
            connection.execute(
                "UPDATE campaigns SET updated_at = ? WHERE id = ?",
                (timestamp, campaign_id),
            )
            existing = _opened("analyses", connection.execute(
                "SELECT * FROM analyses WHERE id = ?", (cursor.lastrowid,)
            ).fetchone())
        connection.commit()
        result = {"proposal": _analysis_projection(existing)}
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    refresh_history(campaign_id)
    return result


def record_reasoning(
    campaign_id: str,
    *,
    after_attempt_ordinal: int,
    rlm_context_id: str,
    rlm_run_id: str,
    recommendation: str,
) -> dict:
    del campaign_id, after_attempt_ordinal, rlm_context_id, rlm_run_id, recommendation
    raise ProposalMigrationError(
        "record_reasoning was removed because its proof was forgeable; "
        "use POST /campaigns/:id/proposals"
    )


def history_projection(campaign_id: str) -> dict:
    connection = _connect()
    try:
        state = _history_state(connection, campaign_id)
    finally:
        connection.close()
    return {
        "campaign_id": campaign_id, "attempt_count": state["attempt_count"],
        "history_path": state["history_path"], "history_chars": state["history_chars"],
        "document_chars": state["document_chars"],
        "character_count": state["history_chars"],
        "history_digest": state["history_digest"],
        "reasoning_required": state["reasoning_required"],
        "proposal_required": state["proposal_required"],
        "required_proposal_kind": state["required_proposal_kind"],
        "current_proposal_id": state["current_proposal_id"],
        "canonical_document": state["canonical_document"],
        "document": state["document"],
    }


def _history_cipher() -> content_seal.ContentCipher | None:
    keys = plugin_keys()
    return keys.content_cipher(HISTORY_STORE) if keys.can_seal else None


def seal_history_document(campaign_id: str, document: str) -> str:
    """The on-disk ``history.jsonl``: one ``encb:v1:`` value bound to the campaign.

    Plaintext only in compat mode (an older Core that delivers no key). The
    RLM and the API receive the document from the store, never from this file.
    """
    cipher = _history_cipher()
    if cipher is None:
        _store_seal().assert_writable()
        return document
    return cipher.seal_text(
        document, content_seal.context("history.jsonl", ("campaign", campaign_id))
    )


def open_history_document(campaign_id: str, stored: str) -> str:
    """Read back an on-disk ``history.jsonl`` (legacy plaintext passes through)."""
    if not stored.startswith(content_seal.TEXT_PREFIX):
        return stored
    cipher = _history_cipher()
    if cipher is None:
        raise content_seal.ContentSealError("does_not_open")
    return str(
        cipher.open_text(
            stored, content_seal.context("history.jsonl", ("campaign", campaign_id))
        ).value
    )


def seal_ledger_line(campaign_id: str, line: str) -> str:
    """One ``results.tsv`` row sealed for its campaign.

    The ledger is append-only and its rows have no id of their own, so each
    row binds only the campaign (rows of one ledger could be reordered or
    repeated without detection; they cannot be moved to another campaign).
    """
    cipher = _history_cipher()
    if cipher is None:
        _store_seal().assert_writable()
        return line
    return cipher.seal_text(
        line, content_seal.context("results.tsv", ("campaign", campaign_id))
    )


def open_ledger_line(campaign_id: str, line: str) -> str:
    """One ``results.tsv`` row as written (legacy plaintext passes through)."""
    if not line.startswith(content_seal.TEXT_PREFIX):
        return line
    cipher = _history_cipher()
    if cipher is None:
        raise content_seal.ContentSealError("does_not_open")
    return str(
        cipher.open_text(
            line, content_seal.context("results.tsv", ("campaign", campaign_id))
        ).value
    )


LEDGER_FILE = "results.tsv"


def seal_legacy_ledger(campaign_id: str) -> bool:
    """Rewrite a ledger whose rows are still plaintext with sealed rows.

    Atomic (0600 temp in the same directory, fsync, rename) and idempotent: a
    ledger whose rows are all sealed is left untouched. Returns whether it
    rewrote the file. Copies committed to the workspace's git history before
    this ran stay plaintext (workspace repositories are not sealed).
    """
    if not plugin_keys().can_seal:
        return False
    ledger = _workspaces_root() / campaign_id / LEDGER_FILE
    if not ledger.is_file():
        return False
    lines = ledger.read_text(encoding="utf-8").splitlines()
    pending = [
        line for line in lines[1:]
        if line.strip() and not line.startswith(content_seal.TEXT_PREFIX)
    ]
    if not pending:
        return False
    rewritten = [lines[0]] + [
        seal_ledger_line(campaign_id, line)
        if line.strip() and not line.startswith(content_seal.TEXT_PREFIX)
        else line
        for line in lines[1:]
    ]
    temporary = ledger.with_name(f".{LEDGER_FILE}.{os.getpid()}.sealing")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write("\n".join(rewritten) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, ledger)
    return True
