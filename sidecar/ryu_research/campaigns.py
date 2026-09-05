"""Durable campaign, attempt, and one-shot proposal history for autoresearch."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

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
SCHEMA_VERSION = 3
DEFAULT_REASONING_THRESHOLD_CHARS = 12_000
RECENT_ATTEMPT_LIMIT = 8
RECENT_ANALYSIS_LIMIT = 5
RECENT_LOG_CHARS = 400
RECENT_REASONING_CHARS = 1_000
PROPOSAL_KINDS = {"legacy_unverified", "freeform", "rlm_verified"}
_TERMINAL_CAMPAIGN_STATUSES = {"completed", "failed", "cancelled"}
_CAMPAIGN_STATUSES = {"active", "running", *_TERMINAL_CAMPAIGN_STATUSES}
_INITIALIZE_LOCK = threading.Lock()
_HISTORY_LOCK = threading.Lock()
_INITIALIZED_DATABASES: set[Path] = set()


class CampaignBusyError(RuntimeError):
    """Raised when another attempt is already running."""


class ReasoningRequiredError(RuntimeError):
    """Raised when an attempt lacks a current proposal lease."""


class ProposalMigrationError(ValueError):
    """Raised when the removed self-attested reasoning API is used."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _workspaces_root() -> Path:
    override = os.environ.get("RESEARCH_WORKSPACES") or os.environ.get(
        "RYU_RESEARCH_WORKSPACES"
    )
    if override:
        return Path(override)
    ryu_dir = os.environ.get("RYU_DIR")
    return (Path(ryu_dir) if ryu_dir else Path.home() / ".ryu") / "research-workspaces"


def database_path() -> Path:
    return _workspaces_root() / DATABASE_FILE


def _open_connection() -> sqlite3.Connection:
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=5.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    connection.execute("PRAGMA journal_mode = WAL")
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
            policy_json TEXT NOT NULL DEFAULT '{}', stop_reason TEXT
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
            budget_s INTEGER NOT NULL DEFAULT 0, UNIQUE(campaign_id, ordinal)
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
        """
    )
    additions = (
        ("campaigns", "reasoning_threshold_chars", "INTEGER NOT NULL DEFAULT 12000"),
        ("campaigns", "policy_json", "TEXT NOT NULL DEFAULT '{}'"),
        ("campaigns", "stop_reason", "TEXT"),
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
        lines = ledger.read_text(encoding="utf-8").splitlines()[1:] if ledger.is_file() else []
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
                    description,
                    description,
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
            campaign_ids = [
                str(row["id"])
                for row in connection.execute("SELECT id FROM campaigns").fetchall()
            ]
        finally:
            connection.close()
        _INITIALIZED_DATABASES.add(path)
        for campaign_id in campaign_ids:
            refresh_history(campaign_id)
        return [str(row["campaign_id"]) for row in stale]


def _connect() -> sqlite3.Connection:
    initialize_store()
    return _open_connection()


def _history_path(campaign_id: str) -> Path:
    return _workspaces_root() / campaign_id / "history.jsonl"


def _attempt_projection(row: sqlite3.Row, logs_limit: int | None = None) -> dict:
    logs_tail = row["logs_tail"]
    if logs_limit is not None:
        logs_tail = logs_tail[-logs_limit:]
    proposal_id = row["proposal_id"] or row["analysis_id"]
    return {
        "id": row["id"], "ordinal": row["ordinal"], "status": row["status"],
        "decision": row["decision"], "score": row["score"],
        "memory_gb": row["memory_gb"], "elapsed_ms": row["elapsed_ms"],
        "started_at": row["started_at"], "finished_at": row["finished_at"],
        "hypothesis": row["hypothesis"], "description": row["description"],
        "commit_sha": row["commit_sha"], "logs_tail": logs_tail,
        "budget_s": row["budget_s"], "proposal_id": proposal_id,
        "analysis_id": proposal_id,
    }


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


def _canonical_history_jsonl(connection: sqlite3.Connection, campaign: sqlite3.Row) -> str:
    records = [{
        "type": "campaign", "experiment": campaign["experiment"],
        "goal": campaign["goal"], "policy": _policy_from_row(campaign),
    }]
    records.extend(
        _canonical_attempt(row)
        for row in connection.execute(
            "SELECT * FROM attempts WHERE campaign_id = ? ORDER BY ordinal",
            (campaign["id"],),
        )
    )
    return "".join(_canonical_json(record) + "\n" for record in records)


def _document_history_jsonl(connection: sqlite3.Connection, campaign: sqlite3.Row) -> str:
    records = [{
        "type": "campaign", "id": campaign["id"], "name": campaign["name"],
        "experiment": campaign["experiment"], "goal": campaign["goal"],
        "status": campaign["status"], "created_at": campaign["created_at"],
        "best_score": campaign["best_score"], "policy": _policy_from_row(campaign),
        "stop_reason": campaign["stop_reason"],
    }]
    for row in connection.execute(
        "SELECT * FROM attempts WHERE campaign_id = ? ORDER BY ordinal", (campaign["id"],)
    ):
        records.append({"type": "attempt", **_attempt_projection(row)})
    for row in connection.execute(
        "SELECT * FROM analyses WHERE campaign_id = ? ORDER BY id", (campaign["id"],)
    ):
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


def _history_state(connection: sqlite3.Connection, campaign_id: str) -> dict:
    campaign = connection.execute(
        "SELECT * FROM campaigns WHERE id = ?", (campaign_id,)
    ).fetchone()
    if campaign is None:
        raise FileNotFoundError(f"unknown campaign: {campaign_id}")
    attempt_count = int(connection.execute(
        "SELECT COUNT(*) FROM attempts WHERE campaign_id = ?", (campaign_id,)
    ).fetchone()[0])
    canonical = _canonical_history_jsonl(connection, campaign)
    document = _document_history_jsonl(connection, campaign)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    threshold = int(campaign["reasoning_threshold_chars"])
    required_kind = "rlm_verified" if len(canonical) >= threshold else "freeform"
    proposal = None
    for row in connection.execute(
        """SELECT * FROM analyses WHERE campaign_id = ? AND after_attempt_ordinal = ?
        AND history_digest = ? AND consumed_by_attempt_id IS NULL ORDER BY id DESC""",
        (campaign_id, attempt_count, digest),
    ).fetchall():
        if _proposal_is_valid(row, required_kind):
            proposal = row
            break
    return {
        "history_path": str(_history_path(campaign_id).resolve()),
        "history_chars": len(canonical), "document_chars": len(document),
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


def refresh_history(campaign_id: str) -> dict:
    """Atomically refresh the workspace's RLM-readable JSONL projection."""
    with _HISTORY_LOCK:
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
        temporary_path.write_text(state["document"], encoding="utf-8")
        os.replace(temporary_path, path)
        return state


def create_campaign(
    campaign_id: str,
    experiment: str,
    name: str = "",
    goal: str = "",
    reasoning_threshold_chars: int | None = None,
) -> dict:
    threshold = (
        DEFAULT_REASONING_THRESHOLD_CHARS
        if reasoning_threshold_chars is None else int(reasoning_threshold_chars)
    )
    if threshold < 1:
        raise ValueError("reasoning_threshold_chars must be at least 1")
    policy = load_experiment(experiment).policy_snapshot()
    timestamp = _now()
    connection = _connect()
    try:
        with connection:
            connection.execute(
                """INSERT INTO campaigns(
                    id, name, experiment, goal, status, created_at, updated_at,
                    reasoning_threshold_chars, policy_json
                ) VALUES (?, ?, ?, ?, 'active', ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET name = excluded.name,
                    experiment = excluded.experiment, goal = excluded.goal,
                    updated_at = excluded.updated_at,
                    reasoning_threshold_chars = excluded.reasoning_threshold_chars,
                    policy_json = CASE WHEN campaigns.policy_json = '{}'
                    THEN excluded.policy_json ELSE campaigns.policy_json END""",
                (
                    campaign_id, name.strip() or experiment, experiment, goal.strip(),
                    timestamp, timestamp, threshold, _policy_json(policy),
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
    }


def list_campaigns(status: str | None = None) -> dict:
    connection = _connect()
    try:
        parameters: tuple = ()
        where = ""
        if status:
            if status not in _CAMPAIGN_STATUSES:
                raise ValueError(f"unknown campaign status: {status!r}")
            where = "WHERE c.status = ?"
            parameters = (status,)
        rows = connection.execute(
            _summary_sql(where) + " ORDER BY c.updated_at DESC", parameters
        ).fetchall()
        return {"campaigns": [_campaign_summary(row) for row in rows]}
    finally:
        connection.close()


def get_campaign(campaign_id: str, include_full: bool = False) -> dict:
    connection = _connect()
    try:
        row = connection.execute(
            _summary_sql("WHERE c.id = ?"), (campaign_id,)
        ).fetchone()
        if row is None:
            raise FileNotFoundError(f"unknown campaign: {campaign_id}")
        recent_attempt_rows = list(reversed(connection.execute(
            "SELECT * FROM attempts WHERE campaign_id = ? ORDER BY ordinal DESC LIMIT ?",
            (campaign_id, RECENT_ATTEMPT_LIMIT),
        ).fetchall()))
        recent_analysis_rows = list(reversed(connection.execute(
            "SELECT * FROM analyses WHERE campaign_id = ? ORDER BY id DESC LIMIT ?",
            (campaign_id, RECENT_ANALYSIS_LIMIT),
        ).fetchall()))
        if include_full:
            attempt_rows = connection.execute(
                "SELECT * FROM attempts WHERE campaign_id = ? ORDER BY ordinal",
                (campaign_id,),
            ).fetchall()
            analysis_rows = connection.execute(
                "SELECT * FROM analyses WHERE campaign_id = ? ORDER BY id",
                (campaign_id,),
            ).fetchall()
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
        connection.execute("BEGIN IMMEDIATE")
        running = connection.execute(
            "SELECT id, campaign_id FROM attempts WHERE status = 'running' LIMIT 1"
        ).fetchone()
        if running is not None:
            raise CampaignBusyError(
                f"attempt {running['id']} for campaign {running['campaign_id']} is already running"
            )
        campaign = connection.execute(
            "SELECT * FROM campaigns WHERE id = ?", (campaign_id,)
        ).fetchone()
        if campaign is None:
            raise FileNotFoundError(f"unknown campaign: {campaign_id}")
        if campaign["status"] in _TERMINAL_CAMPAIGN_STATUSES:
            raise ValueError(
                f"campaign {campaign_id} is {campaign['status']} and cannot accept attempts"
            )
        history = _history_state(connection, campaign_id)
        proposal = connection.execute(
            "SELECT * FROM analyses WHERE id = ? AND campaign_id = ?",
            (selected_proposal_id, campaign_id),
        ).fetchone()
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
                analysis_id, proposal_id, budget_s
            ) VALUES (?, ?, 'running', 'pending', ?, ?, ?, ?, ?)""",
            (
                campaign_id, ordinal, timestamp, bound_hypothesis,
                selected_proposal_id, selected_proposal_id, effective_budget,
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
        row = connection.execute(
            "SELECT * FROM attempts WHERE id = ?", (attempt_id,)
        ).fetchone()
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
            "SELECT campaign_id FROM attempts WHERE id = ? AND status = 'running'",
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
                timestamp, commit_sha, logs_tail, description.strip(), attempt_id,
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
        row = connection.execute(
            "SELECT * FROM attempts WHERE id = ?", (attempt_id,)
        ).fetchone()
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
        row = connection.execute(
            "SELECT * FROM attempts WHERE campaign_id = ? ORDER BY ordinal DESC LIMIT 1",
            (campaign_id,),
        ).fetchone()
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
            connection.execute(
                """UPDATE attempts SET description = ?
                WHERE commit_sha = ? AND description = ''""",
                (description.strip(), commit_sha),
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
        connection.execute("BEGIN IMMEDIATE")
        state = _history_state(connection, campaign_id)
        timestamp = _now()
        cursor = connection.execute(
            """INSERT INTO analyses(
                campaign_id, status, created_at, finished_at, query, response,
                reasoning, error, history_digest, after_attempt_ordinal, kind
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'legacy_unverified')""",
            (
                campaign_id, status, timestamp,
                None if status == "running" else timestamp,
                query.strip(), response, reasoning, error,
                state["history_digest"], state["attempt_count"],
            ),
        )
        connection.execute(
            "UPDATE campaigns SET updated_at = ? WHERE id = ?",
            (timestamp, campaign_id),
        )
        connection.commit()
        row = connection.execute(
            "SELECT * FROM analyses WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
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
        connection.execute("BEGIN IMMEDIATE")
        state = _history_state(connection, campaign_id)
        supplied_digest = history_digest.strip()
        if not supplied_digest or supplied_digest != state["history_digest"]:
            raise ReasoningRequiredError(
                "proposal history_digest is stale; analyze the current campaign history"
            )
        values = (
            campaign_id, proposal_kind, state["attempt_count"],
            state["history_digest"], query.strip(), response, reasoning_value, error,
            context_id, run_id, next_variant, citations_json, recursions,
        )
        existing = connection.execute(
            """SELECT * FROM analyses WHERE campaign_id = ? AND kind = ?
            AND after_attempt_ordinal = ? AND history_digest = ? AND query = ?
            AND response = ? AND reasoning = ? AND error = ? AND rlm_context_id = ?
            AND rlm_run_id = ? AND recommendation = ? AND citations_json = ?
            AND successful_recursions = ? ORDER BY id DESC LIMIT 1""",
            values,
        ).fetchone()
        if existing is None:
            timestamp = _now()
            cursor = connection.execute(
                """INSERT INTO analyses(
                    campaign_id, kind, status, created_at, finished_at,
                    after_attempt_ordinal, history_digest, query, response,
                    reasoning, error, rlm_context_id, rlm_run_id,
                    recommendation, citations_json, successful_recursions
                ) VALUES (?, ?, 'completed', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    campaign_id, proposal_kind, timestamp, timestamp,
                    state["attempt_count"], state["history_digest"], query.strip(),
                    response, reasoning_value, error, context_id, run_id,
                    next_variant, citations_json, recursions,
                ),
            )
            connection.execute(
                "UPDATE campaigns SET updated_at = ? WHERE id = ?",
                (timestamp, campaign_id),
            )
            existing = connection.execute(
                "SELECT * FROM analyses WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
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
