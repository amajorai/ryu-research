"""Experiment-kind discovery + config parsing.

Each experiment kind is a folder under ``experiments/<kind>/`` with an
``experiment.toml``. The config is the single source of truth for how to run
one attempt — the sidecar never hardcodes a command, metric, or budget.
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence


MetricDirection = Literal["minimize", "maximize"]
GLOBAL_BUDGET_MIN_S = 1
GLOBAL_BUDGET_MAX_S = 3600


@dataclass(frozen=True)
class AttemptOutcome:
    """The attempt facts needed by the pure early-stop evaluator."""

    status: str
    decision: str
    score: float | None


@dataclass(frozen=True)
class StopEvaluation:
    """A policy decision independent of SQLite or the HTTP server."""

    should_stop: bool
    stop_reason: str | None = None
    terminal_status: Literal["completed", "failed"] | None = None


def score_improved(
    candidate: float | None,
    incumbent: float | None,
    direction: MetricDirection,
    min_delta: float = 0.0,
) -> bool:
    """Return whether a finite score is a direction-aware improvement."""
    if candidate is None or not math.isfinite(candidate):
        return False
    if incumbent is None:
        return True
    if not math.isfinite(incumbent):
        return True
    delta = incumbent - candidate if direction == "minimize" else candidate - incumbent
    # Floating-point subtraction can turn an exact threshold comparison such as
    # 1.1 - 1.0 == 0.1 into 0.10000000000000009. Keep the policy's strictness
    # stable across platforms with a tiny numerical margin.
    return delta > min_delta + 1e-12


def target_reached(
    score: float | None,
    target: float | None,
    direction: MetricDirection,
) -> bool:
    """Return whether a finite best score crosses the configured target."""
    if score is None or target is None or not math.isfinite(score):
        return False
    return score <= target if direction == "minimize" else score >= target


def evaluate_stop(
    *,
    attempts: Sequence[AttemptOutcome],
    best_score: float | None,
    metric_direction: MetricDirection,
    early_stop_max_attempts: int,
    early_stop_patience: int,
    early_stop_max_failures: int,
    early_stop_target: float | None,
) -> StopEvaluation:
    """Evaluate terminal policy from immutable attempt outcomes.

    Failure exhaustion takes precedence over count/patience so a campaign that
    ends on repeated crashes is reported as failed rather than completed.
    """
    if target_reached(best_score, early_stop_target, metric_direction):
        return StopEvaluation(True, "target_reached", "completed")

    consecutive_failures = 0
    for attempt in reversed(attempts):
        if attempt.status == "ok":
            break
        consecutive_failures += 1
    if early_stop_max_failures > 0 and consecutive_failures >= early_stop_max_failures:
        return StopEvaluation(True, "max_failures", "failed")

    if early_stop_max_attempts > 0 and len(attempts) >= early_stop_max_attempts:
        return StopEvaluation(True, "max_attempts", "completed")

    attempts_without_winner = 0
    for attempt in reversed(attempts):
        if attempt.decision == "kept":
            break
        attempts_without_winner += 1
    if early_stop_patience > 0 and attempts_without_winner >= early_stop_patience:
        return StopEvaluation(True, "patience_exhausted", "completed")

    return StopEvaluation(False)


def load_toml(path: Path) -> dict:
    """Parse a flat ``experiment.toml``.

    Prefers stdlib ``tomllib`` (Python 3.11+); falls back to a tiny reader for
    our controlled, flat schema so the sidecar runs on any Python 3.9+ with **no
    dependency** (`tomli` not required). The schema is only top-level
    ``key = value`` pairs where value is a quoted string, int, bool, or an array
    of quoted strings.
    """
    try:
        import tomllib  # noqa: PLC0415

        with path.open("rb") as fh:
            return tomllib.load(fh)
    except ModuleNotFoundError:
        return _load_toml_minimal(path.read_text(encoding="utf-8"))


def _strip_comment(line: str) -> str:
    """Drop a trailing ``#`` comment, but not one inside a quoted string."""
    out = []
    in_str = False
    for ch in line:
        if ch == '"':
            in_str = not in_str
        if ch == "#" and not in_str:
            break
        out.append(ch)
    return "".join(out)


def _unescape_basic(s: str) -> str:
    """Apply the TOML basic-string escapes we use (``\\\\`` → ``\\``, ``\\"``,
    ``\\n``, ``\\t``, ``\\r``). Load-bearing for ``metric_regex``, whose ``\\\\s``
    must become ``\\s`` before it is compiled."""
    out = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "\\" and i + 1 < len(s):
            mapped = {"\\": "\\", '"': '"', "n": "\n", "t": "\t", "r": "\r"}.get(s[i + 1])
            if mapped is not None:
                out.append(mapped)
                i += 2
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _parse_scalar(tok: str):
    tok = tok.strip()
    if tok.startswith('"') and tok.endswith('"'):
        return _unescape_basic(tok[1:-1])
    if tok == "true":
        return True
    if tok == "false":
        return False
    try:
        return int(tok)
    except ValueError:
        try:
            return float(tok)
        except ValueError:
            return tok


def _load_toml_minimal(text: str) -> dict:
    out: dict = {}
    for raw in text.splitlines():
        line = _strip_comment(raw).strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            items = []
            for part in inner.split(","):
                part = part.strip()
                if part:
                    items.append(_parse_scalar(part))
            out[key] = items
        else:
            out[key] = _parse_scalar(value)
    return out


@dataclass
class ExperimentConfig:
    """Parsed ``experiment.toml`` for one experiment kind."""

    id: str
    name: str
    description: str
    run_cmd: str
    metric_regex: str
    budget_s: int
    metric_direction: MetricDirection
    budget_min_s: int
    budget_max_s: int
    early_stop_max_attempts: int
    early_stop_patience: int
    early_stop_min_delta: float
    early_stop_max_failures: int
    early_stop_target: float | None
    mutable_files: list[str]
    gpu_required: bool
    program_md: str
    root: Path

    def __post_init__(self) -> None:
        if self.metric_direction not in {"minimize", "maximize"}:
            raise ValueError("metric_direction must be 'minimize' or 'maximize'")
        budgets = (self.budget_min_s, self.budget_s, self.budget_max_s)
        if any(
            value < GLOBAL_BUDGET_MIN_S or value > GLOBAL_BUDGET_MAX_S
            for value in budgets
        ):
            raise ValueError(
                "budget_min_s, budget_s, and budget_max_s must be within 1..3600"
            )
        if not self.budget_min_s <= self.budget_s <= self.budget_max_s:
            raise ValueError("budget_min_s must be <= budget_s <= budget_max_s")
        for name, value in (
            ("early_stop_max_attempts", self.early_stop_max_attempts),
            ("early_stop_patience", self.early_stop_patience),
            ("early_stop_max_failures", self.early_stop_max_failures),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if not math.isfinite(self.early_stop_min_delta) or self.early_stop_min_delta < 0:
            raise ValueError("early_stop_min_delta must be a finite non-negative number")
        if self.early_stop_target is not None and not math.isfinite(
            self.early_stop_target
        ):
            raise ValueError("early_stop_target must be finite when provided")

    def clamp_budget(self, requested_budget_s: object | None) -> int:
        """Parse and clamp a caller budget to the experiment's safe range."""
        if requested_budget_s is None:
            return self.budget_s
        try:
            requested = int(requested_budget_s)
        except (TypeError, ValueError) as exc:
            raise ValueError("budget_s must be an integer") from exc
        return min(self.budget_max_s, max(self.budget_min_s, requested))

    def is_improvement(
        self, candidate: float | None, incumbent: float | None
    ) -> bool:
        return score_improved(
            candidate,
            incumbent,
            self.metric_direction,
            self.early_stop_min_delta,
        )

    def policy_snapshot(self) -> dict:
        """Return the immutable policy stored with a campaign."""
        return {
            "metric_direction": self.metric_direction,
            "budget_s": self.budget_s,
            "budget_min_s": self.budget_min_s,
            "budget_max_s": self.budget_max_s,
            "early_stop_max_attempts": self.early_stop_max_attempts,
            "early_stop_patience": self.early_stop_patience,
            "early_stop_min_delta": self.early_stop_min_delta,
            "early_stop_max_failures": self.early_stop_max_failures,
            "early_stop_target": self.early_stop_target,
        }

    def to_meta(self) -> dict:
        """The public listing shape (``GET /experiments`` row)."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "gpu_required": self.gpu_required,
            "budget_s": self.budget_s,
            "metric_direction": self.metric_direction,
            "budget_min_s": self.budget_min_s,
            "budget_max_s": self.budget_max_s,
            "early_stop_max_attempts": self.early_stop_max_attempts,
            "early_stop_patience": self.early_stop_patience,
            "early_stop_min_delta": self.early_stop_min_delta,
            "early_stop_max_failures": self.early_stop_max_failures,
            "early_stop_target": self.early_stop_target,
            "mutable_files": list(self.mutable_files),
        }


def experiments_dir() -> Path:
    """Directory holding the ``<kind>/`` experiment folders.

    Overridable via ``RYU_RESEARCH_EXPERIMENTS`` (Core points this at the
    installed sidecar's bundled experiments); defaults to the ``experiments``
    folder shipped next to this package.
    """
    override = os.environ.get("RESEARCH_EXPERIMENTS") or os.environ.get(
        "RYU_RESEARCH_EXPERIMENTS"
    )
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "experiments"


def _read_program_md(root: Path) -> str:
    md = root / "program.md"
    if md.is_file():
        return md.read_text(encoding="utf-8")
    return ""


def experiment_config_from_raw(
    kind: str, raw: dict, root: Path, program_md: str = ""
) -> ExperimentConfig:
    """Validate untyped TOML data at the config boundary."""
    for required in ("run_cmd", "metric_regex", "budget_s", "mutable_files"):
        if required not in raw:
            raise ValueError(f"experiment '{kind}' is missing required field '{required}'")

    target_value = raw.get("early_stop_target")
    return ExperimentConfig(
        id=kind,
        name=str(raw.get("name", kind)),
        description=str(raw.get("description", "")),
        run_cmd=str(raw["run_cmd"]),
        metric_regex=str(raw["metric_regex"]),
        budget_s=int(raw["budget_s"]),
        metric_direction=str(raw.get("metric_direction", "minimize")),
        budget_min_s=int(raw.get("budget_min_s", GLOBAL_BUDGET_MIN_S)),
        budget_max_s=int(raw.get("budget_max_s", GLOBAL_BUDGET_MAX_S)),
        early_stop_max_attempts=int(raw.get("early_stop_max_attempts", 0)),
        early_stop_patience=int(raw.get("early_stop_patience", 0)),
        early_stop_min_delta=float(raw.get("early_stop_min_delta", 0.0)),
        early_stop_max_failures=int(raw.get("early_stop_max_failures", 0)),
        early_stop_target=(
            None if target_value is None else float(target_value)
        ),
        mutable_files=[str(file_name) for file_name in raw["mutable_files"]],
        gpu_required=bool(raw.get("gpu_required", False)),
        program_md=program_md,
        root=root,
    )


def load_experiment(kind: str) -> ExperimentConfig:
    """Parse one experiment kind's ``experiment.toml``.

    Raises ``KeyError`` when the kind is unknown and ``ValueError`` when its
    config is missing a required field.
    """
    root = experiments_dir() / kind
    toml_path = root / "experiment.toml"
    if not toml_path.is_file():
        raise KeyError(kind)

    raw = load_toml(toml_path)

    return experiment_config_from_raw(kind, raw, root, _read_program_md(root))


def list_experiments() -> list[ExperimentConfig]:
    """Every discoverable experiment kind, sorted by id."""
    base = experiments_dir()
    if not base.is_dir():
        return []
    out: list[ExperimentConfig] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "experiment.toml").is_file():
            continue
        try:
            out.append(load_experiment(child.name))
        except (KeyError, ValueError, OSError):
            # A malformed experiment folder is skipped, never fatal.
            continue
    return out
