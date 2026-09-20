# Ryu Research Sidecar

A small, **dependency-free** HTTP service that turns any experiment into an
autoresearch loop. The Research app owns its embedded copy and starts it on a
private loopback port (`GET /health` is the liveness check).

It runs one experiment at a time inside a **git-versioned workspace**, parses a
single scalar metric using the configured minimize/maximize direction, and keeps a
durable SQLite campaign history plus a git ledger of attempts:
`prepare proposal → edit → run → keep-if-improved-else-reset`.

## Nothing hardcoded

Each experiment kind is a folder under `experiments/<kind>/` with an
`experiment.toml`:

```toml
run_cmd      = "uv run train.py"                 # run one attempt
metric_regex = "val_bpb\\s*=\\s*([0-9.]+)"       # capture ONE scalar
metric_direction = "minimize"                    # or "maximize"
budget_s     = 300                               # wall-clock cap (seconds)
budget_min_s = 1
budget_max_s = 3600
early_stop_max_attempts = 0
early_stop_patience = 0
early_stop_min_delta = 0.0
early_stop_max_failures = 0
mutable_files = ["train.py"]                      # files the agent may edit
gpu_required  = false                             # optional; default false
```

Adding an experiment kind is a new folder, never a code change.

### Bundled kinds

- **`toy`** — CPU, zero external deps (pure Python, seconds). The zero-setup
  default so the loop works on a Mac with no GPU. `train.py` runs a tiny numeric
  optimization and prints `val_bpb = <float>`.
- **`nanochat`** — a faithful Karpathy-style single-file GPT (`prepare.py` +
  `train.py` + `program.md`), `uv`-managed, single GPU. `gpu_required = true`.

## HTTP API (all JSON)

| Method | Path | Body → Result |
|---|---|---|
| GET | `/health` | → `{"status":"ok"}` |
| GET | `/experiments` | → `{"experiments":[…]}` |
| POST | `/workspace/init` | `{experiment}` → `{workspace_id, mutable_files, program_md, experiment}` |
| GET | `/workspace/{id}/file?path=train.py` | → `{path, content}` |
| PUT | `/workspace/{id}/file` | `{path, content}` → `{ok:true}` |
| POST | `/workspace/{id}/run` | `{budget_s}` → `{score, memory_gb, status, commit, logs_tail}` |
| POST | `/workspace/{id}/git` | `{action:"advance"\|"reset"}` → `{ok, head}` |
| GET | `/workspace/{id}/ledger` | → `{rows:[…]}` |
| POST | `/workspace/{id}/ledger` | `{commit, score, memory_gb, status, description}` → `{ok:true}` |

`run` status is `ok` (metric parsed), `crash` (non-zero exit), or `timeout`
(killed at the budget).

## Run it

```bash
# Zero-setup: needs only Python 3.11+ and git.
python -m ryu_research           # binds 127.0.0.1:8087
```

Environment overrides: `RYU_RESEARCH_HOST`, `RYU_RESEARCH_PORT`,
`RYU_RESEARCH_EXPERIMENTS` (bundled experiments dir), `RYU_RESEARCH_WORKSPACES`
(`~/.ryu/research-workspaces` by default).
