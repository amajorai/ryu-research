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
sandbox_paths = ["/srv/datasets"]                # optional extra READ-ONLY paths (never under ~ or Ryu data)
sandbox_network = false                           # optional; default false (no TCP)
```

Adding an experiment kind is a new folder, never a code change.

### Bundled kinds

- **`toy`** — CPU, zero external deps (pure Python, seconds). The zero-setup
  default so the loop works on any sandbox-capable Linux host with no GPU. `train.py` runs a tiny numeric
  optimization and prints `val_bpb = <float>`.
- **`nanochat`** — a faithful Karpathy-style single-file GPT (`prepare.py` +
  `train.py` + `program.md`), `uv`-managed, single GPU. `gpu_required = true`.

## HTTP API (all JSON)

| Method | Path | Body → Result |
|---|---|---|
| GET | `/health` | → `{"status":"ok"}` |
| GET | `/experiments` | → `{"experiments":[…]}` |
| POST | `/workspace/init` | `{experiment}` → `{workspace_id, workspace_tree, mutable_files, program_md, experiment}` |
| GET | `/workspace/{id}/file?path=train.py` | → `{path, content}` |
| PUT | `/workspace/{id}/file` | `{path, content}` → `{ok:true, workspace_tree}` |
| POST | `/workspace/{id}/run` | `{workspace_tree, proposal_id, hypothesis, budget_s}` → `{score, memory_gb, status, commit, workspace_tree, next_workspace_tree, logs_tail, …}` |
| POST | `/workspace/{id}/git` | `{action:"advance"\|"reset"}` → `{ok, head}` |
| GET | `/workspace/{id}/ledger` | → `{rows:[…]}` |
| POST | `/workspace/{id}/ledger` | `{commit, score, memory_gb, status, description}` → `{ok:true}` |

`run` status is `ok` (metric parsed), `crash` (non-zero exit), or `timeout`
(killed at the budget).

### Ownership, tree binding, and confinement

- Every route except `/health` and `/experiments` requires a caller scope in
  `X-Ryu-Research-Caller` (`node` or `user`, with `-User` / `-Org` for a user),
  sent by the Rust backend from Core-stamped identity. Missing scope → 403.
  Campaigns store an immutable owner; lists are filtered and every id-addressed
  route authorizes the owner (foreign ids → 404). Pre-ownership rows are
  `legacy` and visible only to the `node` scope.
- `run` requires the `workspace_tree` token for the contents it executes. Under a
  per-workspace lock shared with writes, reset and ledger appends, a mismatch is
  rejected (409 `workspace_changed`) before the proposal is consumed. The engine
  commits exactly that tree and runs a private checkout of it, never the
  mutable workspace.
- The command runs confined (`ryu_research/sandbox.py`): Landlock plus a
  seccomp socket filter on Linux (x86_64/aarch64); refused everywhere else,
  macOS included (503 `sandbox_unavailable`). Writable: the per-attempt run dir
  and the workspace's private runtime dir
  (`<data dir>/plugin-data/@ryu+research/sandbox-runtime/<workspace>/`, which
  holds `HOME`, the XDG dirs and `UV_CACHE_DIR`/`UV_TOOL_DIR`/
  `UV_PYTHON_INSTALL_DIR`/`PIP_CACHE_DIR`). Read-only: runtime/system paths and
  declared `sandbox_paths`. The data dir, workspaces root, engine token, home
  and the shared uv/pip dirs are never writable. `AF_UNIX` sockets and
  io_uring are always refused; TCP/UDP only with `sandbox_network = true`,
  and even then only outgoing: `bind()`/`listen()` are refused (seccomp, plus
  Landlock `BIND_TCP` on ABI 4+), so an attempt cannot squat a loopback port
  the backend or Core sends a bearer to. `setsid()`/`setpgid()` are refused,
  so every attempt process stays in its process group and is killed when the
  attempt ends. The engine builds the runtime dir with `O_NOFOLLOW`/`dir_fd`
  and replaces any symlink an earlier attempt planted there.
- The backend's readiness check never sends the engine bearer: it sends a
  nonce to `GET /engine-proof` (`X-Ryu-Engine-Challenge`) and requires
  `HMAC-SHA256(bearer, "ryu-research-engine-proof-v1\n" + nonce)` back.

## Run it

```bash
# Zero-setup: needs only Python 3.11+ and git.
python -m ryu_research           # binds 127.0.0.1:8087
```

The engine always binds loopback. Environment overrides: `RYU_RESEARCH_PORT`,
`RYU_RESEARCH_EXPERIMENTS` (bundled experiments dir), `RYU_RESEARCH_WORKSPACES`
(`$RYU_PLUGIN_DATA_DIR/workspaces` by default, i.e.
`<data dir>/plugin-data/@ryu+research/workspaces`).
