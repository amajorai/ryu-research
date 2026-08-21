<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./icon-dark.png" />
    <img src="./icon-light.png" alt="Research" width="144" />
  </picture>
</p>

<div align="center">

# Research

</div>

Multi-step experiment runs where a frontier model iterates on a task. (Python sidecar held back, not yet public.)

> **The public home of `ryu-research`.** Source, builds, and releases live here —
> binaries for every platform are attached to each release.
>
> This tree is generated from the Ryu monorepo, so commits pushed here
> directly are replaced on the next sync. **Pull requests are welcome** —
> open them here and they are ported into the monorepo, then flow back out.
> Ryu as a whole: https://github.com/amajorai/ryu

## Install

**App:** [Install](ryu://apps/@ryu/research) (opens the Ryu desktop app and asks you to confirm)

**CLI:**

```bash
ryu apps add @ryu/research
```

**Crate:**

```bash
cargo install ryu-research
```

Prebuilt binaries for every platform are attached to [each release](https://github.com/amajorai/ryu/releases).

## License

Apache-2.0 — see [LICENSE](./LICENSE).

## Parts

- **`backend/` (`ryu-research`)** — the thin **Rust surface** in front of the sidecar. An
  extracted Core capability crate owning two things:
  - the **`/api/research/*` data path** — a thin JSON proxy (`status`, init `workspace`,
    per-workspace `ledger`) the desktop drives; and
  - the **`research.*` MCP tool contract** — 8 tool schemas + HTTP dispatch that a workflow
    `agent`/`tool` node loops over (init → read/write → run → keep/reset → ledger).

  **The `/api/research/*` data path is now served OUT-OF-PROCESS** by the `ryu-research` bin
  (`[[bin]]`, `kind:local`, `public_mount`, `RYU_RESEARCH_BIN`/`RYU_RESEARCH_PORT`, default
  `:7995`) — there is no in-process `research_routes` merge and no `research` cargo feature. The
  crate nonetheless stays a **NON-optional path dependency** of Core, but *only* for the
  `research.*` MCP tool listing (`tool_specs`, surfaced via `sidecar::mcp::research`), which must
  compile in every build. The one kernel coupling (lazy-starting the Core-managed Python engine
  and the on-disk install check) is inverted through the `ResearchHost` trait, so the crate has
  **zero dependency on `apps/core`**.
- **`sidecar/` (Python `ryu_research`)** — the actual **out-of-process engine**: git-versioned
  experiment workspaces, a single scalar metric parsed from stdout (lower = better), and a
  keep-if-improved-else-reset git ledger. Dependency-free stdlib HTTP on **:8087**. The
  sidecar *lifecycle* (`python -m ryu_research`, venv provisioning, health) stays in Core's
  generic sidecar manager, not in this crate.
- **No companion UI.** The desktop renders research via Core-side pages.

## Manifest (Core fixture)

- **id** `@ryu/research`, no runnables, no `permission_grants`.
- **App events (`contributes.hook_events`):** `@ryu/research#workspace.created`, raised
  by the Rust sidecar through `events.emit` when `POST /workspace/init` comes back with a
  workspace, so a plugin hook or workflow can drive the loop without this app knowing it
  exists. It is the **only** emittable moment here: the attempt lifecycle (`run`,
  `keep`/`reset`, ledger appends) travels through the `research.*` MCP tools, which
  dispatch **in Core's own process** — no minted `RYU_EXT_TOKEN`, so an emitter there
  would silently no-op. An event that can never fire is worse than no event, so those are
  deliberately not declared.

## Core-vs-Gateway

A research run decides *what runs* (which experiment, which workspace) → Core. The
researcher agent's own model calls stay Gateway-governed.

## Swap seam

The metric contract (one scalar, lower better) and git ledger are the only fixed interface;
the workspace, runner command, and researcher model are all supplied per run.
