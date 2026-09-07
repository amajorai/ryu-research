<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./icon-dark.png" />
    <img src="./icon-light.png" alt="Research" width="144" />
  </picture>
</p>

<div align="center">

# Research

</div>

Multi-step experiment runs where a frontier model iterates on a task, with its embedded Python engine payload.

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

- **`sidecar/`** is the dependency-free Python engine. It owns git-versioned
  workspaces and a SQLite database for campaigns, attempts, timestamps, statuses,
  scores, winner decisions, bounded logs, and RLM audit references. `results.tsv`
  remains an imported/exported compatibility format, not the source of truth.
- **`backend/` (`ryu-research`)** is the Rust app sidecar and MCP server. It
  authenticates to the private Python engine with a dedicated token, exposes the
  campaign read model to the companion, and brokers oversized-history analysis to
  the bound `rlm.query` provider without placing that history in the caller's prompt.
- **`ui/`** builds the self-contained Experiments companion. It talks through the
  generic `window.ryu.app.request` bridge; there is no Research-specific Desktop or
  Core API.

## Experiment lifecycle

`research.init_workspace` creates one campaign and returns the same id as both
`workspace_id` and `campaign_id`. `research.next_variant` creates or reuses one
proposal lease for the current history. Each `research.run` consumes that lease,
inserts a durable running attempt before executing, finalizes it even on crash or
timeout, and compares its score using the experiment's minimize/maximize policy. A
strict improvement is kept; every other candidate is reset. The legacy `keep`,
`reset`, and `ledger` tools remain compatible but cannot override the engine's
decision.

When serialized history reaches the campaign threshold, `research.next_variant`
automatically submits the canonical snapshot to `rlm.query`. Research accepts only
an answer with a matching SHA-256 input digest, campaign-scoped citations, and a
successful recursive step. The resulting proposal is consumed by exactly one next
attempt. Caller-supplied RLM ids and recommendations cannot satisfy the gate.

## Security boundaries

Core's extension bearer authenticates the public Rust sidecar. A separate owner-only
engine token authenticates Rust/MCP calls to Python and is never exposed to
experiment subprocesses. RLM is a peer app reached through Ryu's capability/MCP
surfaces; Research does not import or dial RLM internals.

## Core-vs-Gateway

Research decides what experiment code runs, so execution remains Core-governed.
Model calls used by the researcher and RLM remain Gateway-governed.
