"""Ryu autoresearch sidecar.

A small, dependency-free HTTP service the Research app starts on a private
loopback port. It runs one *experiment* at a time inside a git-versioned
workspace, parses a direction-aware scalar metric from stdout, and keeps durable
campaign history plus a git ledger of attempts
(prepare proposal → edit → run → keep-if-improved-else-reset).

Nothing is hardcoded: each experiment kind is a folder under ``experiments/``
carrying an ``experiment.toml`` (``run_cmd``, ``metric_regex``,
``metric_direction``, bounded budgets, and ``mutable_files``). Adding a kind is a
folder, never a code change.
"""

__version__ = "0.1.0"
