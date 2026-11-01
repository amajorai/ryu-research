"""Entrypoint: ``python -m ryu_research`` (or the ``ryu-research`` script).

Binds the stdlib HTTP server on loopback (127.0.0.1) at ``RESEARCH_PORT`` (the
``RYU_RESEARCH_PORT`` name is honored too for parity with the other sidecars);
default port 8087. The research backend sets the port when it spawns the engine.
"""

from __future__ import annotations

import os

from .server import serve


def _env(*names: str, default: str) -> str:
    """First set environment variable among ``names``, else ``default``."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def main() -> None:
    # Loopback only: the research backend reaches the engine on this machine.
    host = "127.0.0.1"
    port = int(_env("RESEARCH_PORT", "RYU_RESEARCH_PORT", default="8087"))
    serve(host, port)


if __name__ == "__main__":
    main()
