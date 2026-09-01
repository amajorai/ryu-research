"""Entrypoint: ``python -m ryu_research`` (or the ``ryu-research`` script).

Binds the stdlib HTTP server on ``RESEARCH_HOST``/``RESEARCH_PORT`` (the
``RYU_RESEARCH_*`` names are honored too for parity with the other sidecars);
default 127.0.0.1:8087. Core sets these when it spawns the sidecar.
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
    host = _env("RESEARCH_HOST", "RYU_RESEARCH_HOST", default="127.0.0.1")
    port = int(_env("RESEARCH_PORT", "RYU_RESEARCH_PORT", default="8087"))
    serve(host, port)


if __name__ == "__main__":
    main()
