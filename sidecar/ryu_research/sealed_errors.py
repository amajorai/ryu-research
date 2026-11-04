"""The answer the engine gives when sealed (encrypted) content cannot be read.

Kept out of ``server.py`` so that module does not grow, and so the sentence
sits in a small module the i18n extractor can read on its own.
"""

from __future__ import annotations

# Core delivers the sealing key to the Research sidecar only, not to the MCP
# server it starts; an engine that MCP started holds no key.
SEALED_ERROR = (
    "Research content is encrypted and this engine holds no key. Open the "
    "Research app so Ryu restarts the engine with its key, then try again."
)


def sealed_error_body(exc: Exception, sealing: str) -> dict:
    """The 503 body for a seal failure.

    ``error`` is the whole sentence the Research screen shows (a bare seal
    code such as ``does_not_open`` cannot be translated); the exception's own
    text and code stay alongside for logs and agents.
    """
    body = {
        "error": SEALED_ERROR,
        "sealing": sealing,
        "hint": SEALED_ERROR,
        "detail": str(exc),
    }
    code = getattr(exc, "code", None)
    if isinstance(code, str):
        body["code"] = code
    return body
