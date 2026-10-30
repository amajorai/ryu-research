"""Dependency-free HTTP server exposing the research contract.

Uses only the Python standard library (``http.server``), so the sidecar runs on
a bare Python with no pip install. All bodies are JSON. Routes:

    GET  /health
    POST /admin/shutdown
    GET  /engine-proof
    GET  /experiments
    POST /workspace/init
    GET  /workspace/{id}/file?path=...
    PUT  /workspace/{id}/file
    POST /workspace/{id}/run
    POST /workspace/{id}/git
    GET  /workspace/{id}/ledger
    POST /workspace/{id}/ledger
    GET  /campaigns
    GET  /campaigns/{id}
    GET  /campaigns/{id}/history
    POST /campaigns/{id}/status
    POST /campaigns/{id}/analyses
    POST /campaigns/{id}/proposals
    POST /campaigns/{id}/finish

Every route except ``/health`` and ``/engine-proof`` requires the private engine
bearer. ``/engine-proof`` lets the Research backend check that the process on
the engine port knows the bearer *before* it sends the bearer there: it answers
the ``X-Ryu-Engine-Challenge`` nonce with an HMAC keyed by the bearer, so
something else squatting the port cannot pass as the engine. Every route
except ``/health``, ``/engine-proof`` and ``/experiments`` also requires a caller scope (see
:class:`ryu_research.campaigns.CallerScope`), sent by the trusted Research backend
in the ``X-Ryu-Research-Caller`` / ``-User`` / ``-Org`` headers. Requests without a
valid scope fail closed with 403. Campaign creation stamps the scope immutably,
lists are filtered by it, and every id-addressed workspace or campaign route
authorizes the stored owner before reading or mutating; a foreign id is
indistinguishable from an unknown one (404).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import campaigns, experiments, sandbox, workspaces
from . import ryu_content_seal as content_seal

# Precompiled route matchers for the /workspace/{id}/... family.
_FILE_RE = re.compile(r"^/workspace/([^/]+)/file$")
_RUN_RE = re.compile(r"^/workspace/([^/]+)/run$")
_GIT_RE = re.compile(r"^/workspace/([^/]+)/git$")
_LEDGER_RE = re.compile(r"^/workspace/([^/]+)/ledger$")
_CAMPAIGN_RE = re.compile(r"^/campaigns/([^/]+)$")
_CAMPAIGN_HISTORY_RE = re.compile(r"^/campaigns/([^/]+)/history$")
_CAMPAIGN_STATUS_RE = re.compile(r"^/campaigns/([^/]+)/status$")
_CAMPAIGN_ANALYSES_RE = re.compile(r"^/campaigns/([^/]+)/analyses$")
_CAMPAIGN_PROPOSALS_RE = re.compile(r"^/campaigns/([^/]+)/proposals$")
_CAMPAIGN_REASONING_RE = re.compile(r"^/campaigns/([^/]+)/reasoning$")
_CAMPAIGN_FINISH_RE = re.compile(r"^/campaigns/([^/]+)/finish$")

# Every id-addressed route. A workspace id is its campaign id.
_SCOPED_ID_ROUTES = (
    _FILE_RE,
    _RUN_RE,
    _GIT_RE,
    _LEDGER_RE,
    _CAMPAIGN_HISTORY_RE,
    _CAMPAIGN_STATUS_RE,
    _CAMPAIGN_ANALYSES_RE,
    _CAMPAIGN_PROPOSALS_RE,
    _CAMPAIGN_REASONING_RE,
    _CAMPAIGN_FINISH_RE,
    _CAMPAIGN_RE,
)

ENGINE_CHALLENGE_HEADER = "X-Ryu-Engine-Challenge"
# Domain separation: a proof is never a value valid for any other purpose.
ENGINE_PROOF_CONTEXT = b"ryu-research-engine-proof-v1\n"
_ENGINE_CHALLENGE_RE = re.compile(r"^[0-9a-f]{32,128}$")

CALLER_HEADER = "X-Ryu-Research-Caller"
CALLER_USER_HEADER = "X-Ryu-Research-Caller-User"
CALLER_ORG_HEADER = "X-Ryu-Research-Caller-Org"

MAX_REQUEST_BODY_BYTES = 2 * 1024 * 1024
# Core delivers the sealing key to the Research sidecar only, not to the MCP
# server it starts; an engine that MCP started holds no key.
SEALED_HINT = (
    "Research content is encrypted and this engine holds no key. Open the "
    "Research app so Ryu restarts the engine with its key, then try again."
)
MAX_ACTIVE_CAMPAIGNS = 8
_CAMPAIGN_ADMISSION_LOCK = threading.Lock()


class RequestBodyTooLargeError(ValueError):
    """Raised before an HTTP request body can exceed its byte ceiling."""


class CampaignAdmissionError(ValueError):
    """Raised when the sidecar's active campaign capacity is exhausted."""


class CallerScopeRequiredError(PermissionError):
    """Raised when a request carries no verified caller scope."""


def _expected_token() -> str:
    """Resolve the private Rust-proxy → engine credential.

    The file fallback is read for every request so Core can securely adopt a
    development engine that was already running when it created the token.
    """
    configured = (os.environ.get("RYU_RESEARCH_ENGINE_TOKEN") or "").strip()
    if configured:
        return configured
    token_file = workspaces.workspaces_root() / ".engine-token"
    try:
        return token_file.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def engine_proof(token: str, challenge: str) -> str:
    """HMAC-SHA256 over ``challenge`` keyed by the engine bearer (hex)."""
    return hmac.new(
        token.encode("utf-8"), ENGINE_PROOF_CONTEXT + challenge.encode("ascii"), hashlib.sha256
    ).hexdigest()


class Handler(BaseHTTPRequestHandler):
    server_version = "ryu-research/0.1"

    # Quieter logging: route to nothing (Core captures stdout separately).
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        return

    # ── helpers ──────────────────────────────────────────────────────────────

    def _authorized(self) -> bool:
        expected_token = _expected_token()
        if not expected_token:
            return False
        header = self.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return False
        presented = header[len("Bearer ") :].strip()
        return hmac.compare_digest(presented, expected_token)

    def _send_sealed(self, exc: Exception, sealing: str) -> None:
        if getattr(exc, "code", None) == "already_a_ryu_envelope":
            return self._send(
                400, {"error": "a value may not start with an encrypted-content prefix"}
            )
        return self._send(
            503, {"error": str(exc), "sealing": sealing, "hint": SEALED_HINT}
        )

    def _caller_scope(self) -> campaigns.CallerScope:
        """The verified caller scope, or :class:`CallerScopeRequiredError`.

        Only the bearer-authenticated backend can reach this point, and it
        derives these headers from Core-stamped metadata, never from tool input.
        """
        kind = (self.headers.get(CALLER_HEADER) or "").strip().lower()
        user_id = (self.headers.get(CALLER_USER_HEADER) or "").strip()
        org_id = (self.headers.get(CALLER_ORG_HEADER) or "").strip()
        try:
            if kind in {campaigns.OWNER_NODE, campaigns.OWNER_USER}:
                # A node scope carrying user/org headers is malformed and rejected.
                return campaigns.CallerScope(kind, user_id, org_id)
        except ValueError as exc:
            raise CallerScopeRequiredError(str(exc)) from exc
        raise CallerScopeRequiredError("a verified Research caller scope is required")

    def _authorize_path(self, path: str, scope: campaigns.CallerScope) -> None:
        """Authorize the campaign/workspace id a scoped route addresses."""
        for pattern in _SCOPED_ID_ROUTES:
            match = pattern.match(path)
            if match:
                campaigns.authorize_campaign(match.group(1), scope)
                return

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        transfer_encoding = (self.headers.get("Transfer-Encoding") or "").lower()
        if transfer_encoding:
            if transfer_encoding != "chunked":
                raise ValueError("unsupported transfer encoding")
            raw_buffer = bytearray()
            while True:
                line = self.rfile.readline(4096)
                if not line or not line.endswith(b"\r\n"):
                    raise ValueError("invalid chunked request body")
                size_text = line[:-2].split(b";", 1)[0].strip()
                try:
                    size = int(size_text, 16)
                except ValueError as exc:
                    raise ValueError("invalid chunk size") from exc
                if size < 0:
                    raise ValueError("invalid chunk size")
                if size == 0:
                    while True:
                        trailer = self.rfile.readline(4096)
                        if trailer in {b"\r\n", b"\n", b""}:
                            break
                    break
                if size > MAX_REQUEST_BODY_BYTES - len(raw_buffer):
                    self.close_connection = True
                    raise RequestBodyTooLargeError("request body is too large")
                chunk = self.rfile.read(size)
                if len(chunk) != size or self.rfile.read(2) != b"\r\n":
                    raise ValueError("invalid chunked request body")
                raw_buffer.extend(chunk)
            raw = bytes(raw_buffer)
        else:
            length_text = self.headers.get("Content-Length", "0") or "0"
            try:
                length = int(length_text)
            except ValueError as exc:
                raise ValueError("invalid Content-Length") from exc
            if length < 0:
                raise ValueError("invalid Content-Length")
            if length > MAX_REQUEST_BODY_BYTES:
                self.close_connection = True
                raise RequestBodyTooLargeError("request body is too large")
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("incomplete request body")
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _init_workspace(self, body: dict, scope: campaigns.CallerScope) -> dict:
        with _CAMPAIGN_ADMISSION_LOCK:
            active = campaigns.list_campaigns("active").get("campaigns", [])
            if len(active) >= MAX_ACTIVE_CAMPAIGNS:
                raise CampaignAdmissionError(
                    "the Research sidecar has reached its active campaign limit"
                )
            return workspaces.init_workspace(
                str(body.get("experiment", "")).strip(),
                name=str(body.get("name", "")),
                goal=str(body.get("goal", "")),
                reasoning_threshold_chars=body.get("reasoning_threshold_chars"),
                improvement_run_id=str(
                    body.get("improvementRunId", body.get("improvement_run_id", ""))
                ),
                owner=scope,
            )

    # ── GET ──────────────────────────────────────────────────────────────────

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/health":
                # The key state only (no content): lets the Rust sidecar that
                # holds the delivered key replace an engine started without it.
                return self._send(200, {"status": "ok", "sealing": _sealing_state()})

            if path == "/engine-proof":
                challenge = (self.headers.get(ENGINE_CHALLENGE_HEADER) or "").strip()
                if not _ENGINE_CHALLENGE_RE.match(challenge):
                    return self._send(400, {"error": "invalid engine challenge"})
                expected_token = _expected_token()
                if not expected_token:
                    return self._send(503, {"error": "engine credential unavailable"})
                return self._send(200, {"proof": engine_proof(expected_token, challenge)})

            if not self._authorized():
                return self._send(401, {"error": "unauthorized"})

            if path == "/experiments":
                rows = [c.to_meta() for c in experiments.list_experiments()]
                return self._send(200, {"experiments": rows})

            scope = self._caller_scope()
            if path == "/campaigns":
                query = parse_qs(parsed.query)
                status = (query.get("status") or [None])[0]
                return self._send(200, campaigns.list_campaigns(status, scope))

            self._authorize_path(path, scope)

            m = _CAMPAIGN_HISTORY_RE.match(path)
            if m:
                return self._send(200, campaigns.history_projection(m.group(1)))

            m = _CAMPAIGN_RE.match(path)
            if m:
                query = parse_qs(parsed.query)
                include_full = (query.get("full") or [""])[0].lower() in {
                    "1",
                    "true",
                    "yes",
                }
                detail = campaigns.get_campaign(m.group(1), include_full=include_full)
                try:
                    detail["campaign"]["workspace_tree"] = workspaces.workspace_tree(
                        m.group(1)
                    )
                except (OSError, ValueError, subprocess.SubprocessError):
                    detail["campaign"]["workspace_tree"] = None
                return self._send(200, detail)

            m = _FILE_RE.match(path)
            if m:
                qs = parse_qs(parsed.query)
                file_path = (qs.get("path") or [""])[0]
                if not file_path:
                    return self._send(400, {"error": "missing ?path"})
                return self._send(200, workspaces.read_file(m.group(1), file_path))

            m = _LEDGER_RE.match(path)
            if m:
                return self._send(200, workspaces.read_ledger(m.group(1)))

            return self._send(404, {"error": f"no route for GET {path}"})
        except CallerScopeRequiredError as exc:
            return self._send(403, {"error": str(exc)})
        except FileNotFoundError as exc:
            return self._send(404, {"error": str(exc)})
        except (KeyError, ValueError) as exc:
            return self._send(400, {"error": str(exc)})
        except content_seal.SealUnavailableError as exc:
            return self._send_sealed(exc, "unavailable")
        except content_seal.ContentSealError as exc:
            return self._send_sealed(exc, "unreadable")
        except Exception as exc:  # noqa: BLE001
            return self._send(500, {"error": str(exc)})

    # ── POST / PUT ───────────────────────────────────────────────────────────

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if not self._authorized():
                return self._send(401, {"error": "unauthorized"})
            # Engine administration, not a tenant action: the bearer alone
            # authorizes it, so it carries no caller scope.
            if path == "/admin/shutdown":
                # Authenticated hand-off: the sidecar that holds the sealing key
                # stops an engine that was started without one, then restarts it.
                # Never while an experiment runs: stopping would mark the
                # attempt interrupted and roll its workspace back. The caller
                # keeps this engine and retries on a later request.
                if not _claim_idle_for_shutdown():
                    return self._send(
                        409, {"error": "an experiment is running", "stopping": False}
                    )
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return self._send(202, {"stopping": True})

            scope = self._caller_scope()
            self._authorize_path(path, scope)
            body = self._read_json()

            if path == "/workspace/init":
                experiment = str(body.get("experiment", "")).strip()
                if not experiment:
                    return self._send(400, {"error": "missing 'experiment'"})
                return self._send(200, self._init_workspace(body, scope))

            m = _RUN_RE.match(path)
            if m:
                return self._send(
                    200,
                    workspaces.run_experiment(
                        m.group(1),
                        body.get("budget_s"),
                        hypothesis=str(body.get("hypothesis", "")),
                        description=str(body.get("description", "")),
                        proposal_id=body.get("proposal_id"),
                        analysis_id=body.get("analysis_id"),
                        workspace_tree=body.get("workspace_tree"),
                    ),
                )

            m = _GIT_RE.match(path)
            if m:
                action = str(body.get("action", "")).strip()
                return self._send(200, workspaces.git_action(m.group(1), action))

            m = _LEDGER_RE.match(path)
            if m:
                return self._send(
                    200,
                    workspaces.append_ledger(
                        m.group(1),
                        body.get("commit", ""),
                        body.get("score"),
                        body.get("memory_gb"),
                        str(body.get("status", "")),
                        str(body.get("description", "")),
                    ),
                )

            m = _CAMPAIGN_STATUS_RE.match(path)
            if m:
                status = str(body.get("status", "")).strip()
                if not status:
                    return self._send(400, {"error": "missing 'status'"})
                return self._send(200, campaigns.set_campaign_status(m.group(1), status))

            m = _CAMPAIGN_ANALYSES_RE.match(path)
            if m:
                status = str(body.get("status", "completed")).strip() or "completed"
                return self._send(
                    200,
                    campaigns.record_analysis(
                        m.group(1),
                        status=status,
                        query=str(body.get("query", "")),
                        response=str(body.get("response", "")),
                        reasoning=str(body.get("reasoning", "")),
                        error=str(body.get("error", "")),
                        history_digest=str(body.get("history_digest", "")),
                    ),
                )

            m = _CAMPAIGN_REASONING_RE.match(path)
            if m:
                return self._send(
                    200,
                    campaigns.record_reasoning(
                        m.group(1),
                        after_attempt_ordinal=body.get("after_attempt_ordinal"),
                        rlm_context_id=str(body.get("rlm_context_id", "")),
                        rlm_run_id=str(body.get("rlm_run_id", "")),
                        recommendation=str(body.get("recommendation", "")),
                    ),
                )

            m = _CAMPAIGN_PROPOSALS_RE.match(path)
            if m:
                return self._send(
                    200,
                    campaigns.create_proposal(
                        m.group(1),
                        history_digest=str(body.get("history_digest", "")),
                        kind=str(body.get("kind", "")),
                        recommendation=str(body.get("recommendation", "")),
                        query=str(body.get("query", "")),
                        response=str(body.get("response", "")),
                        reasoning=str(body.get("reasoning", "")),
                        error=str(body.get("error", "")),
                        evidence=body.get("evidence"),
                        rlm_context_id=str(body.get("rlm_context_id", "")),
                        rlm_run_id=str(body.get("rlm_run_id", "")),
                        citations=body.get("citations"),
                        successful_recursions=body.get("successful_recursions", 0),
                    ),
                )

            m = _CAMPAIGN_FINISH_RE.match(path)
            if m:
                status = str(body.get("status", "")).strip()
                if status not in {"completed", "failed", "cancelled"}:
                    return self._send(
                        400,
                        {"error": "status must be completed, failed, or cancelled"},
                    )
                return self._send(200, campaigns.set_campaign_status(m.group(1), status))

            return self._send(404, {"error": f"no route for POST {path}"})
        except CallerScopeRequiredError as exc:
            return self._send(403, {"error": str(exc)})
        except RequestBodyTooLargeError as exc:
            return self._send(413, {"error": str(exc)})
        except workspaces.WorkspaceStateMismatchError as exc:
            return self._send(409, {"error": str(exc), "code": "workspace_changed"})
        except sandbox.SandboxUnavailableError as exc:
            return self._send(503, {"error": str(exc), "code": "sandbox_unavailable"})
        except CampaignAdmissionError as exc:
            return self._send(429, {"error": str(exc)})
        except FileNotFoundError as exc:
            return self._send(404, {"error": str(exc)})
        except KeyError as exc:
            return self._send(404, {"error": f"unknown experiment: {exc}"})
        except (ValueError, json.JSONDecodeError) as exc:
            return self._send(400, {"error": str(exc)})
        except campaigns.CampaignBusyError as exc:
            return self._send(409, {"error": str(exc)})
        except campaigns.ReasoningRequiredError as exc:
            return self._send(409, {"error": str(exc), "reasoning_required": True})
        except content_seal.SealUnavailableError as exc:
            return self._send_sealed(exc, "unavailable")
        except content_seal.ContentSealError as exc:
            return self._send_sealed(exc, "unreadable")
        except Exception as exc:  # noqa: BLE001
            return self._send(500, {"error": str(exc)})

    def do_PUT(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if not self._authorized():
                return self._send(401, {"error": "unauthorized"})
            scope = self._caller_scope()
            self._authorize_path(path, scope)
            body = self._read_json()
            m = _FILE_RE.match(path)
            if m:
                file_path = str(body.get("path", "")).strip()
                if not file_path:
                    return self._send(400, {"error": "missing 'path'"})
                return self._send(
                    200,
                    workspaces.write_file(m.group(1), file_path, str(body.get("content", ""))),
                )
            return self._send(404, {"error": f"no route for PUT {path}"})
        except CallerScopeRequiredError as exc:
            return self._send(403, {"error": str(exc)})
        except FileNotFoundError as exc:
            return self._send(404, {"error": str(exc)})
        except RequestBodyTooLargeError as exc:
            return self._send(413, {"error": str(exc)})
        except workspaces.WorkspaceResourceLimitError as exc:
            return self._send(413, {"error": str(exc)})
        except campaigns.CampaignBusyError as exc:
            return self._send(409, {"error": str(exc)})
        except PermissionError as exc:
            # write_file rejects paths outside the declared mutable_files.
            return self._send(403, {"error": str(exc)})
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            return self._send(400, {"error": str(exc)})
        except content_seal.SealUnavailableError as exc:
            return self._send_sealed(exc, "unavailable")
        except content_seal.ContentSealError as exc:
            return self._send_sealed(exc, "unreadable")
        except Exception as exc:  # noqa: BLE001
            return self._send(500, {"error": str(exc)})


def _claim_idle_for_shutdown() -> bool:
    """Take the run lock for good when no attempt is running.

    Holding ``workspaces._RUN_LOCK`` until the process exits means no attempt can
    start between this check and the shutdown. False (lock released) when an
    attempt holds the lock or the store records one as running.
    """
    if not workspaces._RUN_LOCK.acquire(blocking=False):  # noqa: SLF001
        return False
    try:
        running = campaigns.has_running_attempt()
    except Exception:  # noqa: BLE001 - an unreadable store must not be stopped blind
        running = True
    if running:
        workspaces._RUN_LOCK.release()  # noqa: SLF001
        return False
    return True


def _sealing_state() -> str:
    try:
        return str(campaigns.sealing_health()["state"])
    except Exception:  # noqa: BLE001 - health must answer even when the store cannot
        return "unavailable"


def serve(host: str, port: int) -> None:
    for campaign_id in campaigns.initialize_store():
        workspaces.recover_interrupted_workspace(campaign_id)
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"ryu-research listening on http://{host}:{port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
