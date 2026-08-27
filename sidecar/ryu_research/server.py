"""Dependency-free HTTP server exposing the research contract.

Uses only the Python standard library (``http.server``), so the sidecar runs on
a bare Python with no pip install. All bodies are JSON. Routes:

    GET  /health
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
"""

from __future__ import annotations

import hmac
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import campaigns, experiments, workspaces

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

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    # ── GET ──────────────────────────────────────────────────────────────────

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/health":
                return self._send(200, {"status": "ok"})

            if not self._authorized():
                return self._send(401, {"error": "unauthorized"})

            if path == "/experiments":
                rows = [c.to_meta() for c in experiments.list_experiments()]
                return self._send(200, {"experiments": rows})

            if path == "/campaigns":
                query = parse_qs(parsed.query)
                status = (query.get("status") or [None])[0]
                return self._send(200, campaigns.list_campaigns(status))

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
                return self._send(
                    200,
                    campaigns.get_campaign(m.group(1), include_full=include_full),
                )

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
        except FileNotFoundError as exc:
            return self._send(404, {"error": str(exc)})
        except (KeyError, ValueError) as exc:
            return self._send(400, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            return self._send(500, {"error": str(exc)})

    # ── POST / PUT ───────────────────────────────────────────────────────────

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if not self._authorized():
                return self._send(401, {"error": "unauthorized"})
            body = self._read_json()

            if path == "/workspace/init":
                experiment = str(body.get("experiment", "")).strip()
                if not experiment:
                    return self._send(400, {"error": "missing 'experiment'"})
                return self._send(
                    200,
                    workspaces.init_workspace(
                        experiment,
                        name=str(body.get("name", "")),
                        goal=str(body.get("goal", "")),
                        reasoning_threshold_chars=body.get(
                            "reasoning_threshold_chars"
                        ),
                    ),
                )

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
        except Exception as exc:  # noqa: BLE001
            return self._send(500, {"error": str(exc)})

    def do_PUT(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if not self._authorized():
                return self._send(401, {"error": "unauthorized"})
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
        except PermissionError as exc:
            # write_file rejects paths outside the declared mutable_files.
            return self._send(403, {"error": str(exc)})
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            return self._send(400, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            return self._send(500, {"error": str(exc)})


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
