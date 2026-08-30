from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from ryu_research.server import Handler


class EngineAuthenticationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.environment = patch.dict(
            os.environ,
            {
                "RESEARCH_WORKSPACES": self.temporary_directory.name,
                "RYU_RESEARCH_ENGINE_TOKEN": "",
            },
        )
        self.environment.start()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.environment.stop()
        self.temporary_directory.cleanup()

    def _get(self, path: str, token: str = "") -> tuple[int, dict]:
        request = urllib.request.Request(self.base_url + path)
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            try:
                return error.code, json.loads(error.read())
            finally:
                error.close()

    def _post(self, path: str, payload: dict, token: str) -> tuple[int, dict]:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            try:
                return error.code, json.loads(error.read())
            finally:
                error.close()

    def test_health_is_public_and_protected_routes_fail_closed(self) -> None:
        self.assertEqual(200, self._get("/health")[0])
        self.assertEqual(401, self._get("/campaigns")[0])
        self.assertEqual(
            401,
            self._post(
                "/campaigns/unknown/proposals",
                {"kind": "freeform", "history_digest": "digest"},
                "wrong-token",
            )[0],
        )

    def test_token_file_is_reloaded_without_restarting_engine(self) -> None:
        self.assertEqual(401, self._get("/campaigns", "adopted-token")[0])
        (Path(self.temporary_directory.name) / ".engine-token").write_text(
            "adopted-token\n", encoding="utf-8"
        )
        status, payload = self._get("/campaigns", "adopted-token")
        self.assertEqual(200, status)
        self.assertEqual([], payload["campaigns"])

    def test_environment_token_takes_precedence(self) -> None:
        (Path(self.temporary_directory.name) / ".engine-token").write_text(
            "file-token", encoding="utf-8"
        )
        with patch.dict(os.environ, {"RYU_RESEARCH_ENGINE_TOKEN": "env-token"}):
            self.assertEqual(401, self._get("/campaigns", "file-token")[0])
            self.assertEqual(200, self._get("/campaigns", "env-token")[0])

    def test_campaign_proposal_and_history_routes(self) -> None:
        token = "route-token"
        (Path(self.temporary_directory.name) / ".engine-token").write_text(
            token, encoding="utf-8"
        )
        status, initialized = self._post(
            "/workspace/init",
            {
                "experiment": "toy",
                "name": "HTTP campaign",
                "goal": "Exercise the durable API",
                "reasoning_threshold_chars": 100_000,
            },
            token,
        )
        self.assertEqual(200, status)
        campaign_id = initialized["campaign_id"]

        history_status, history = self._get(
            f"/campaigns/{campaign_id}/history", token
        )
        self.assertEqual(200, history_status)
        self.assertEqual("freeform", history["required_proposal_kind"])
        proposal_status, envelope = self._post(
            f"/campaigns/{campaign_id}/proposals",
            {
                "history_digest": history["history_digest"],
                "kind": "freeform",
                "evidence": {"source": "human hypothesis"},
            },
            token,
        )
        self.assertEqual(200, proposal_status)
        proposal = envelope["proposal"]
        self.assertEqual("freeform", proposal["kind"])

        run_status, result = self._post(
            f"/workspace/{campaign_id}/run",
            {
                "hypothesis": "Establish a baseline",
                "proposal_id": proposal["id"],
                "budget_s": 0,
            },
            token,
        )
        self.assertEqual(200, run_status)
        self.assertEqual(1, result["budget_s"])
        replay_status, replay = self._post(
            f"/workspace/{campaign_id}/run",
            {
                "hypothesis": "Replay",
                "analysis_id": proposal["id"],
            },
            token,
        )
        self.assertEqual(409, replay_status)
        self.assertTrue(replay["reasoning_required"])

        _, detail = self._get(f"/campaigns/{campaign_id}?full=1", token)
        self.assertEqual("Exercise the durable API", detail["campaign"]["goal"])
        self.assertEqual(proposal["id"], detail["campaign"]["attempts"][0]["proposal_id"])
        self.assertEqual(1, len(detail["campaign"]["proposals"]))

    def test_verified_route_rejects_forged_proof_and_old_reasoning_route(self) -> None:
        token = "verified-route-token"
        (Path(self.temporary_directory.name) / ".engine-token").write_text(
            token, encoding="utf-8"
        )
        _, initialized = self._post(
            "/workspace/init",
            {"experiment": "toy", "reasoning_threshold_chars": 1},
            token,
        )
        campaign_id = initialized["campaign_id"]
        _, history = self._get(f"/campaigns/{campaign_id}/history", token)
        forged_status, forged = self._post(
            f"/campaigns/{campaign_id}/proposals",
            {
                "history_digest": history["history_digest"],
                "kind": "rlm_verified",
                "rlm_context_id": "context",
                "rlm_run_id": "run",
                "recommendation": "Try it",
                "citations": [],
                "successful_recursions": 1,
            },
            token,
        )
        self.assertEqual(400, forged_status)
        self.assertIn("requires", forged["error"])
        old_status, old = self._post(
            f"/campaigns/{campaign_id}/reasoning",
            {
                "after_attempt_ordinal": 0,
                "rlm_context_id": "context",
                "rlm_run_id": "run",
                "recommendation": "Self attested",
            },
            token,
        )
        self.assertEqual(400, old_status)
        self.assertIn("removed", old["error"])


if __name__ == "__main__":
    unittest.main()
