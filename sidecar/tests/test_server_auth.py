from __future__ import annotations

import json
import http.client
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from ryu_research import campaigns, workspaces
from ryu_research.server import Handler, MAX_REQUEST_BODY_BYTES
from ryu_research.workspaces import MAX_WORKSPACE_FILE_BYTES

NODE_CALLER = {"X-Ryu-Research-Caller": "node"}


def user_caller(user_id: str, org_id: str = "org-1") -> dict:
    return {
        "X-Ryu-Research-Caller": "user",
        "X-Ryu-Research-Caller-User": user_id,
        "X-Ryu-Research-Caller-Org": org_id,
    }


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

    def _get(
        self, path: str, token: str = "", caller: dict | None = None
    ) -> tuple[int, dict]:
        request = urllib.request.Request(
            self.base_url + path, headers=NODE_CALLER if caller is None else caller
        )
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

    def _post(
        self, path: str, payload: dict, token: str, caller: dict | None = None
    ) -> tuple[int, dict]:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                **(NODE_CALLER if caller is None else caller),
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

    def test_health_reports_only_the_key_state_and_shutdown_needs_the_token(self) -> None:
        status, body = self._get("/health")
        self.assertEqual(200, status)
        self.assertEqual({"status", "sealing"}, set(body))
        with patch.dict(os.environ, {"RYU_RESEARCH_ENGINE_TOKEN": "engine-token"}):
            status, _ = self._post("/admin/shutdown", {}, "wrong-token")
            self.assertEqual(401, status)
            status, body = self._post("/admin/shutdown", {}, "engine-token")
        self.assertEqual(202, status)
        self.thread.join(timeout=5)
        self.assertFalse(self.thread.is_alive())
        # A stopping engine keeps the run lock so no attempt starts after the
        # check; this process lives on, so hand it back for the next test.
        self.assertTrue(workspaces._RUN_LOCK.locked())
        workspaces._RUN_LOCK.release()
        # tearDown shuts the (already stopped) server down again.
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def test_shutdown_is_refused_while_an_experiment_holds_the_run_lock(self) -> None:
        with patch.dict(os.environ, {"RYU_RESEARCH_ENGINE_TOKEN": "engine-token"}):
            self.assertTrue(workspaces._RUN_LOCK.acquire(blocking=False))
            try:
                status, body = self._post("/admin/shutdown", {}, "engine-token")
            finally:
                workspaces._RUN_LOCK.release()
            self.assertEqual(409, status)
            self.assertFalse(body["stopping"])
            self.assertTrue(self.thread.is_alive())
            self.assertEqual(200, self._get("/health")[0])
            self.assertEqual(200, self._get("/experiments", "engine-token")[0])

    def test_shutdown_is_refused_while_the_store_records_a_running_attempt(self) -> None:
        with patch.dict(os.environ, {"RYU_RESEARCH_ENGINE_TOKEN": "engine-token"}):
            with patch.object(campaigns, "has_running_attempt", return_value=True):
                status, _ = self._post("/admin/shutdown", {}, "engine-token")
            self.assertEqual(409, status)
            # The refusal hands the run lock back: attempts can still start.
            self.assertFalse(workspaces._RUN_LOCK.locked())
            self.assertEqual(200, self._get("/health")[0])

    def _put(
        self, path: str, payload: dict, token: str, caller: dict | None = None
    ) -> tuple[int, dict]:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                **(NODE_CALLER if caller is None else caller),
            },
            method="PUT",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            try:
                return error.code, json.loads(error.read())
            finally:
                error.close()

    def _post_declared_limit(self, path: str, length: int, token: str) -> tuple[int, dict]:
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        try:
            connection.putrequest("POST", path)
            connection.putheader("Authorization", f"Bearer {token}")
            connection.putheader("Content-Type", "application/json")
            for name, value in NODE_CALLER.items():
                connection.putheader(name, value)
            connection.putheader("Content-Length", str(length))
            connection.endheaders()
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def _post_chunked_limit(self, path: str, length: int, token: str) -> tuple[int, dict]:
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        try:
            connection.putrequest("POST", path)
            connection.putheader("Authorization", f"Bearer {token}")
            connection.putheader("Content-Type", "application/json")
            for name, value in NODE_CALLER.items():
                connection.putheader(name, value)
            connection.putheader("Transfer-Encoding", "chunked")
            connection.endheaders()
            # The server rejects the chunk from its size line before reading its
            # payload, so the test need not write an oversized body to the socket.
            connection.send(f"{length:X}\r\n".encode("ascii"))
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

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

    def test_engine_proof_answers_a_challenge_without_receiving_the_bearer(self) -> None:
        # Known-answer vector shared with the Rust supervisor's
        # `engine_proof_matches_the_python_engine` test.
        token = "engine-proof-test-token"
        challenge = "0123456789abcdef" * 2
        expected = "35b921cca206a80f65bc2b3c65f75a4136692b293e9f7d2846362e8a05ebeb00"
        self.assertEqual(503, self._get_challenge(challenge)[0])
        with patch.dict(os.environ, {"RYU_RESEARCH_ENGINE_TOKEN": token}):
            status, body = self._get_challenge(challenge)
            self.assertEqual(200, status)
            self.assertEqual(expected, body["proof"])
            self.assertNotIn(token, json.dumps(body))
            # Only a well-formed hex nonce is answered.
            for bad in ("", "short", "Z" * 32, "0" * 129):
                self.assertEqual(400, self._get_challenge(bad)[0], bad)

    def _get_challenge(self, challenge: str) -> tuple[int, dict]:
        return self._get("/engine-proof", caller={"X-Ryu-Engine-Challenge": challenge})

    def test_declared_body_limit_rejects_before_json_parse(self) -> None:
        token = "body-limit-token"
        (Path(self.temporary_directory.name) / ".engine-token").write_text(
            token, encoding="utf-8"
        )
        status, payload = self._post_declared_limit(
            "/workspace/init", MAX_REQUEST_BODY_BYTES + 1, token
        )
        self.assertEqual(413, status)
        self.assertIn("too large", payload["error"])

    def test_chunked_body_limit_rejects_before_full_read(self) -> None:
        token = "chunked-limit-token"
        (Path(self.temporary_directory.name) / ".engine-token").write_text(
            token, encoding="utf-8"
        )
        status, payload = self._post_chunked_limit(
            "/workspace/init", MAX_REQUEST_BODY_BYTES + 1, token
        )
        self.assertEqual(413, status)
        self.assertIn("too large", payload["error"])

    def test_mutable_file_admission_rejects_oversized_content(self) -> None:
        token = "file-limit-token"
        (Path(self.temporary_directory.name) / ".engine-token").write_text(
            token, encoding="utf-8"
        )
        status, initialized = self._post(
            "/workspace/init", {"experiment": "toy"}, token
        )
        self.assertEqual(200, status)
        write_status, payload = self._put(
            f"/workspace/{initialized['workspace_id']}/file",
            {"path": "train.py", "content": "x" * (MAX_WORKSPACE_FILE_BYTES + 1)},
            token,
        )
        self.assertEqual(413, write_status)
        self.assertIn("limit", payload["error"])

    def test_active_campaign_admission_rejects_before_workspace_creation(self) -> None:
        token = "campaign-limit-token"
        (Path(self.temporary_directory.name) / ".engine-token").write_text(
            token, encoding="utf-8"
        )
        with patch(
            "ryu_research.server.campaigns.list_campaigns",
            return_value={"campaigns": [{} for _ in range(8)]},
        ):
            status, payload = self._post(
                "/workspace/init", {"experiment": "toy"}, token
            )
        self.assertEqual(429, status)
        self.assertIn("active campaign", payload["error"])

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
                "improvementRunId": "improvement-http",
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
                "workspace_tree": initialized["workspace_tree"],
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
                "workspace_tree": result["next_workspace_tree"],
            },
            token,
        )
        self.assertEqual(409, replay_status)
        self.assertTrue(replay["reasoning_required"])

        _, detail = self._get(f"/campaigns/{campaign_id}?full=1", token)
        self.assertEqual("Exercise the durable API", detail["campaign"]["goal"])
        self.assertEqual(
            "improvement-http", detail["campaign"]["improvement_run_id"]
        )
        self.assertEqual(proposal["id"], detail["campaign"]["attempts"][0]["proposal_id"])
        self.assertEqual(1, len(detail["campaign"]["proposals"]))

    def test_cancellation_refuses_an_active_attempt_then_retries_and_survives_reopen(self) -> None:
        token = "cancel-route-token"
        (Path(self.temporary_directory.name) / ".engine-token").write_text(
            token, encoding="utf-8"
        )
        status, initialized = self._post(
            "/workspace/init", {"experiment": "toy"}, token
        )
        self.assertEqual(200, status)
        campaign_id = initialized["campaign_id"]

        history_status, history = self._get(
            f"/campaigns/{campaign_id}/history", token
        )
        self.assertEqual(200, history_status)
        proposal_status, envelope = self._post(
            f"/campaigns/{campaign_id}/proposals",
            {
                "history_digest": history["history_digest"],
                "kind": "freeform",
            },
            token,
        )
        self.assertEqual(200, proposal_status)
        attempt = campaigns.start_attempt(
            campaign_id,
            "Keep the campaign busy",
            proposal_id=envelope["proposal"]["id"],
        )

        refused_status, refused = self._post(
            f"/campaigns/{campaign_id}/finish", {"status": "cancelled"}, token
        )
        self.assertEqual(409, refused_status)
        self.assertIn("attempt is running", refused["error"])
        _, unchanged = self._get(f"/campaigns/{campaign_id}?full=1", token)
        self.assertEqual("running", unchanged["campaign"]["status"])

        campaigns.finish_attempt(
            attempt["id"],
            status="interrupted",
            score=None,
            memory_gb=0,
            elapsed_ms=1,
            commit_sha="",
            logs_tail="cancel fixture released",
        )
        retry_status, cancelled = self._post(
            f"/campaigns/{campaign_id}/finish", {"status": "cancelled"}, token
        )
        self.assertEqual(200, retry_status)
        self.assertEqual("cancelled", cancelled["campaign"]["status"])
        self.assertEqual("manual_cancelled", cancelled["campaign"]["stop_reason"])

        campaigns._INITIALIZED_DATABASES.clear()
        campaigns.initialize_store()
        _, reopened = self._get(f"/campaigns/{campaign_id}?full=1", token)
        self.assertEqual("cancelled", reopened["campaign"]["status"])
        self.assertEqual("manual_cancelled", reopened["campaign"]["stop_reason"])

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
