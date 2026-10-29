"""Regression tests for per-principal campaign ownership in the engine API.

On a shared node, a user's agent must not enumerate or read another user's
campaigns (goal, attempts, analyses, logs) or act on their workspace, and a
request without a verified caller scope must fail closed.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from ryu_research import campaigns, workspaces
from ryu_research.server import Handler

TOKEN = "ownership-token"


def _caller(user_id: str | None, org_id: str = "org-1") -> dict:
    if user_id is None:
        return {"X-Ryu-Research-Caller": "node"}
    return {
        "X-Ryu-Research-Caller": "user",
        "X-Ryu-Research-Caller-User": user_id,
        "X-Ryu-Research-Caller-Org": org_id,
    }


class CampaignOwnershipTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.environment = patch.dict(
            os.environ,
            {
                "RESEARCH_WORKSPACES": self.temporary_directory.name,
                "RYU_RESEARCH_ENGINE_TOKEN": TOKEN,
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

    def _request(
        self,
        method: str,
        path: str,
        caller: dict,
        payload: dict | None = None,
    ) -> tuple[int, dict]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={
                "Authorization": f"Bearer {TOKEN}",
                "Content-Type": "application/json",
                **caller,
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            try:
                return error.code, json.loads(error.read())
            finally:
                error.close()

    def _init(self, caller: dict, goal: str) -> dict:
        status, body = self._request(
            "POST", "/workspace/init", caller, {"experiment": "toy", "goal": goal}
        )
        self.assertEqual(200, status, body)
        return body

    def test_users_cannot_enumerate_read_or_mutate_each_others_campaigns(self) -> None:
        alice, bob = _caller("alice"), _caller("bob")
        private = self._init(alice, "alice private goal")
        campaign_id = private["campaign_id"]

        status, listing = self._request("GET", "/campaigns", bob)
        self.assertEqual(200, status)
        self.assertEqual([], listing["campaigns"])
        self.assertNotIn("alice private goal", json.dumps(listing))

        foreign_requests = [
            ("GET", f"/campaigns/{campaign_id}", None),
            ("GET", f"/campaigns/{campaign_id}?full=1", None),
            ("GET", f"/campaigns/{campaign_id}/history", None),
            ("GET", f"/workspace/{campaign_id}/file?path=train.py", None),
            ("GET", f"/workspace/{campaign_id}/ledger", None),
            ("PUT", f"/workspace/{campaign_id}/file", {"path": "train.py", "content": "x"}),
            (
                "POST",
                f"/workspace/{campaign_id}/run",
                {"proposal_id": 1, "workspace_tree": private["workspace_tree"]},
            ),
            ("POST", f"/workspace/{campaign_id}/git", {"action": "reset"}),
            ("POST", f"/workspace/{campaign_id}/ledger", {"commit": "x", "status": "ok"}),
            ("POST", f"/campaigns/{campaign_id}/proposals", {"kind": "freeform"}),
            ("POST", f"/campaigns/{campaign_id}/analyses", {"status": "completed"}),
            ("POST", f"/campaigns/{campaign_id}/status", {"status": "cancelled"}),
            ("POST", f"/campaigns/{campaign_id}/finish", {"status": "cancelled"}),
        ]
        for method, path, payload in foreign_requests:
            status, body = self._request(method, path, bob, payload)
            self.assertEqual(404, status, (method, path, body))
            self.assertNotIn("alice private goal", json.dumps(body))

        # The same user in another organization is a different principal.
        status, _ = self._request("GET", f"/campaigns/{campaign_id}", _caller("alice", "org-2"))
        self.assertEqual(404, status)

        # The owner still sees and drives its campaign; nothing was mutated.
        status, detail = self._request("GET", f"/campaigns/{campaign_id}", alice)
        self.assertEqual(200, status)
        self.assertEqual("alice private goal", detail["campaign"]["goal"])
        self.assertEqual("active", detail["campaign"]["status"])
        self.assertEqual(private["workspace_tree"], detail["campaign"]["workspace_tree"])
        status, listing = self._request("GET", "/campaigns", alice)
        self.assertEqual([campaign_id], [row["id"] for row in listing["campaigns"]])

        # The node operator (personal node / node token) sees every campaign.
        status, listing = self._request("GET", "/campaigns", _caller(None))
        self.assertIn(campaign_id, [row["id"] for row in listing["campaigns"]])

    def test_requests_without_a_verified_scope_fail_closed(self) -> None:
        campaign_id = self._init(_caller("alice"), "goal")["campaign_id"]
        for caller in (
            {},
            {"X-Ryu-Research-Caller": "unresolved"},
            {"X-Ryu-Research-Caller": "user"},
            {"X-Ryu-Research-Caller": "node", "X-Ryu-Research-Caller-User": "mallory"},
        ):
            status, _ = self._request("GET", "/campaigns", caller)
            self.assertEqual(403, status, caller)
            status, _ = self._request("GET", f"/campaigns/{campaign_id}", caller)
            self.assertEqual(403, status, caller)
            status, _ = self._request(
                "POST", "/workspace/init", caller, {"experiment": "toy"}
            )
            self.assertEqual(403, status, caller)
        # The public catalog needs no principal.
        self.assertEqual(200, self._request("GET", "/experiments", {})[0])

    def test_rows_created_before_ownership_are_quarantined_to_the_node_operator(self) -> None:
        legacy = workspaces.init_workspace("toy", goal="pre-ownership goal")
        connection = sqlite3.connect(campaigns.database_path())
        try:
            owner = connection.execute(
                "SELECT owner_kind, owner_user_id FROM campaigns WHERE id = ?",
                (legacy["campaign_id"],),
            ).fetchone()
        finally:
            connection.close()
        self.assertEqual(("legacy", ""), owner)

        status, listing = self._request("GET", "/campaigns", _caller("alice"))
        self.assertEqual([], listing["campaigns"])
        status, _ = self._request(
            "GET", f"/campaigns/{legacy['campaign_id']}", _caller("alice")
        )
        self.assertEqual(404, status)
        status, detail = self._request(
            "GET", f"/campaigns/{legacy['campaign_id']}", _caller(None)
        )
        self.assertEqual(200, status)
        self.assertEqual("pre-ownership goal", detail["campaign"]["goal"])


class OwnershipMigrationTests(unittest.TestCase):
    def test_existing_database_gains_quarantined_owner_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"RESEARCH_WORKSPACES": directory}
        ):
            path = Path(directory) / campaigns.DATABASE_FILE
            connection = sqlite3.connect(path)
            connection.executescript(
                """
                CREATE TABLE campaigns (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, experiment TEXT NOT NULL,
                    goal TEXT NOT NULL DEFAULT '', status TEXT NOT NULL,
                    created_at TEXT NOT NULL, started_at TEXT, updated_at TEXT NOT NULL,
                    finished_at TEXT, best_score REAL, best_attempt_id INTEGER,
                    current_attempt_id INTEGER
                );
                INSERT INTO campaigns(id, name, experiment, goal, status, created_at, updated_at)
                VALUES ('old', 'Old', 'toy', 'old goal', 'active', 't', 't');
                """
            )
            connection.close()
            campaigns._INITIALIZED_DATABASES.clear()
            self.assertEqual(
                [], campaigns.list_campaigns(scope=campaigns.CallerScope("user", "alice"))["campaigns"]
            )
            visible = campaigns.list_campaigns(scope=campaigns.CallerScope("node"))
            self.assertEqual(["old"], [row["id"] for row in visible["campaigns"]])
            with self.assertRaises(FileNotFoundError):
                campaigns.authorize_campaign("old", campaigns.CallerScope("user", "alice"))


if __name__ == "__main__":
    unittest.main()
