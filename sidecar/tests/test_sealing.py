"""Research content is sealed at rest: database, history file and ledger."""

from __future__ import annotations

import base64
import os
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ryu_research import campaigns, workspaces
from ryu_research import ryu_content_seal as content_seal

SOURCE_EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"
GOAL = "goal-secret-lower-bpb"
HYPOTHESIS = "hypothesis-secret-variant"
DESCRIPTION = "description-secret-reset"
KEY = base64.b64encode(bytes([5]) * 32).decode()


class SealingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.workspaces = self.root / "workspaces"
        self.experiments_directory = self.root / "experiment-kinds"
        shutil.copytree(SOURCE_EXPERIMENTS_DIR, self.experiments_directory)
        self.environment = patch.dict(
            os.environ,
            {
                "RESEARCH_WORKSPACES": str(self.workspaces),
                "RESEARCH_EXPERIMENTS": str(self.experiments_directory),
                "RYU_PLUGIN_DATA_DIR": str(self.root),
            },
        )
        self.environment.start()
        self._use_keys({content_seal.SEAL_KEY_ENV: KEY, content_seal.SEAL_SCOPE_ENV: "plugin:@ryu/research"})

    def tearDown(self) -> None:
        self.environment.stop()
        content_seal.reset_process_keys_for_tests({})
        campaigns._INITIALIZED_DATABASES.clear()
        campaigns._STORE_SEALS.clear()
        self.temporary_directory.cleanup()

    def _use_keys(self, key_env: dict[str, str]) -> None:
        """Simulate a fresh engine process with this delivered key environment."""
        content_seal.reset_process_keys_for_tests(key_env)
        campaigns._INITIALIZED_DATABASES.clear()
        campaigns._STORE_SEALS.clear()

    def _run_campaign(self) -> str:
        campaign_id = workspaces.init_workspace("toy", goal=GOAL)["campaign_id"]
        history = campaigns.history_projection(campaign_id)
        proposal = campaigns.create_proposal(
            campaign_id, history_digest=history["history_digest"], kind="freeform"
        )["proposal"]
        # The child prints whatever key it can see; none must reach it.
        source = workspaces.read_file(campaign_id, "train.py")["content"]
        written = workspaces.write_file(
            campaign_id,
            "train.py",
            "import os\nprint('child-key=' + repr(os.environ.get('RYU_SEAL_KEY')))\n" + source,
        )
        with patch.dict(os.environ, {"RYU_SEAL_KEY": "leaked-into-engine-env"}):
            result = workspaces.run_experiment(
                campaign_id,
                hypothesis=HYPOTHESIS,
                description=DESCRIPTION,
                proposal_id=proposal["id"],
                workspace_tree=written["workspace_tree"],
            )
        self.assertIn("child-key=None", result["logs_tail"])
        return campaign_id

    def _disk(self) -> bytes:
        database = campaigns.database_path()
        raw = database.read_bytes()
        wal = Path(f"{database}-wal")
        if wal.exists():
            raw += wal.read_bytes()
        return raw

    def test_campaign_content_is_sealed_and_reads_back(self) -> None:
        campaign_id = self._run_campaign()
        detail = campaigns.get_campaign(campaign_id, include_full=True)["campaign"]
        self.assertEqual(GOAL, detail["goal"])
        self.assertEqual(HYPOTHESIS, detail["attempts"][0]["hypothesis"])
        self.assertIn("val_bpb", detail["attempts"][0]["logs_tail"])
        ledger_rows = workspaces.read_ledger(campaign_id)["rows"]
        self.assertEqual(DESCRIPTION, ledger_rows[0]["description"])
        self.assertIn(GOAL, campaigns.history_projection(campaign_id)["document"])

        connection = sqlite3.connect(campaigns.database_path())
        goal, = connection.execute("SELECT goal FROM campaigns").fetchone()
        logs, hypothesis = connection.execute("SELECT logs_tail, hypothesis FROM attempts").fetchone()
        connection.close()
        for value in (goal, logs, hypothesis):
            self.assertTrue(value.startswith(content_seal.TEXT_PREFIX))
        for secret in (GOAL, HYPOTHESIS, b"val_bpb".decode()):
            self.assertNotIn(secret.encode(), self._disk())

        workspace = self.workspaces / campaign_id
        history_file = (workspace / "history.jsonl").read_text(encoding="utf-8")
        self.assertTrue(history_file.startswith(content_seal.TEXT_PREFIX))
        self.assertNotIn(GOAL, history_file)
        self.assertIn(GOAL, workspaces.read_file(campaign_id, "history.jsonl")["content"])
        ledger_text = (workspace / "results.tsv").read_text(encoding="utf-8")
        self.assertNotIn(DESCRIPTION, ledger_text)
        self.assertIn(DESCRIPTION, workspaces.read_file(campaign_id, "results.tsv")["content"])

    def test_idempotent_proposal_matches_sealed_text(self) -> None:
        campaign_id = workspaces.init_workspace("toy", reasoning_threshold_chars=1)["campaign_id"]
        history = campaigns.history_projection(campaign_id)
        payload = {
            "history_digest": history["history_digest"],
            "kind": "rlm_verified",
            "recommendation": "Try a smaller learning rate",
            "rlm_context_id": "context-1",
            "rlm_run_id": "run-1",
            "citations": [{"attempt": 0}],
            "successful_recursions": 1,
        }
        first = campaigns.create_proposal(campaign_id, **payload)["proposal"]
        again = campaigns.create_proposal(campaign_id, **payload)["proposal"]
        self.assertEqual(first["id"], again["id"])
        self.assertEqual("Try a smaller learning rate", again["recommendation"])

    def test_plaintext_from_an_older_core_is_sealed_once_a_key_arrives(self) -> None:
        self._use_keys({})  # compat: run by Core, no key delivered
        campaign_id = self._run_campaign()
        self.assertIn(GOAL.encode(), self._disk())
        workspace = self.workspaces / campaign_id
        self.assertIn(DESCRIPTION, (workspace / "results.tsv").read_text(encoding="utf-8"))
        self.assertEqual(
            "waiting for a Ryu update", campaigns.sealing_health()["sealing"]
        )

        self._use_keys({content_seal.SEAL_KEY_ENV: KEY, content_seal.SEAL_SCOPE_ENV: "plugin:@ryu/research"})
        detail = campaigns.get_campaign(campaign_id, include_full=True)["campaign"]
        self.assertEqual(GOAL, detail["goal"])
        self.assertEqual("ok", campaigns.sealing_health()["sealing"])
        for secret in (GOAL, HYPOTHESIS):
            self.assertNotIn(secret.encode(), self._disk())
        self.assertNotIn(DESCRIPTION, (workspace / "results.tsv").read_text(encoding="utf-8"))
        self.assertNotIn(GOAL, (workspace / "history.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(DESCRIPTION, workspaces.read_ledger(campaign_id)["rows"][0]["description"])
        # A second start changes nothing (idempotent).
        before = (workspace / "results.tsv").read_bytes()
        self._use_keys({content_seal.SEAL_KEY_ENV: KEY, content_seal.SEAL_SCOPE_ENV: "plugin:@ryu/research"})
        campaigns.list_campaigns()
        self.assertEqual(before, (workspace / "results.tsv").read_bytes())

    def test_losing_the_key_refuses_writes_instead_of_writing_plaintext(self) -> None:
        campaign_id = self._run_campaign()
        self._use_keys({})
        with self.assertRaises(content_seal.SealUnavailableError):
            campaigns.create_campaign("c-2", "toy", goal="new goal")
        with self.assertRaises(content_seal.ContentSealError):
            campaigns.get_campaign(campaign_id)
        self.assertEqual("no key", campaigns.sealing_health()["sealing"])


if __name__ == "__main__":
    unittest.main()
