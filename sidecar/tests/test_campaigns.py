from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from ryu_research import campaigns, workspaces
from ryu_research.experiments import (
    AttemptOutcome,
    evaluate_stop,
    load_experiment,
    score_improved,
)

SOURCE_EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"


class CampaignStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.experiments_directory = self.root / "experiment-kinds"
        shutil.copytree(SOURCE_EXPERIMENTS_DIR, self.experiments_directory)
        self.environment = patch.dict(
            os.environ,
            {
                "RESEARCH_WORKSPACES": str(self.root),
                "RESEARCH_EXPERIMENTS": str(self.experiments_directory),
            },
        )
        self.environment.start()

    def tearDown(self) -> None:
        self.environment.stop()
        self.temporary_directory.cleanup()

    def _proposal(
        self,
        campaign_id: str,
        *,
        kind: str | None = None,
        recommendation: str = "Try the verified next variant.",
    ) -> dict:
        history = campaigns.history_projection(campaign_id)
        proposal_kind = kind or history["required_proposal_kind"]
        payload = {
            "history_digest": history["history_digest"],
            "kind": proposal_kind,
        }
        if proposal_kind == "rlm_verified":
            payload.update(
                {
                    "recommendation": recommendation,
                    "rlm_context_id": "context-1",
                    "rlm_run_id": "run-1",
                    "citations": [{"attempt": history["attempt_count"]}],
                    "successful_recursions": 1,
                }
            )
        return campaigns.create_proposal(campaign_id, **payload)["proposal"]

    def _start(
        self,
        campaign_id: str,
        hypothesis: str = "Test a freeform variant",
        *,
        budget_s: object | None = None,
    ) -> dict:
        proposal = self._proposal(campaign_id)
        return campaigns.start_attempt(
            campaign_id,
            hypothesis,
            proposal_id=proposal["id"],
            budget_s=budget_s,
        )

    def _finish(
        self,
        attempt: dict,
        *,
        status: str = "ok",
        score: object = 1.0,
    ) -> dict:
        return campaigns.finish_attempt(
            attempt["id"],
            status=status,
            score=score,
            memory_gb=0,
            elapsed_ms=5,
            commit_sha=f"commit-{attempt['id']}",
            logs_tail=f"score={score}",
        )

    def _write_experiment(
        self,
        kind: str,
        *,
        direction: str = "minimize",
        budget_s: int = 10,
        budget_min_s: int = 2,
        budget_max_s: int = 20,
        max_attempts: int = 0,
        patience: int = 0,
        min_delta: float = 0,
        max_failures: int = 0,
        target: float | None = None,
    ) -> None:
        root = self.experiments_directory / kind
        root.mkdir(parents=True)
        target_line = "" if target is None else f"early_stop_target = {target}\n"
        (root / "experiment.toml").write_text(
            'name = "Policy test"\n'
            'description = "Policy fixture"\n'
            'run_cmd = "python train.py"\n'
            'metric_regex = "score=([0-9.]+)"\n'
            f'metric_direction = "{direction}"\n'
            f"budget_s = {budget_s}\n"
            f"budget_min_s = {budget_min_s}\n"
            f"budget_max_s = {budget_max_s}\n"
            f"early_stop_max_attempts = {max_attempts}\n"
            f"early_stop_patience = {patience}\n"
            f"early_stop_min_delta = {min_delta}\n"
            f"early_stop_max_failures = {max_failures}\n"
            f"{target_line}"
            'mutable_files = ["train.py"]\n'
            "gpu_required = false\n",
            encoding="utf-8",
        )
        (root / "train.py").write_text("print('score=1')\n", encoding="utf-8")

    def test_config_validates_policy_and_pure_direction_helpers(self) -> None:
        self._write_experiment("maximize", direction="maximize", min_delta=0.1)
        config = load_experiment("maximize")
        self.assertTrue(config.is_improvement(1.2, 1.0))
        self.assertFalse(config.is_improvement(1.1, 1.0))
        self.assertTrue(score_improved(0.8, 1.0, "minimize", 0.1))
        decision = evaluate_stop(
            attempts=[AttemptOutcome("crash", "rejected", None)],
            best_score=None,
            metric_direction="maximize",
            early_stop_max_attempts=0,
            early_stop_patience=0,
            early_stop_max_failures=1,
            early_stop_target=None,
        )
        self.assertEqual("failed", decision.terminal_status)
        self.assertEqual("max_failures", decision.stop_reason)

    def test_config_rejects_invalid_budget_bounds_and_non_finite_target(self) -> None:
        self._write_experiment("invalid-order", budget_s=10, budget_min_s=11)
        with self.assertRaisesRegex(ValueError, "budget_min_s"):
            load_experiment("invalid-order")
        self._write_experiment("invalid-global", budget_s=3601, budget_max_s=3601)
        with self.assertRaisesRegex(ValueError, "within 1..3600"):
            load_experiment("invalid-global")
        self._write_experiment("invalid-target", target=float("inf"))
        with self.assertRaisesRegex(ValueError, "early_stop_target"):
            load_experiment("invalid-target")

    def test_run_records_attempts_and_enforces_the_winner(self) -> None:
        initialized = workspaces.init_workspace(
            "toy", name="Optimizer sweep", goal="Lower validation BPB"
        )
        campaign_id = initialized["campaign_id"]
        baseline_source = workspaces.read_file(campaign_id, "train.py")["content"]
        proposal = self._proposal(campaign_id)
        first = workspaces.run_experiment(
            campaign_id,
            hypothesis="Establish baseline",
            proposal_id=proposal["id"],
        )
        self.assertEqual("kept", first["decision"])

        workspaces.write_file(
            campaign_id,
            "train.py",
            baseline_source.replace("LEARNING_RATE = 0.002", "LEARNING_RATE = 2.0"),
        )
        proposal = self._proposal(campaign_id)
        second = workspaces.run_experiment(
            campaign_id,
            hypothesis="Try a deliberately unstable learning rate",
            description="Stress the reset path",
            proposal_id=proposal["id"],
        )
        self.assertEqual("rejected", second["decision"])
        self.assertEqual(
            baseline_source,
            workspaces.read_file(campaign_id, "train.py")["content"],
        )
        detail = campaigns.get_campaign(campaign_id)["campaign"]
        self.assertEqual("minimize", detail["metric_direction"])
        self.assertEqual(first["score"], detail["best_score"])
        self.assertEqual(
            ["kept", "rejected"], [a["decision"] for a in detail["attempts"]]
        )
        self.assertEqual(2, len(workspaces.read_ledger(campaign_id)["rows"]))

    def test_bounded_log_buffer_keeps_only_the_tail(self) -> None:
        buffer = workspaces._BoundedTextBuffer(8)
        buffer.append("first")
        buffer.append("-second")
        self.assertEqual("t-second", buffer.text())

    def test_process_tree_memory_sums_only_the_root_and_descendants(self) -> None:
        listing = "10 1 100\n11 10 200\n12 11 300\n99 1 9000\n"
        completed = subprocess.CompletedProcess(
            args=["ps"], returncode=0, stdout=listing, stderr=""
        )
        with patch.object(workspaces.subprocess, "run", return_value=completed):
            self.assertEqual(600, workspaces._process_tree_rss_kb(10))

    def test_timeout_stops_descendants_and_keeps_metric_with_bounded_logs(self) -> None:
        self._write_experiment(
            "process-tree",
            budget_s=1,
            budget_min_s=1,
            budget_max_s=2,
        )
        campaign_id = workspaces.init_workspace("process-tree")["campaign_id"]
        heartbeat = self.root / "descendant-heartbeat"
        program = """
import os
import subprocess
import sys
import time

heartbeat = os.environ["RESEARCH_TEST_HEARTBEAT"]
child = (
    "import os,time\\n"
    "p=os.environ['RESEARCH_TEST_HEARTBEAT']\\n"
    "while True:\\n"
    "    with open(p, 'w') as stream: stream.write(str(time.time()))\\n"
    "    time.sleep(0.05)\\n"
)
subprocess.Popen([sys.executable, "-c", child])
print("score=1", flush=True)
print("x" * 200_000, flush=True)
time.sleep(60)
"""
        workspaces.write_file(campaign_id, "train.py", program)
        proposal = self._proposal(campaign_id)
        with patch.dict(
            os.environ,
            {"RESEARCH_TEST_HEARTBEAT": str(heartbeat)},
        ):
            result = workspaces.run_experiment(
                campaign_id,
                budget_s=1,
                hypothesis="Bound the full process tree",
                proposal_id=proposal["id"],
            )
        self.assertEqual("timeout", result["status"])
        self.assertEqual(1.0, result["score"])
        self.assertLessEqual(len(result["logs_tail"]), 4_000)
        self.assertTrue(heartbeat.is_file())
        before = heartbeat.read_text(encoding="utf-8")
        time.sleep(0.3)
        self.assertEqual(before, heartbeat.read_text(encoding="utf-8"))

    def test_only_one_attempt_can_be_active(self) -> None:
        first_id = workspaces.init_workspace("toy")["campaign_id"]
        second_id = workspaces.init_workspace("toy")["campaign_id"]
        attempt = self._start(first_id)
        second_proposal = self._proposal(second_id)
        with self.assertRaises(campaigns.CampaignBusyError):
            campaigns.start_attempt(
                second_id, "second", proposal_id=second_proposal["id"]
            )
        self._finish(attempt, status="interrupted", score=None)

    def test_recovers_stale_running_attempts(self) -> None:
        campaign_id = workspaces.init_workspace("toy")["campaign_id"]
        self._start(campaign_id, "will be interrupted")
        campaigns._INITIALIZED_DATABASES.clear()
        self.assertEqual([campaign_id], campaigns.initialize_store())
        detail = campaigns.get_campaign(campaign_id)["campaign"]
        self.assertEqual("active", detail["status"])
        self.assertIsNone(detail["current_attempt_id"])
        self.assertEqual("interrupted", detail["attempts"][0]["status"])

    def test_migrates_legacy_tsv_once(self) -> None:
        workspace = self.root / "legacy-campaign"
        workspace.mkdir()
        shutil.copy2(self.experiments_directory / "toy" / "experiment.toml", workspace)
        (workspace / "results.tsv").write_text(
            "commit\tscore\tmemory_gb\tstatus\tdescription\n"
            "aaa\t1.5\t0.1\tok\tbaseline\n"
            "bbb\t2.0\t0.2\tok\tworse\n"
            "ccc\t1.25\t0.2\tok\tbetter\n",
            encoding="utf-8",
        )
        campaigns.initialize_store()
        campaigns._INITIALIZED_DATABASES.clear()
        campaigns.initialize_store()
        detail = campaigns.get_campaign("legacy-campaign")["campaign"]
        self.assertEqual(3, detail["attempt_count"])
        self.assertEqual(1.25, detail["best_score"])
        self.assertEqual(
            ["kept", "rejected", "kept"], [a["decision"] for a in detail["attempts"]]
        )

    def test_history_digest_excludes_analyses_and_proposals(self) -> None:
        campaign_id = workspaces.init_workspace("toy", goal="Find a stable minimum")[
            "campaign_id"
        ]
        before = campaigns.history_projection(campaign_id)
        campaigns.record_analysis(
            campaign_id,
            status="completed",
            query="Legacy analysis",
            response="Informational only",
            history_digest="forged",
        )
        proposal = self._proposal(campaign_id)
        after = campaigns.history_projection(campaign_id)
        self.assertEqual(before["history_digest"], after["history_digest"])
        self.assertEqual(before["history_chars"], after["history_chars"])
        self.assertGreater(after["document_chars"], before["document_chars"])
        self.assertIn('"type":"proposal"', after["document"])
        repeated = campaigns.create_proposal(
            campaign_id,
            history_digest=after["history_digest"],
            kind="freeform",
        )["proposal"]
        self.assertEqual(proposal["id"], repeated["id"])

    def test_empty_compatibility_analysis_does_not_claim_completion(self) -> None:
        campaign_id = workspaces.init_workspace("toy")["campaign_id"]
        result = campaigns.record_analysis(campaign_id, status="running")

        self.assertEqual(
            "No analysis summary recorded.", result["analysis"]["summary"]
        )

    def test_verified_proposal_requires_real_proof_and_binds_hypothesis(self) -> None:
        campaign_id = workspaces.init_workspace("toy", reasoning_threshold_chars=1)[
            "campaign_id"
        ]
        history = campaigns.history_projection(campaign_id)
        with self.assertRaisesRegex(ValueError, "rlm_verified requires"):
            campaigns.create_proposal(
                campaign_id,
                history_digest=history["history_digest"],
                kind="rlm_verified",
                recommendation="Forged",
                rlm_context_id="context",
                rlm_run_id="run",
                citations=[],
                successful_recursions=1,
            )
        proposal = self._proposal(
            campaign_id,
            kind="rlm_verified",
            recommendation="Increase momentum carefully.",
        )
        with self.assertRaisesRegex(campaigns.ReasoningRequiredError, "exactly match"):
            campaigns.start_attempt(
                campaign_id,
                "Use a different change",
                proposal_id=proposal["id"],
            )
        attempt = campaigns.start_attempt(
            campaign_id,
            "  Increase   momentum carefully. ",
            analysis_id=proposal["id"],
        )
        self.assertEqual("Increase momentum carefully.", attempt["hypothesis"])
        self.assertEqual(proposal["id"], attempt["proposal_id"])

    def test_proposal_is_one_shot_and_stale_after_history_changes(self) -> None:
        campaign_id = workspaces.init_workspace("toy")["campaign_id"]
        proposal = self._proposal(campaign_id)
        attempt = campaigns.start_attempt(
            campaign_id, "First variant", proposal_id=proposal["id"]
        )
        self._finish(attempt)
        with self.assertRaises(campaigns.ReasoningRequiredError):
            campaigns.start_attempt(
                campaign_id, "Replay lease", proposal_id=proposal["id"]
            )
        stale_digest = proposal["history_digest"]
        with self.assertRaisesRegex(campaigns.ReasoningRequiredError, "stale"):
            campaigns.create_proposal(
                campaign_id, history_digest=stale_digest, kind="freeform"
            )

    def test_budget_is_clamped_and_persisted(self) -> None:
        self._write_experiment("budgeted")
        campaign_id = campaigns.create_campaign("budget-campaign", "budgeted")["id"]
        low = self._start(campaign_id, budget_s=-99)
        self.assertEqual(2, low["budget_s"])
        self._finish(low)
        high = self._start(campaign_id, budget_s=999)
        self.assertEqual(20, high["budget_s"])
        self.assertEqual(
            20,
            campaigns.get_campaign(campaign_id, include_full=True)["campaign"][
                "attempts"
            ][1]["budget_s"],
        )

    def test_maximize_winner_and_max_attempt_stop(self) -> None:
        self._write_experiment(
            "max-two", direction="maximize", max_attempts=2, min_delta=0.1
        )
        campaign_id = campaigns.create_campaign("max-campaign", "max-two")["id"]
        first = self._finish(self._start(campaign_id), score=1.0)
        second = self._finish(self._start(campaign_id), score=1.05)
        self.assertTrue(first["improved"])
        self.assertFalse(second["improved"])
        self.assertEqual("completed", second["campaign_status"])
        self.assertEqual("max_attempts", second["stop_reason"])
        detail = campaigns.get_campaign(campaign_id)["campaign"]
        self.assertEqual(1.0, detail["best_score"])
        self.assertTrue(detail["should_stop"])
        with self.assertRaisesRegex(ValueError, "cannot accept attempts"):
            campaigns.start_attempt(
                campaign_id,
                "terminal",
                proposal_id=self._proposal(campaign_id)["id"],
            )

    def test_target_patience_and_consecutive_failure_stop_reasons(self) -> None:
        self._write_experiment("target", target=0.5)
        target_id = campaigns.create_campaign("target-campaign", "target")["id"]
        target_result = self._finish(self._start(target_id), score=0.4)
        self.assertEqual("target_reached", target_result["stop_reason"])

        self._write_experiment("patient", patience=1)
        patience_id = campaigns.create_campaign("patience-campaign", "patient")["id"]
        self._finish(self._start(patience_id), score=1.0)
        patience_result = self._finish(self._start(patience_id), score=2.0)
        self.assertEqual("patience_exhausted", patience_result["stop_reason"])

        self._write_experiment("fragile", max_failures=2)
        failure_id = campaigns.create_campaign("failure-campaign", "fragile")["id"]
        self._finish(self._start(failure_id), status="crash", score=None)
        failure_result = self._finish(
            self._start(failure_id), status="timeout", score=None
        )
        self.assertEqual("failed", failure_result["campaign_status"])
        self.assertEqual("max_failures", failure_result["stop_reason"])

    def test_record_reasoning_rejects_self_attested_proof(self) -> None:
        campaign_id = workspaces.init_workspace("toy")["campaign_id"]
        with self.assertRaises(campaigns.ProposalMigrationError):
            campaigns.record_reasoning(
                campaign_id,
                after_attempt_ordinal=0,
                rlm_context_id="context",
                rlm_run_id="run",
                recommendation="Forged proof",
            )

    def test_schema_migrates_existing_database_columns(self) -> None:
        database = self.root / "experiments.db"
        connection = sqlite3.connect(database)
        connection.executescript(
            """
            CREATE TABLE campaigns (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, experiment TEXT NOT NULL,
                goal TEXT NOT NULL DEFAULT '', status TEXT NOT NULL,
                created_at TEXT NOT NULL, started_at TEXT, updated_at TEXT NOT NULL,
                finished_at TEXT, best_score REAL, best_attempt_id INTEGER,
                current_attempt_id INTEGER
            );
            CREATE TABLE attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT, campaign_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL, status TEXT NOT NULL, decision TEXT NOT NULL,
                score REAL, memory_gb REAL, elapsed_ms INTEGER,
                started_at TEXT NOT NULL, finished_at TEXT,
                hypothesis TEXT NOT NULL DEFAULT '', description TEXT NOT NULL DEFAULT '',
                commit_sha TEXT NOT NULL DEFAULT '', logs_tail TEXT NOT NULL DEFAULT '',
                UNIQUE(campaign_id, ordinal)
            );
            CREATE TABLE analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT, campaign_id TEXT NOT NULL,
                status TEXT NOT NULL, created_at TEXT NOT NULL, finished_at TEXT,
                query TEXT NOT NULL DEFAULT '', response TEXT NOT NULL DEFAULT '',
                reasoning TEXT NOT NULL DEFAULT '', error TEXT NOT NULL DEFAULT '',
                history_digest TEXT NOT NULL DEFAULT ''
            );
            """
        )
        connection.close()
        campaigns.initialize_store()
        connection = sqlite3.connect(database)
        try:
            campaign_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(campaigns)")
            }
            attempt_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(attempts)")
            }
            analysis_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(analyses)")
            }
            version = connection.execute("PRAGMA user_version").fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(campaigns.SCHEMA_VERSION, version)
        self.assertTrue({"policy_json", "stop_reason"} <= campaign_columns)
        self.assertTrue({"proposal_id", "budget_s"} <= attempt_columns)
        self.assertTrue(
            {
                "kind",
                "citations_json",
                "successful_recursions",
                "consumed_by_attempt_id",
            }
            <= analysis_columns
        )

    def test_non_finite_scores_cannot_win(self) -> None:
        campaign_id = workspaces.init_workspace("toy")["campaign_id"]
        result = self._finish(self._start(campaign_id), score=float("nan"))
        self.assertEqual("rejected", result["decision"])
        self.assertIsNone(result["score"])

    def test_default_detail_is_bounded_and_full_detail_is_complete(self) -> None:
        campaign_id = workspaces.init_workspace("toy")["campaign_id"]
        connection = sqlite3.connect(campaigns.database_path())
        try:
            for ordinal in range(1, 13):
                connection.execute(
                    """INSERT INTO attempts(
                        campaign_id, ordinal, status, decision, started_at,
                        hypothesis, logs_tail, budget_s
                    ) VALUES (?, ?, 'ok', 'rejected', ?, ?, ?, 60)""",
                    (
                        campaign_id,
                        ordinal,
                        "2026-01-01T00:00:00.000Z",
                        f"attempt {ordinal}",
                        "x" * 1_000,
                    ),
                )
            connection.commit()
        finally:
            connection.close()
        bounded = campaigns.get_campaign(campaign_id)["campaign"]
        complete = campaigns.get_campaign(campaign_id, include_full=True)["campaign"]
        self.assertEqual(8, len(bounded["attempts"]))
        self.assertTrue(
            all(len(row["logs_tail"]) <= 400 for row in bounded["attempts"])
        )
        self.assertEqual(12, len(complete["attempts"]))

    def test_experiment_environment_scrubs_private_ryu_credentials(self) -> None:
        with patch.dict(
            os.environ,
            {
                "PATH": "kept-path",
                "RYU_RESEARCH_ENGINE_TOKEN": "engine-secret",
                "RYU_EXT_TOKEN": "extension-secret",
                "RYU_MASTER_KEY": "master-secret",
            },
        ):
            child_environment = workspaces._experiment_environment()
        self.assertEqual("kept-path", child_environment["PATH"])
        self.assertNotIn("RYU_RESEARCH_ENGINE_TOKEN", child_environment)
        self.assertNotIn("RYU_EXT_TOKEN", child_environment)
        self.assertNotIn("RYU_MASTER_KEY", child_environment)


if __name__ == "__main__":
    unittest.main()
