"""History refresh cost: rows open once, fresh campaigns are skipped at start."""

from __future__ import annotations

import base64
import os
import shutil
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from ryu_research import campaigns, workspaces
from ryu_research import ryu_content_seal as content_seal

SOURCE_EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"
KEY = base64.b64encode(bytes([7]) * 32).decode()
ATTEMPTS = 200
LOG_BYTES = 4096


class HistoryRefreshTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        experiments_directory = self.root / "experiment-kinds"
        shutil.copytree(SOURCE_EXPERIMENTS_DIR, experiments_directory)
        self.environment = patch.dict(
            os.environ,
            {
                "RESEARCH_WORKSPACES": str(self.root / "workspaces"),
                "RESEARCH_EXPERIMENTS": str(experiments_directory),
                "RYU_PLUGIN_DATA_DIR": str(self.root),
            },
        )
        self.environment.start()
        self._restart()

    def tearDown(self) -> None:
        self.environment.stop()
        content_seal.reset_process_keys_for_tests({})
        campaigns._INITIALIZED_DATABASES.clear()
        campaigns._STORE_SEALS.clear()
        self.temporary_directory.cleanup()

    def _restart(self) -> None:
        """Simulate a fresh keyed engine process."""
        content_seal.reset_process_keys_for_tests(
            {
                content_seal.SEAL_KEY_ENV: KEY,
                content_seal.SEAL_SCOPE_ENV: "plugin:@ryu/research",
            }
        )
        campaigns._INITIALIZED_DATABASES.clear()
        campaigns._STORE_SEALS.clear()

    def _campaign_with_attempts(self, count: int) -> str:
        campaign_id = workspaces.init_workspace("toy", goal="goal")["campaign_id"]
        log = "x" * LOG_BYTES
        connection = campaigns._connect()
        try:
            with connection:
                for ordinal in range(1, count + 1):
                    ids = {"campaign_id": campaign_id, "ordinal": ordinal}
                    connection.execute(
                        """INSERT INTO attempts(campaign_id, ordinal, status, decision,
                        started_at, hypothesis, logs_tail)
                        VALUES (?, ?, 'finished', 'rejected', ?, ?, ?)""",
                        (
                            campaign_id, ordinal, campaigns._now(),
                            campaigns._seal("attempts", "hypothesis", ids, f"h{ordinal}"),
                            campaigns._seal("attempts", "logs_tail", ids, log),
                        ),
                    )
                connection.execute(
                    "UPDATE campaigns SET updated_at = ? WHERE id = ?",
                    (campaigns._now(), campaign_id),
                )
        finally:
            connection.close()
        return campaign_id

    def test_refresh_opens_each_attempt_once_and_stays_bounded_on_the_pure_path(self) -> None:
        campaign_id = self._campaign_with_attempts(ATTEMPTS)
        with patch.object(content_seal, "_LibChaCha", None):
            opened = []
            original = campaigns._opened

            def counting(table, row):
                if table == "attempts":
                    opened.append(row["ordinal"])
                return original(table, row)

            started = time.monotonic()
            with patch.object(campaigns, "_opened", side_effect=counting):
                state = campaigns.refresh_history(campaign_id)
            elapsed = time.monotonic() - started
        self.assertEqual(ATTEMPTS, state["attempt_count"])
        self.assertEqual(sorted(opened), list(range(1, ATTEMPTS + 1)))
        # ~1.7 MB through the pure-Python AEAD (open the logs, seal the file).
        self.assertLess(elapsed, 15.0)

    def test_startup_rebuilds_only_campaigns_changed_since_their_refresh(self) -> None:
        fresh = self._campaign_with_attempts(2)
        changed = self._campaign_with_attempts(2)
        campaigns.refresh_history(fresh)
        campaigns.refresh_history(changed)
        connection = sqlite3.connect(campaigns.database_path())
        try:
            with connection:
                connection.execute(
                    "UPDATE campaigns SET updated_at = ? WHERE id = ?",
                    ("9999-01-01T00:00:00.000Z", changed),
                )
        finally:
            connection.close()
        missing = self._campaign_with_attempts(1)
        campaigns.refresh_history(missing)
        (campaigns._history_path(missing)).unlink()

        self._restart()
        refreshed = []
        original = campaigns.refresh_history
        with patch.object(
            campaigns,
            "refresh_history",
            side_effect=lambda campaign_id: refreshed.append(campaign_id)
            or original(campaign_id),
        ):
            campaigns.initialize_store()
        self.assertEqual(sorted([changed, missing]), sorted(refreshed))
        self.assertTrue(campaigns._history_path(missing).is_file())

    def test_startup_reseals_a_plaintext_history_file(self) -> None:
        campaign_id = self._campaign_with_attempts(1)
        campaigns.refresh_history(campaign_id)
        path = campaigns._history_path(campaign_id)
        path.write_text('{"type":"campaign"}\n', encoding="utf-8")
        self._restart()
        campaigns.initialize_store()
        self.assertTrue(path.read_text(encoding="utf-8").startswith(content_seal.TEXT_PREFIX))

    def test_a_writer_recomputes_history_when_another_connection_committed(self) -> None:
        campaign_id = self._campaign_with_attempts(1)
        stale = campaigns.history_projection(campaign_id)["history_digest"]
        original = campaigns._history_state
        calls = []

        def racing(connection, cid, *, include_document=True):
            calls.append(cid)
            state = original(connection, cid, include_document=include_document)
            if len(calls) == 1:
                # Another writer adds an attempt before the lock is taken.
                other = campaigns._connect()
                try:
                    with other:
                        ids = {"campaign_id": cid, "ordinal": 2}
                        other.execute(
                            """INSERT INTO attempts(campaign_id, ordinal, status,
                            decision, started_at, hypothesis)
                            VALUES (?, 2, 'finished', 'rejected', ?, ?)""",
                            (cid, campaigns._now(),
                             campaigns._seal("attempts", "hypothesis", ids, "h2")),
                        )
                finally:
                    other.close()
            return state

        with patch.object(campaigns, "_history_state", side_effect=racing):
            with self.assertRaises(campaigns.ReasoningRequiredError):
                campaigns.create_proposal(campaign_id, history_digest=stale, kind="freeform")
        self.assertEqual(2, len(calls))


if __name__ == "__main__":
    unittest.main()
