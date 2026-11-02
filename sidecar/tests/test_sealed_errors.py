from __future__ import annotations

import unittest

from ryu_research import ryu_content_seal as content_seal
from ryu_research.sealed_errors import SEALED_ERROR
from ryu_research.server import Handler


class _Recorder:
    """Just enough of a Handler to capture what `_send_sealed` answers."""

    def __init__(self) -> None:
        self.sent: list[tuple[int, dict]] = []

    def _send(self, status: int, payload: dict) -> None:
        self.sent.append((status, payload))


def sealed_answer(exc: Exception, sealing: str) -> tuple[int, dict]:
    recorder = _Recorder()
    Handler._send_sealed(recorder, exc, sealing)  # type: ignore[arg-type]
    return recorder.sent[0]


class SealedErrorTests(unittest.TestCase):
    def test_a_bare_seal_code_is_answered_with_the_whole_sentence(self) -> None:
        status, body = sealed_answer(
            content_seal.ContentSealError("does_not_open"), "unreadable"
        )
        self.assertEqual(503, status)
        self.assertEqual(SEALED_ERROR, body["error"])
        self.assertEqual("does_not_open", body["code"])
        self.assertIn("does_not_open", body["detail"])
        self.assertEqual("unreadable", body["sealing"])

    def test_an_unavailable_store_keeps_its_detail(self) -> None:
        status, body = sealed_answer(
            content_seal.SealUnavailableError("campaigns holds sealed data"),
            "unavailable",
        )
        self.assertEqual(503, status)
        self.assertEqual(SEALED_ERROR, body["error"])
        self.assertEqual("campaigns holds sealed data", body["detail"])
        self.assertNotIn("code", body)

    def test_an_envelope_prefix_is_still_a_bad_request(self) -> None:
        status, body = sealed_answer(
            content_seal.ContentSealError("already_a_ryu_envelope"), "unreadable"
        )
        self.assertEqual(400, status)
        self.assertIn("encrypted-content prefix", body["error"])


if __name__ == "__main__":
    unittest.main()
