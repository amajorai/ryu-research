from __future__ import annotations

import unittest

from ryu_research.workspaces import _safe_rel, workspace_dir


class WorkspacePathTests(unittest.TestCase):
    def test_workspace_id_rejects_dot_segments(self) -> None:
        for workspace_id in [".", ".."]:
            with self.assertRaises(ValueError):
                workspace_dir(workspace_id)

    def test_workspace_id_and_file_path_reject_traversal(self) -> None:
        with self.assertRaises(ValueError):
            workspace_dir("../outside")
        with self.assertRaises(ValueError):
            _safe_rel("../outside.txt")
        with self.assertRaises(ValueError):
            _safe_rel("/tmp/outside.txt")


if __name__ == "__main__":
    unittest.main()
