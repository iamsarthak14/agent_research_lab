"""Offline unit tests for output-protection (run directory isolation).

These tests call the actual production functions from research_agent.py.
No network access, no model inference, no downloads.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from research_agent import resolve_output_dir, write_outputs


class TestResolveOutputDir(unittest.TestCase):
    """Tests for resolve_output_dir."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.saved_cwd = os.getcwd()
        os.chdir(self.tmpdir)

    def tearDown(self):
        os.chdir(self.saved_cwd)
        shutil.rmtree(self.tmpdir)

    def test_default_creates_timestamped_run_directory(self):
        """When output_dir is None, a unique timestamped directory is created."""
        result = resolve_output_dir(None)
        # Should be under outputs/ with a timestamp directory name
        self.assertTrue(result.is_dir())
        self.assertEqual(result.parent.name, "outputs")
        # Directory name starts with 8-digit date + hyphen + 6-digit time
        name = result.name
        date_part = name.split("-")[0]
        self.assertTrue(date_part.isdigit() and len(date_part) == 8)

    def test_explicit_output_dir_used_directly(self):
        """When output_dir is provided, it is used directly."""
        result = resolve_output_dir("my_custom_output")
        self.assertTrue(result.is_dir())
        self.assertEqual(result.name, "my_custom_output")

    def test_multiple_default_calls_create_different_dirs(self):
        """Two consecutive default calls create different directories."""
        import time
        r1 = resolve_output_dir(None)
        time.sleep(1.1)  # Ensure different timestamp
        r2 = resolve_output_dir(None)
        self.assertNotEqual(r1, r2)
        # Both under outputs/
        self.assertTrue(r1.parent.samefile(Path("outputs")))
        self.assertTrue(r2.parent.samefile(Path("outputs")))

    def test_explicit_dir_preserves_previous_runs(self):
        """Explicit output_dir does not interfere with existing outputs/."""
        explicit = resolve_output_dir("explicit_run")
        outputs = Path("outputs")
        # outputs/ exists but explicit_run is separate
        self.assertTrue(outputs.is_dir())
        self.assertFalse(str(explicit).startswith(str(outputs.resolve()) + os.sep))

    def test_collision_safe_same_second(self):
        """Two calls within the same second get unique directories."""
        r1 = resolve_output_dir(None)
        # Force a collision by pre-creating the next directory name
        next_name = r1.name  # e.g. "20260101-120000"
        (r1.parent / next_name).mkdir(exist_ok=True)
        # Now resolve_output_dir should find the existing dir and append .1
        r2 = resolve_output_dir(None)
        self.assertNotEqual(r1, r2)
        # r2 should have a .N suffix
        self.assertIn(".", r2.name)
        # Both should be under outputs/
        self.assertEqual(r1.parent, r2.parent)
        self.assertEqual(r1.parent, Path("outputs"))

    def test_collision_safe_chain(self):
        """Multiple collisions within the same second get .1, .2, etc."""
        r1 = resolve_output_dir(None)
        ts = r1.name  # e.g. "20260101-120000"
        outputs = Path("outputs")
        # Pre-create 3 colliding directories
        for i in range(1, 4):
            (outputs / f"{ts}.{i}").mkdir(parents=True, exist_ok=True)
        # First call gets .4
        r2 = resolve_output_dir(None)
        self.assertEqual(r2.name, f"{ts}.4")
        # Pre-create .5
        (outputs / f"{ts}.5").mkdir(parents=True, exist_ok=True)
        # Next call gets .6
        r3 = resolve_output_dir(None)
        self.assertEqual(r3.name, f"{ts}.6")


class TestWriteOutputsPreservesPrevious(unittest.TestCase):
    """Tests that second run does not overwrite first run's results."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.saved_cwd = os.getcwd()
        os.chdir(self.tmpdir)

    def tearDown(self):
        os.chdir(self.saved_cwd)
        shutil.rmtree(self.tmpdir)

    def test_second_run_creates_new_directory(self):
        """Two runs with default output_dir create separate directories."""
        out1 = resolve_output_dir(None)
        papers1 = [{"title": "paper one", "triage": {"relevance": 3}}]
        write_outputs(papers1, ["query1"], out1)

        # Verify first run's outputs exist
        self.assertTrue((out1 / "papers.jsonl").exists())
        self.assertTrue((out1 / "paper_matrix.csv").exists())
        self.assertTrue((out1 / "research_map.md").exists())

        # Second run
        import time
        time.sleep(1.1)
        out2 = resolve_output_dir(None)
        papers2 = [{"title": "paper two", "triage": {"relevance": 5}}]
        write_outputs(papers2, ["query2"], out2)

        # Second run's outputs exist
        self.assertTrue((out2 / "papers.jsonl").exists())
        self.assertTrue((out2 / "paper_matrix.csv").exists())
        self.assertTrue((out2 / "research_map.md").exists())

        # First run's outputs are STILL THERE (not deleted or overwritten)
        self.assertTrue((out1 / "papers.jsonl").exists())
        self.assertTrue((out1 / "paper_matrix.csv").exists())
        self.assertTrue((out1 / "research_map.md").exists())

        # Contents are different
        with (out1 / "papers.jsonl").open("r") as f:
            content1 = f.read()
        with (out2 / "papers.jsonl").open("r") as f:
            content2 = f.read()
        self.assertIn("paper one", content1)
        self.assertIn("paper two", content2)
        self.assertNotIn("paper two", content1)
        self.assertNotIn("paper one", content2)

    def test_explicit_output_dir_overwrites_same_directory(self):
        """When the same explicit output_dir is used, results ARE overwritten."""
        out = resolve_output_dir("shared_outputs")
        papers1 = [{"title": "first run", "triage": {"relevance": 1}}]
        write_outputs(papers1, ["q1"], out)

        # Second run to the SAME explicit directory overwrites
        papers2 = [{"title": "second run", "triage": {"relevance": 2}}]
        write_outputs(papers2, ["q2"], out)

        with (out / "papers.jsonl").open("r") as f:
            content = f.read()
        self.assertIn("second run", content)
        self.assertNotIn("first run", content)


if __name__ == '__main__':
    unittest.main()
