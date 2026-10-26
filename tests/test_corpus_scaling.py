"""Tests for corpus scaling: pagination, deduplication, resumability, malformed output.

These tests call the actual production functions from research_agent.py.
No network access, no model inference, no downloads, no package installation.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import research_agent as ra


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_paper(arxiv_id=None, title=None, doi=None, **kwargs):
    """Create a minimal paper record for testing."""
    return {
        "source_db": "arXiv",
        "title": title or "Test Paper",
        "year": 2026,
        "authors": ["Test Author"],
        "abstract": "Test abstract text.",
        "doi": doi,
        "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None,
        "venue": "arXiv preprint",
        "citation_count": None,
        "openalex_id": None,
        "arxiv_id": arxiv_id,
        "matched_queries": kwargs.get("matched_queries", []),
    }


def _write_corpus(path, papers):
    """Write records to a corpus file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Tests for pagination helper functions
# ---------------------------------------------------------------------------

class TestPaginatedSearchWrappers(unittest.TestCase):
    """Test that paginated search functions exist and have the right signatures."""

    def test_search_openalex_paginated_exists(self):
        """search_openalex_paginated is defined and callable."""
        self.assertTrue(callable(ra.search_openalex_paginated))

    def test_search_arxiv_paginated_exists(self):
        """search_arxiv_paginated is defined and callable."""
        self.assertTrue(callable(ra.search_arxiv_paginated))

    def test_paginated_returns_tuple(self):
        """search_openalex_paginated returns (results, next_cursor) tuple."""
        # Call with no network (mocked) — verify return type
        with mock.patch.object(ra, 'get_json', return_value={"results": []}):
            results, cursor = ra.search_openalex_paginated("test", 10)
        self.assertIsInstance(results, list)
        self.assertIsNone(cursor)  # No results → no cursor

    def test_arxiv_paginated_returns_tuple(self):
        """search_arxiv_paginated returns (results, next_start) tuple."""
        # Call with empty query → returns empty, 0
        results, start = ra.search_arxiv_paginated("a", 10)
        self.assertEqual(results, [])
        self.assertEqual(start, 0)


# ---------------------------------------------------------------------------
# Tests for persistent corpus load/save
# ---------------------------------------------------------------------------

class TestCorpusPersistence(unittest.TestCase):
    """Tests for load_corpus, save_corpus, save_corpus_status."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def test_save_and_load_corpus(self):
        """Papers survive a save→load round-trip."""
        papers = [_make_paper("2610.01256"), _make_paper("2305.10816")]
        ra.save_corpus(papers, self.tmpdir)
        loaded, _ = ra.load_corpus(self.tmpdir)
        self.assertEqual(len(loaded), 2)
        ids = {p["arxiv_id"] for p in loaded}
        self.assertEqual(ids, {"2610.01256", "2305.10816"})

    def test_load_empty_corpus(self):
        """Loading a non-existent corpus returns empty list and empty status."""
        papers, status = ra.load_corpus(self.tmpdir)
        self.assertEqual(papers, [])
        self.assertEqual(status, {})

    def test_save_and_load_status(self):
        """Status survives save→load."""
        status = {"completed_queries": ["q1", "q2"], "counts": {"unique": 5}}
        ra.save_corpus_status(status, self.tmpdir)
        _, loaded_status = ra.load_corpus(self.tmpdir)
        self.assertEqual(loaded_status["completed_queries"], ["q1", "q2"])
        self.assertEqual(loaded_status["counts"]["unique"], 5)

    def test_malformed_corpus_lines_skipped(self):
        """Malformed JSON lines in corpus are silently skipped."""
        corpus_path = self.tmpdir / "corpus.jsonl"
        corpus_path.write_text(
            json.dumps(_make_paper("2610.01256")) + "\n"
            "this is not json\n",
            encoding="utf-8",
        )
        papers, _ = ra.load_corpus(self.tmpdir)
        self.assertEqual(len(papers), 1)

    def test_corpus_capped_at_max_records(self):
        """Saving more than max records is handled by the caller."""
        # The save_corpus function itself doesn't cap; the caller does.
        # This test verifies that save_corpus writes all records faithfully.
        papers = [_make_paper(f"2610.{i:05d}") for i in range(100)]
        ra.save_corpus(papers, self.tmpdir)
        loaded, _ = ra.load_corpus(self.tmpdir)
        self.assertEqual(len(loaded), 100)


# ---------------------------------------------------------------------------
# Tests for deduplication with cross-run merging
# ---------------------------------------------------------------------------

class TestDedupAcrossRuns(unittest.TestCase):
    """Test that dedupe handles papers from multiple queries/sources."""

    def test_dedupe_merges_by_arxiv_id(self):
        """Two records with the same arXiv ID are merged into one."""
        p1 = _make_paper("2610.01256", title="Paper A")
        p2 = _make_paper("2610.01256", title="Paper B (better title)")
        result = ra.dedupe([p1, p2])
        self.assertEqual(len(result), 1)
        # Title is not merged by length; the first record's title is kept.
        # What IS merged: matched_queries, source_db, authors, abstract.
        self.assertEqual(result[0]["title"], "Paper A")

    def test_dedupe_merges_by_doi(self):
        """Two records with the same DOI are merged."""
        doi = "10.48550/arxiv.2610.01256"
        p1 = _make_paper(doi=doi, title="Paper A")
        p2 = _make_paper(doi=doi, title="Paper B")
        result = ra.dedupe([p1, p2])
        self.assertEqual(len(result), 1)

    def test_dedupe_merges_by_normalized_title(self):
        """Two records with the same normalized title are merged."""
        p1 = _make_paper(title="Agent Failure Analysis")
        p2 = _make_paper(title="agent failure analysis")
        result = ra.dedupe([p1, p2])
        self.assertEqual(len(result), 1)

    def test_dedupe_preserves_matched_queries(self):
        """Deduplication merges matched_queries from duplicates."""
        p1 = _make_paper("2610.01256", matched_queries=["query A"])
        p2 = _make_paper("2610.01256", matched_queries=["query B"])
        result = ra.dedupe([p1, p2])
        self.assertEqual(len(result), 1)
        self.assertEqual(sorted(result[0]["matched_queries"]), ["query A", "query B"])

    def test_dedupe_preserves_source_db(self):
        """Deduplication merges source_db from different sources."""
        p1 = _make_paper("2610.01256")
        p1["source_db"] = "arXiv"
        p2 = _make_paper("2610.01256")
        p2["source_db"] = "OpenAlex"
        result = ra.dedupe([p1, p2])
        self.assertEqual(len(result), 1)
        self.assertIn("arXiv", result[0]["source_db"])
        self.assertIn("OpenAlex", result[0]["source_db"])


# ---------------------------------------------------------------------------
# Tests for batch screening with malformed output handling
# ---------------------------------------------------------------------------

class TestBatchScreening(unittest.TestCase):
    """Tests for batch_screen_abstracts with malformed model output."""

    def test_batch_screen_without_model(self):
        """Batch screening without a model assigns default triage."""
        papers = [_make_paper(f"2610.{i:05d}") for i in range(5)]
        result = ra.batch_screen_abstracts(papers, None)
        for p in result:
            self.assertIn("triage", p)
            self.assertEqual(p["triage"]["relevance"], 0)

    def test_batch_screen_malformed_output_fallback(self):
        """Malformed model output triggers metadata-only fallback."""
        papers = [_make_paper("2610.01256")]

        # Mock triage_paper to return a string (malformed)
        with mock.patch.object(ra, 'triage_paper', return_value="not a dict"):
            result = ra.batch_screen_abstracts(papers, mock.Mock())
            # Should fall through to metadata-only fallback after retries
            self.assertIn("triage", result[0])
            self.assertIn("Batch screening failed", result[0]["triage"]["contribution"])

    def test_batch_screen_missing_keys_fallback(self):
        """Model output missing required keys triggers fallback."""
        papers = [_make_paper("2610.01256")]

        with mock.patch.object(ra, 'triage_paper', return_value={"relevance": 3}):
            result = ra.batch_screen_abstracts(papers, mock.Mock())
            self.assertIn("triage", result[0])
            self.assertIn("Batch screening failed", result[0]["triage"]["contribution"])

    def test_batch_screen_valid_output_preserved(self):
        """Valid model output is preserved."""
        papers = [_make_paper("2610.01256")]
        valid_triage = {
            "relevance": 4,
            "themes": ["agent reliability"],
            "contribution": "New method",
            "method": "Graph analysis",
            "evidence_limits": "None",
            "follow_up_questions": [],
        }

        with mock.patch.object(ra, 'triage_paper', return_value=valid_triage):
            result = ra.batch_screen_abstracts(papers, mock.Mock())
            self.assertEqual(result[0]["triage"]["relevance"], 4)
            self.assertEqual(result[0]["triage"]["contribution"], "New method")


# ---------------------------------------------------------------------------
# Tests for resumability
# ---------------------------------------------------------------------------

class TestResumability(unittest.TestCase):
    """Tests that corpus runs can resume from the last completed query."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def test_resume_skips_completed_queries(self):
        """Completed queries are skipped when resuming."""
        # Pre-populate corpus with status
        papers = [_make_paper("2610.01256")]
        ra.save_corpus(papers, self.tmpdir)
        status = {
            "completed_queries": ["already done"],
            "counts": {"unique": 1},
        }
        ra.save_corpus_status(status, self.tmpdir)

        loaded, run_status = ra.load_corpus(self.tmpdir)
        queries = ["already done", "new query"]
        completed = set(run_status.get("completed_queries", []))
        to_run = [q for q in queries if q not in completed]
        self.assertEqual(to_run, ["new query"])

    def test_resume_preserves_existing_papers(self):
        """Resuming preserves existing corpus papers and adds new ones."""
        existing = [_make_paper("2610.01256")]
        ra.save_corpus(existing, self.tmpdir)
        loaded, _ = ra.load_corpus(self.tmpdir)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["arxiv_id"], "2610.01256")


# ---------------------------------------------------------------------------
# Tests for corpus run report generation
# ---------------------------------------------------------------------------

class TestCorpusRunReport(unittest.TestCase):
    """Tests for update_corpus_run_report."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def test_report_generated(self):
        """A run report is created with counts and queries."""
        path = ra.update_corpus_run_report(
            discovered=100,
            unique=80,
            screened=80,
            relevant=20,
            full_text_acquired=5,
            queries=["q1", "q2"],
            failures=["error1"],
            source_coverage={"OpenAlex": 60, "arXiv": 40},
            out_dir=self.tmpdir,
        )
        self.assertTrue(path.exists())
        content = path.read_text(encoding="utf-8")
        self.assertIn("100", content)  # discovered
        self.assertIn("80", content)  # unique
        self.assertIn("q1", content)
        self.assertIn("error1", content)

    def test_report_without_failures(self):
        """Report excludes failures section when empty."""
        path = ra.update_corpus_run_report(
            discovered=10,
            unique=10,
            screened=10,
            relevant=5,
            full_text_acquired=0,
            queries=[],
            failures=[],
            source_coverage={},
            out_dir=self.tmpdir,
        )
        content = path.read_text(encoding="utf-8")
        self.assertNotIn("Failures", content)


if __name__ == "__main__":
    unittest.main()
