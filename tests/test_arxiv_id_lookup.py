"""Offline tests for load_paper_by_arxiv_id and exact-ID CLI behaviour.

These tests call the actual production functions from research_agent.py.
No model inference, no downloads, no package installation.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import research_agent as ra_original


def _write_papers_jsonl(path, records):
    """Write a papers.jsonl file for testing."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')


def _make_paper(arxiv_id=None, **kwargs):
    """Create a minimal paper record."""
    return {
        "source_db": "arXiv",
        "title": "Test Paper",
        "year": 2026,
        "authors": ["Test Author"],
        "abstract": "Test abstract text.",
        "doi": None,
        "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None,
        "venue": "arXiv preprint",
        "citation_count": None,
        "openalex_id": None,
        "arxiv_id": arxiv_id,
        "full_text": {
            "status": "cached",
            "arxiv_id": arxiv_id,
            "pages": 12,
            "characters": 42000,
            "text_path": f"outputs/fulltext/{arxiv_id}.txt" if arxiv_id else None,
        },
        "triage": {
            "relevance": 0,
            "themes": [],
            "contribution": "",
            "method": "",
            "evidence_limits": "Skipped",
            "follow_up_questions": [],
        },
    }


# ---------------------------------------------------------------------------
# Tests for load_paper_by_arxiv_id
# ---------------------------------------------------------------------------

class TestLoadPaperByArxivId(unittest.TestCase):
    """Tests for exact arXiv ID matching from papers.jsonl."""

    def setUp(self):
        """Create a temporary papers.jsonl and patch OUT for this test."""
        self.tmpdir = tempfile.mkdtemp()
        self.original_out = ra_original.OUT
        self.patched_out = Path(self.tmpdir)
        ra_original.OUT = self.patched_out

    def tearDown(self):
        """Restore original OUT."""
        ra_original.OUT = self.original_out

    def _write_records(self, records):
        _write_papers_jsonl(str(self.patched_out / 'papers.jsonl'), records)

    # --- Positive cases ---

    def test_exact_id_match(self):
        """A record with exact matching arxiv_id is returned."""
        self._write_records([_make_paper("2610.01256")])
        result = ra_original.load_paper_by_arxiv_id("2610.01256")
        self.assertIsNotNone(result)
        self.assertEqual(result["arxiv_id"], "2610.01256")

    def test_exact_id_match_case_insensitive(self):
        """Matching is case-insensitive (normalised to lowercase)."""
        self._write_records([_make_paper("2610.01256")])
        result = ra_original.load_paper_by_arxiv_id("2610.01256".upper())
        self.assertIsNotNone(result)

    def test_exact_id_match_with_trailing_whitespace(self):
        """Trailing whitespace in the query ID is stripped before matching."""
        self._write_records([_make_paper("2610.01256")])
        result = ra_original.load_paper_by_arxiv_id("2610.01256  ")
        self.assertIsNotNone(result)

    # --- Negative cases ---

    def test_missing_arxiv_id_in_record(self):
        """Records without arxiv_id are skipped."""
        self._write_records([_make_paper(None)])
        result = ra_original.load_paper_by_arxiv_id("2610.01256")
        self.assertIsNone(result)

    def test_no_matching_record(self):
        """No match when the ID does not exist."""
        self._write_records([_make_paper("0812.4858")])
        result = ra_original.load_paper_by_arxiv_id("2610.01256")
        self.assertIsNone(result)

    def test_substring_only_match_rejected(self):
        """A query ID that is a substring of another ID must not match."""
        self._write_records([_make_paper("2610.18780")])
        result = ra_original.load_paper_by_arxiv_id("2610.01256")
        self.assertIsNone(result)

    def test_no_papers_jsonl(self):
        """Returns None when papers.jsonl does not exist."""
        result = ra_original.load_paper_by_arxiv_id("2610.01256")
        self.assertIsNone(result)

    def test_empty_file(self):
        """An empty papers.jsonl returns None."""
        self._write_records([])
        result = ra_original.load_paper_by_arxiv_id("2610.01256")
        self.assertIsNone(result)

    def test_duplicate_records_returns_none(self):
        """Multiple matching records are ambiguous — returns None."""
        records = [_make_paper("2610.01256"), _make_paper("2610.01256")]
        self._write_records(records)
        result = ra_original.load_paper_by_arxiv_id("2610.01256")
        self.assertIsNone(result)

    def test_malformed_json_skipped(self):
        """Malformed JSON lines are silently skipped."""
        self._write_records([])  # Ensure file exists
        with open(str(self.patched_out / 'papers.jsonl'), 'a', encoding='utf-8') as fh:
            fh.write("this is not json\n")
            fh.write(json.dumps(_make_paper("2610.01256")) + '\n')
        result = ra_original.load_paper_by_arxiv_id("2610.01256")
        self.assertIsNotNone(result)

    def test_record_without_full_text(self):
        """A record without full_text is still returned (full_text is optional)."""
        paper = _make_paper("2610.01256")
        del paper["full_text"]
        self._write_records([paper])
        result = ra_original.load_paper_by_arxiv_id("2610.01256")
        self.assertIsNotNone(result)
        self.assertNotIn("full_text", result)


# ---------------------------------------------------------------------------
# Tests for CLI integration (mocked)
# ---------------------------------------------------------------------------

class TestArxivIdCommandLine(unittest.TestCase):
    """Tests that --arxiv-id mode skips network search."""

    def setUp(self):
        """Patch OUT to a temp directory for the CLI test."""
        self.tmpdir = tempfile.mkdtemp()
        self.original_out = ra_original.OUT
        self.patched_out = Path(self.tmpdir)
        ra_original.OUT = self.patched_out
        # Write a valid paper record
        _write_papers_jsonl(str(self.patched_out / 'papers.jsonl'), [_make_paper("2610.01256")])

    def tearDown(self):
        """Restore original OUT."""
        ra_original.OUT = self.original_out

    def test_arxiv_id_skips_search_functions(self):
        """When --arxiv-id is provided, search_openalex and search_arxiv
        must NOT be called."""
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        original_argv = sys.argv
        sys.argv = [
            'research_agent',
            '--arxiv-id', '2610.01256',
            '--skip-model',
        ]

        # Mock search functions to raise errors (proving they're not called)
        # Mock dedupe and write_outputs to return benign results
        with mock.patch.object(ra_original, 'search_openalex', side_effect=RuntimeError("Should not be called")) as mock_ox, \
             mock.patch.object(ra_original, 'search_arxiv', side_effect=RuntimeError("Should not be called")) as mock_ax, \
             mock.patch.object(ra_original, 'dedupe', return_value=[]) as mock_dd, \
             mock.patch.object(ra_original, 'write_outputs') as mock_wo, \
             redirect_stdout(f):
            try:
                ra_original.main()
            except SystemExit:
                pass  # Expected — no valid output directory

            # Verify search functions were NOT called
            self.assertFalse(mock_ox.called, "search_openalex should not be called in --arxiv-id mode")
            self.assertFalse(mock_ax.called, "search_arxiv should not be called in --arxiv-id mode")

        sys.argv = original_argv

    def test_invalid_arxiv_id_format_rejected(self):
        """Invalid arXiv ID format is rejected with SystemExit."""
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        original_argv = sys.argv
        sys.argv = [
            'research_agent',
            '--arxiv-id', 'invalid-format',
            '--skip-model',
        ]

        with redirect_stdout(f):
            with self.assertRaises(SystemExit) as ctx:
                ra_original.main()
            self.assertIn("Invalid arXiv ID", str(ctx.exception))

        sys.argv = original_argv

    def test_missing_arxiv_id_in_cache_rejected(self):
        """Non-existent arXiv ID in cache is rejected."""
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        original_argv = sys.argv
        sys.argv = [
            'research_agent',
            '--arxiv-id', '9999.99999',
            '--skip-model',
        ]

        # Write a paper with a different ID
        _write_papers_jsonl(str(self.patched_out / 'papers.jsonl'), [_make_paper("0812.4858")])

        with redirect_stdout(f):
            with self.assertRaises(SystemExit) as ctx:
                ra_original.main()
            self.assertIn("No cached record found", str(ctx.exception))

        sys.argv = original_argv


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
