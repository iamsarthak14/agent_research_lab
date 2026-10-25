"""Offline unit tests for _parse_pages.

These tests call the actual production function from research_agent.py.
No model inference, no downloads, no package installation.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from research_agent import _parse_pages


class TestParsePages(unittest.TestCase):
    """Tests for _parse_pages."""

    def test_empty_input(self):
        """Empty string returns empty list."""
        self.assertEqual(_parse_pages(""), [])

    def test_no_page_markers(self):
        """Text with no page markers returns empty list."""
        text = "just some text\nno markers at all"
        self.assertEqual(_parse_pages(text), [])

    def test_single_valid_page(self):
        """Single page with content returns one tuple."""
        result = _parse_pages("[PAGE 1]\nhello world")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], (1, "hello world"))

    def test_multiple_valid_pages(self):
        """Multiple pages in order return all pages."""
        text = "[PAGE 1]\naaa\n[PAGE 2]\nbbb\n[PAGE 3]\nccc"
        result = _parse_pages(text)
        self.assertEqual(len(result), 3)
        self.assertEqual(result, [(1, "aaa"), (2, "bbb"), (3, "ccc")])

    def test_empty_page_content(self):
        """Consecutive markers with no content between produce empty page strings."""
        text = "[PAGE 1]\n[PAGE 2]\ncontent"
        result = _parse_pages(text)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], (1, ""))  # Empty page content
        self.assertEqual(result[1], (2, "content"))

    def test_consecutive_page_markers_no_whitespace(self):
        """Two consecutive markers on the same line (no newline between) do not match.

        The regex ^\\[PAGE (\\d+)\\]$ requires the entire line to be exactly
        one marker. '[PAGE 1][PAGE 2]' on one line matches neither marker, so
        the entire line is treated as content before the first page marker and
        is discarded.
        """
        text = "[PAGE 1][PAGE 2]\ncontent"
        result = _parse_pages(text)
        # Neither marker matches (line is '[PAGE 1][PAGE 2]', not '[PAGE 1]').
        # So no pages are found; the content line is also before any valid marker.
        self.assertEqual(result, [])

    def test_consecutive_page_markers_with_newline(self):
        """Consecutive markers on separate lines produce pages (first is empty)."""
        text = "[PAGE 1]\n[PAGE 2]\ncontent"
        result = _parse_pages(text)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], (1, ""))
        self.assertEqual(result[1], (2, "content"))

    def test_malformed_page_markers(self):
        """Non-digit page markers are ignored; only valid [PAGE \\d+] matches."""
        text = "[PAGE ABC]\ncontent\n[PAGE 1]\nreal"
        result = _parse_pages(text)
        # Malformed marker is skipped; only valid page markers create pages.
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], (1, "real"))

    def test_page_numbers_out_of_order(self):
        """Page numbers are extracted as-is; order is preserved in output."""
        text = "[PAGE 3]\naaa\n[PAGE 1]\nbbb\n[PAGE 2]\nccc"
        result = _parse_pages(text)
        self.assertEqual(len(result), 3)
        # Pages are in document order (order of appearance), not sorted.
        self.assertEqual(result, [(3, "aaa"), (1, "bbb"), (2, "ccc")])

    def test_content_before_first_page_marker(self):
        """Content before the first [PAGE N] marker is silently discarded."""
        text = "Title page\n[PAGE 1]\nactual content\n[PAGE 2]\nmore"
        result = _parse_pages(text)
        self.assertEqual(len(result), 2)
        # "Title page" is discarded; pages start at the first valid marker.
        self.assertEqual(result[0], (1, "actual content"))
        self.assertEqual(result[1], (2, "more"))

    def test_only_header_no_content(self):
        """A page marker with no trailing content produces an empty page string."""
        text = "[PAGE 1]\n"
        result = _parse_pages(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], (1, ""))


if __name__ == '__main__':
    unittest.main()
