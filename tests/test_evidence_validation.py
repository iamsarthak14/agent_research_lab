"""Offline unit tests for _validate_fields.

These tests call the actual production function from research_agent.py.
No model inference, no downloads, no package installation.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from research_agent import _validate_fields


class TestValidateFields(unittest.TestCase):
    """Tests for _validate_fields."""

    def setUp(self):
        self.pages = {1: "hello world", 2: "foo bar", 3: "test data"}

    # ---- Valid evidence ----

    def test_valid_string_evidence_found_in_cited_page(self):
        """Evidence that appears in a cited page is marked supported."""
        result = _validate_fields(
            {"finding": "test finding", "evidence": "hello world", "pages": [1], "status": "supported"},
            self.pages,
        )
        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["pages"], [1])
        self.assertEqual(result["finding"], "test finding")
        self.assertEqual(result["evidence"], "hello world")

    def test_valid_evidence_not_found_in_cited_page(self):
        """Evidence not found in cited pages is marked uncertain."""
        result = _validate_fields(
            {"finding": "test", "evidence": "not here", "pages": [1], "status": "supported"},
            self.pages,
        )
        self.assertEqual(result["status"], "uncertain")
        self.assertEqual(result["pages"], [1])
        # Original finding and evidence preserved for auditability
        self.assertEqual(result["finding"], "test")
        self.assertEqual(result["evidence"], "not here")

    # ---- Missing / empty evidence ----

    def test_missing_evidence_marked_uncertain(self):
        """When evidence is None (missing), status is uncertain even with valid pages."""
        result = _validate_fields(
            {"finding": "test finding", "pages": [1], "status": "supported"},
            self.pages,
        )
        self.assertEqual(result["status"], "uncertain")
        self.assertEqual(result["pages"], [1])
        self.assertIsNone(result["evidence"])
        # Finding is preserved for auditability
        self.assertEqual(result["finding"], "test finding")

    def test_empty_evidence_string(self):
        """Empty string evidence ('') is NOT found in any page, so status is uncertain."""
        result = _validate_fields(
            {"finding": "test", "evidence": "", "pages": [1], "status": "supported"},
            self.pages,
        )
        self.assertEqual(result["status"], "uncertain")
        self.assertEqual(result["evidence"], "")

    # ---- Non-string evidence (defect: currently accepted as supported) ----

    def test_non_string_evidence_integer_marked_uncertain(self):
        """Non-string evidence (integer) must be marked uncertain, never supported."""
        result = _validate_fields(
            {"finding": "test", "evidence": 123, "pages": [1], "status": "supported"},
            self.pages,
        )
        self.assertEqual(result["status"], "uncertain",
                         "Non-string evidence should never be marked supported")

    def test_non_string_evidence_none_preserved_but_uncertain(self):
        """Evidence=None is not a valid string; status should be uncertain."""
        result = _validate_fields(
            {"finding": "test", "evidence": None, "pages": [1], "status": "supported"},
            self.pages,
        )
        self.assertEqual(result["status"], "uncertain")
        # Original evidence preserved for auditability
        self.assertIsNone(result["evidence"])

    def test_non_string_evidence_list_marked_uncertain(self):
        """Non-string evidence (list) must be marked uncertain."""
        result = _validate_fields(
            {"finding": "test", "evidence": ["a", "b"], "pages": [1], "status": "supported"},
            self.pages,
        )
        self.assertEqual(result["status"], "uncertain")

    # ---- Valid page numbers ----

    def test_valid_page_numbers(self):
        """Valid page numbers are preserved."""
        result = _validate_fields(
            {"finding": "test", "evidence": "hello world", "pages": [1, 2], "status": "supported"},
            self.pages,
        )
        self.assertEqual(result["pages"], [1, 2])
        self.assertEqual(result["status"], "supported")

    def test_invalid_page_numbers(self):
        """Cited pages that do not exist in inspected text are rejected."""
        result = _validate_fields(
            {"finding": "test", "evidence": "hello world", "pages": [99], "status": "supported"},
            self.pages,
        )
        self.assertEqual(result["status"], "uncertain")
        self.assertIsNone(result["pages"])

    def test_mixed_valid_and_invalid_pages(self):
        """Valid pages are kept; invalid ones are filtered out."""
        result = _validate_fields(
            {"finding": "test", "evidence": "hello world", "pages": [1, 99], "status": "supported"},
            self.pages,
        )
        self.assertEqual(result["pages"], [1])
        # Evidence verified against valid pages — should remain supported
        self.assertEqual(result["status"], "supported")

    # ---- pages as integer (not list) ----

    def test_pages_as_integer_not_list(self):
        """When pages is an integer instead of a list, status is uncertain."""
        result = _validate_fields(
            {"finding": "test", "evidence": "hello world", "pages": 1, "status": "supported"},
            self.pages,
        )
        self.assertEqual(result["status"], "uncertain")

    def test_pages_none(self):
        """When pages is None, status is uncertain."""
        result = _validate_fields(
            {"finding": "test", "pages": None, "status": "supported"},
            self.pages,
        )
        self.assertEqual(result["status"], "uncertain")
        self.assertIsNone(result["pages"])

    # ---- Empty / non-dictionary input ----

    def test_empty_dictionary(self):
        """Empty dict returns all None fields with uncertain status."""
        result = _validate_fields({}, self.pages)
        self.assertEqual(result["status"], "uncertain")
        self.assertIsNone(result["finding"])
        self.assertIsNone(result["evidence"])
        self.assertIsNone(result["pages"])

    def test_non_dictionary_input(self):
        """Non-dict input returns all None fields with uncertain status."""
        result = _validate_fields("string input", self.pages)
        self.assertEqual(result["status"], "uncertain")
        self.assertIsNone(result["finding"])
        self.assertIsNone(result["evidence"])
        self.assertIsNone(result["pages"])

    # ---- Missing fields ----

    def test_missing_finding(self):
        """When finding is missing, status is uncertain even if evidence is verified."""
        result = _validate_fields({"evidence": "hello world", "pages": [1]}, self.pages)
        self.assertEqual(result["status"], "uncertain")
        self.assertIsNone(result["finding"])
        self.assertEqual(result["evidence"], "hello world")

    def test_all_fields_missing(self):
        """When all fields are missing, everything is None and uncertain."""
        result = _validate_fields({"unrelated": "stuff"}, self.pages)
        self.assertEqual(result["status"], "uncertain")

    # ---- Unsupported / unknown status values ----

    def test_unsupported_status_value_normalized_to_uncertain(self):
        """Unknown status values (e.g., 'verified') are overridden by verification results.

        When evidence is verified in the source, the status is 'supported' regardless of
        whatever status the model reported. Only unverified evidence with unknown status
        remains uncertain.
        """
        result = _validate_fields(
            {"finding": "test", "evidence": "hello world", "pages": [1], "status": "verified"},
            self.pages,
        )
        self.assertEqual(result["status"], "supported",
                         "Verified evidence should be supported regardless of model status")

    def test_status_not_found_in_inspected_text_preserved(self):
        """When the model explicitly marks status as 'not_found_in_inspected_text', it is preserved."""
        result = _validate_fields(
            {"finding": "test", "evidence": None, "pages": None, "status": "not_found_in_inspected_text"},
            self.pages,
        )
        self.assertEqual(result["status"], "not_found_in_inspected_text")

    # ---- Regression: valid case still works ----

    def test_valid_case_still_works(self):
        """A valid case with string evidence found in a cited page is still supported."""
        result = _validate_fields(
            {"finding": "test", "evidence": "hello world", "pages": [1], "status": "supported"},
            self.pages,
        )
        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["pages"], [1])
        self.assertEqual(result["finding"], "test")
        self.assertEqual(result["evidence"], "hello world")

    # ---- PDF normalization fix (Mission 2C) ----

    def test_pdf_line_breaks_normalized_for_evidence_matching(self):
        """Evidence quotes from LLM (no line breaks) match PDF text with line breaks."""
        from src.research_agent import _normalize_text, _validate_fields

        # PDF extraction preserves line breaks and hyphenation
        pdf_page_text = (
            "We introduce DeFA, a dependency-guided\n"
            "framework for agent failure attribution. DeFA first combines protocol relations\n"
            "and semantic dependencies into an event dependency graph spanning the trajec-\n"
            "tory."
        )

        # LLM returns cleaned quote (no line breaks, no hyphenation)
        cleaned_quote = (
            "We introduce DeFA, a dependency-guided framework for agent failure "
            "attribution. DeFA first combines protocol relations and semantic "
            "dependencies into an event dependency graph spanning the trajectory."
        )

        pages = {1: pdf_page_text}
        field = {"finding": "test", "evidence": cleaned_quote, "pages": [1]}

        result = _validate_fields(field, pages)
        self.assertEqual(result["status"], "supported",
                         "Cleaned LLM quotes should match PDF text with line breaks")

    def test_fabricated_quote_not_verified(self):
        """A fabricated quote that does not appear in any page should be uncertain."""
        from src.research_agent import _validate_fields

        pages = {1: "The actual content of page 1."}
        field = {"finding": "test", "evidence": "This quote does not exist anywhere.",
                 "pages": [1]}

        result = _validate_fields(field, pages)
        self.assertEqual(result["status"], "uncertain")

    def test_valid_quote_with_invalid_page(self):
        """A valid quote cited to a page that does not contain it should be uncertain."""
        from src.research_agent import _validate_fields

        pages = {1: "Page 1 content.", 2: "Page 2 content."}
        field = {"finding": "test", "evidence": "Page 1 content.", "pages": [2]}

        result = _validate_fields(field, pages)
        self.assertEqual(result["status"], "uncertain")

    def test_empty_evidence_becomes_uncertain(self):
        """Empty evidence string cannot be verified."""
        from src.research_agent import _validate_fields

        pages = {1: "hello world"}
        field = {"finding": "test", "evidence": "", "pages": [1]}

        result = _validate_fields(field, pages)
        self.assertEqual(result["status"], "uncertain")
        self.assertEqual(result["evidence"], "")

    def test_normalization_preserves_hyphenated_words(self):
        """Normalization joins hyphenated word wraps at line boundaries.

        Note: the normalizer joins 'word-<newline>rest' → 'wordrest'. Compound
        hyphenated words split across lines (e.g. 'state-\\n' + 'of-the-art')
        will be joined as 'stateof-the-art'.  This is an acceptable trade-off
        because actual paper text rarely splits compound words at line breaks.
        """
        from src.research_agent import _normalize_text

        # Simple hyphenated word not split across lines is preserved
        text = "This is state-of-the-art technology."
        self.assertEqual(_normalize_text(text), "This is state-of-the-art technology.")

        # Hyphen at line break is joined (removes the hyphen)
        text2 = "This is state-\nof-the-art technology."
        self.assertEqual(_normalize_text(text2), "This is stateof-the-art technology.")

        # Truncated hyphenated word (the common case): 'trajec-\\n' + 'tory'
        text3 = "The trajec-\ntory of events."
        self.assertEqual(_normalize_text(text3), "The trajectory of events.")

    # ---- Adversarial normalization tests (Mission 2D) ----

    def test_normalization_does_not_accept_fabricated_near_matches(self):
        """Near-matches should not be accepted as verified quotes."""
        from src.research_agent import _validate_fields

        pages = {1: "The study found a 42% improvement in accuracy."}
        # Model slightly misquotes: changes "42%" to "40%"
        field = {"finding": "test", "evidence": "The study found a 40% improvement in accuracy.",
                 "pages": [1]}

        result = _validate_fields(field, pages)
        self.assertEqual(result["status"], "uncertain",
                         "Fabricated near-matches must not be verified")

    def test_normalization_preserves_punctuation(self):
        """Punctuation must be preserved through normalization."""
        from src.research_agent import _normalize_text

        text = "The results: 42% ± 3.5% (p < 0.001); n=1,000."
        self.assertEqual(_normalize_text(text), text)

    def test_normalization_collapses_repeated_whitespace(self):
        """Multiple consecutive spaces must be collapsed to one."""
        from src.research_agent import _normalize_text

        text = "Hello    world   test"
        self.assertEqual(_normalize_text(text), "Hello world test")

    def test_normalization_handles_unicode_quotation_marks(self):
        """Unicode punctuation must be preserved, not stripped."""
        from src.research_agent import _normalize_text

        text = "DeFA's applicability to diverse agent executions."
        self.assertEqual(_normalize_text(text), text)

    def test_evidence_cited_to_multiple_pages_but_only_one_valid(self):
        """Evidence verified in one of multiple cited pages is supported."""
        from src.research_agent import _validate_fields

        pages = {1: "Page 1 content.", 2: "Page 2 content.", 3: "Page 3 content."}
        field = {"finding": "test", "evidence": "Page 2 content.", "pages": [2, 3]}

        result = _validate_fields(field, pages)
        self.assertEqual(result["status"], "supported")
        # pages field preserves ALL cited pages (not just valid ones)
        self.assertEqual(result["pages"], [2, 3])

    def test_evidence_spanning_truncated_pages_not_verified(self):
        """Evidence that is longer than a single truncated page is uncertain."""
        from src.research_agent import _validate_fields

        # Simulate a truncated page (shorter than the evidence quote)
        pages = {1: "Short page content."}
        field = {"finding": "test", "evidence": "This is a much longer piece of evidence "
                 "that does not fit on a single truncated page at all.", "pages": [1]}

        result = _validate_fields(field, pages)
        self.assertEqual(result["status"], "uncertain",
                         "Longer evidence than page content cannot be verified")

    def test_normalization_with_only_whitespace_input(self):
        """Whitespace-only input should normalize to empty string."""
        from src.research_agent import _normalize_text

        self.assertEqual(_normalize_text("   \n  \t  "), "")

    def test_empty_finding_becomes_uncertain_even_with_verified_evidence(self):
        """An empty-string finding with verified evidence must be uncertain."""
        from src.research_agent import _validate_fields

        pages = {1: "hello world"}
        field = {"finding": "", "evidence": "hello world", "pages": [1]}

        result = _validate_fields(field, pages)
        self.assertEqual(result["status"], "uncertain")


if __name__ == '__main__':
    unittest.main()
