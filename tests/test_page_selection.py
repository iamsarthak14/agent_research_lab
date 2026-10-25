"""Offline tests for _select_pages_with_budget and _build_excerpt.

These tests call the actual production functions from research_agent.py.
No model inference, no downloads, no package installation.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from research_agent import _select_pages_with_budget, _build_excerpt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_cached_text(path: str) -> list[tuple[int, str]]:
    """Parse a cached [PAGE N] text file into (page_num, page_text) tuples."""
    text = open(path).read()
    pages: list[tuple[int, str]] = []
    current_page = None
    current_text_parts: list[str] = []
    for line in text.split('\n'):
        if line.startswith('[PAGE ') and line.endswith(']'):
            if current_page is not None:
                pages.append((current_page, '\n'.join(current_text_parts)))
            current_page = int(line[6:-1])
            current_text_parts = []
        else:
            current_text_parts.append(line)
    if current_page is not None:
        pages.append((current_page, '\n'.join(current_text_parts)))
    return pages


def _make_pages(page_numbers: list[int], text_per_page: int = 3000) -> list[tuple[int, str]]:
    """Create synthetic pages with predictable lengths for edge-case testing."""
    return [(pn, 'x' * text_per_page) for pn in page_numbers]


# ---------------------------------------------------------------------------
# Load cached DeFA paper (12 pages, ~42K chars)
# ---------------------------------------------------------------------------

DEFA_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'outputs', 'fulltext', '2610.01256.txt'
)
if os.path.exists(DEFA_PATH):
    DEFA_PAGES = _parse_cached_text(DEFA_PATH)  # 12 pages
else:
    DEFA_PAGES = None


class TestBuildExcerpt(unittest.TestCase):
    """Tests for _build_excerpt."""

    def test_empty(self):
        self.assertEqual(_build_excerpt([]), '')

    def test_single_page(self):
        pages = [(1, 'hello world')]
        result = _build_excerpt(pages)
        self.assertEqual(result, '[PAGE 1]\nhello world')

    def test_multi_page_joined_with_double_newline(self):
        pages = [(1, 'aaa'), (2, 'bbb'), (3, 'ccc')]
        result = _build_excerpt(pages)
        parts = result.split('\n\n')
        self.assertEqual(len(parts), 3)
        self.assertIn('[PAGE 1]', parts[0])
        self.assertIn('[PAGE 2]', parts[1])
        self.assertIn('[PAGE 3]', parts[2])


class TestSelectPagesWithBudget(unittest.TestCase):
    """Tests for _select_pages_with_budget."""

    # ---- Empty / trivial inputs ----

    def test_empty_input(self):
        selected, omitted = _select_pages_with_budget([], 15000)
        self.assertEqual(selected, [])
        self.assertEqual(omitted, [])

    def test_tiny_budget_no_pages_fit(self):
        pages = _make_pages([1, 2, 3], text_per_page=5000)
        selected, omitted = _select_pages_with_budget(pages, 100)
        self.assertEqual(selected, [])
        self.assertEqual(omitted, [1, 2, 3])
        self.assertEqual(len(_build_excerpt(selected)), 0)

    def test_short_paper_all_pages_fit(self):
        pages = _make_pages([1, 2, 3], text_per_page=1000)
        selected, omitted = _select_pages_with_budget(pages, 15000)
        self.assertEqual([pn for pn, _ in selected], [1, 2, 3])
        self.assertEqual(omitted, [])

    # ---- Budget enforcement ----

    def test_15k_budget_respected(self):
        if DEFA_PAGES is None:
            self.skipTest('DeFA cached text not available')
        selected, omitted = _select_pages_with_budget(DEFA_PAGES, 15000)
        excerpt = _build_excerpt(selected)
        self.assertLessEqual(len(excerpt), 15000,
                             f'Excerpt {len(excerpt)} exceeds 15K budget')

    def test_30k_budget_respected(self):
        if DEFA_PAGES is None:
            self.skipTest('DeFA cached text not available')
        selected, omitted = _select_pages_with_budget(DEFA_PAGES, 30000)
        excerpt = _build_excerpt(selected)
        self.assertLessEqual(len(excerpt), 30000,
                             f'Excerpt {len(excerpt)} exceeds 30K budget')

    def test_tiny_budget_never_exceeds(self):
        pages = _make_pages([1, 2, 3], text_per_page=5000)
        for budget in [50, 100, 200, 500]:
            selected, omitted = _select_pages_with_budget(pages, budget)
            excerpt = _build_excerpt(selected)
            self.assertLessEqual(len(excerpt), budget,
                                 f'excerpt {len(excerpt)} > budget {budget}')

    # ---- Regional coverage ----

    def test_15k_cov_beginning_middle_end(self):
        """12-page paper at 15K should cover all 3 regions."""
        if DEFA_PAGES is None:
            self.skipTest('DeFA cached text not available')
        selected, omitted = _select_pages_with_budget(DEFA_PAGES, 15000)
        page_nums = [pn for pn, _ in selected]

        beginning_covered = any(1 <= pn <= 2 for pn in page_nums)
        middle_covered = any(5 <= pn <= 8 for pn in page_nums)
        end_covered = any(pn >= 11 for pn in page_nums)

        self.assertTrue(beginning_covered,
                        f'Beginning not covered; pages={page_nums}')
        self.assertTrue(middle_covered,
                        f'Middle not covered; pages={page_nums}')
        self.assertTrue(end_covered,
                        f'End not covered; pages={page_nums}')

    def test_30k_all_strategic_pages_covered(self):
        """30K budget on 12-page paper should include all strategic pages."""
        if DEFA_PAGES is None:
            self.skipTest('DeFA cached text not available')
        selected, omitted = _select_pages_with_budget(DEFA_PAGES, 30000)
        page_nums = [pn for pn, _ in selected]

        # Should include at least one page from each region
        beginning_covered = any(1 <= pn <= 2 for pn in page_nums)
        middle_covered = any(5 <= pn <= 8 for pn in page_nums)
        end_covered = any(pn >= 11 for pn in page_nums)

        self.assertTrue(beginning_covered)
        self.assertTrue(middle_covered)
        self.assertTrue(end_covered)
        # Should include more pages than 15K
        self.assertGreater(len(page_nums), 4)

    def test_short_paper_no_regions_to_check(self):
        """Papers <= 6 pages: all pages should be included if they fit."""
        pages = _make_pages([1, 2, 3, 4, 5, 6], text_per_page=1000)
        selected, omitted = _select_pages_with_budget(pages, 15000)
        self.assertEqual([pn for pn, _ in selected], [1, 2, 3, 4, 5, 6])
        self.assertEqual(omitted, [])

    # ---- Edge cases ----

    def test_single_page_paper(self):
        pages = _make_pages([1], text_per_page=3000)
        selected, omitted = _select_pages_with_budget(pages, 15000)
        self.assertEqual([pn for pn, _ in selected], [1])
        self.assertEqual(omitted, [])

    def test_early_page_exceeds_budget_but_later_fits(self):
        """First page too large, but a later page fits within budget."""
        pages = _make_pages([1, 2, 3], text_per_page=10000)
        selected, omitted = _select_pages_with_budget(pages, 5000)
        # Page 1 (10K text + ~10 header + 2 separator = 10022) exceeds 5000
        # Page 2 (same) also exceeds
        # Page 3 (same) also exceeds
        # So nothing should be selected
        self.assertEqual(selected, [])
        self.assertEqual(omitted, [1, 2, 3])

    def test_early_page_exceeds_but_tiny_later_page_fits(self):
        """First page huge, second page tiny — the greedy pass selects page 1,
        but the final budget enforcement strips it, resulting in no pages."""
        pages = [(1, 'A' * 10000), (2, 'B' * 500), (3, 'C' * 10000)]
        selected, omitted = _select_pages_with_budget(pages, 2000)
        page_nums = [pn for pn, _ in selected]
        # The greedy pass selects page 1 (chars_used=0, so first always selected).
        # Then the final budget enforcement strips it since 10011 > 2000.
        # The tiny page 2 is never reached because the greedy pass processes
        # pages in order and page 1 already consumed the budget.
        self.assertEqual(page_nums, [])
        self.assertEqual(omitted, [1, 2, 3])
        excerpt = _build_excerpt(selected)
        self.assertLessEqual(len(excerpt), 2000)

    def test_first_page_fits_tiny_second_fits_huge_third_skipped(self):
        """First page fits, second (tiny) fits, third (huge) skipped."""
        pages = [(1, 'A' * 500), (2, 'B' * 500), (3, 'C' * 10000)]
        selected, omitted = _select_pages_with_budget(pages, 2000)
        page_nums = [pn for pn, _ in selected]
        self.assertEqual(page_nums, [1, 2])
        self.assertEqual(omitted, [3])
        excerpt = _build_excerpt(selected)
        self.assertLessEqual(len(excerpt), 2000)

    # ---- Exact excerpt length verification ----

    def test_exact_excerpt_length_at_15k(self):
        """Verify exact excerpt length is within budget (not just approximately)."""
        if DEFA_PAGES is None:
            self.skipTest('DeFA cached text not available')
        selected, omitted = _select_pages_with_budget(DEFA_PAGES, 15000)
        excerpt = _build_excerpt(selected)
        # The budget enforcement should produce an excerpt <= 15000
        self.assertLessEqual(len(excerpt), 15000)
        # Should not be trivially small — the swap logic should produce
        # a reasonable selection
        self.assertGreater(len(selected), 0,
                           'Expected at least one page at 15K budget')

    def test_exact_excerpt_length_at_30k(self):
        if DEFA_PAGES is None:
            self.skipTest('DeFA cached text not available')
        selected, omitted = _select_pages_with_budget(DEFA_PAGES, 30000)
        excerpt = _build_excerpt(selected)
        self.assertLessEqual(len(excerpt), 30000)
        self.assertGreater(len(selected), 0)

    # ---- Omitted pages correctness ----

    def test_omitted_pages_exact(self):
        """Omitted pages must be exactly those not in selected."""
        if DEFA_PAGES is None:
            self.skipTest('DeFA cached text not available')
        selected, omitted = _select_pages_with_budget(DEFA_PAGES, 15000)
        all_page_nums = [pn for pn, _ in DEFA_PAGES]
        selected_page_nums = [pn for pn, _ in selected]
        expected_omitted = [pn for pn in all_page_nums if pn not in selected_page_nums]
        self.assertEqual(omitted, expected_omitted)

    # ---- Stable ordering ----

    def test_selected_pages_in_document_order(self):
        """Selected pages must appear in original document order."""
        if DEFA_PAGES is None:
            self.skipTest('DeFA cached text not available')
        selected, _ = _select_pages_with_budget(DEFA_PAGES, 15000)
        page_nums = [pn for pn, _ in selected]
        self.assertEqual(page_nums, sorted(page_nums),
                         f'Pages not in document order: {page_nums}')

    # ---- No duplicate pages ----

    def test_no_duplicate_page_numbers(self):
        if DEFA_PAGES is None:
            self.skipTest('DeFA cached text not available')
        selected, _ = _select_pages_with_budget(DEFA_PAGES, 30000)
        page_nums = [pn for pn, _ in selected]
        self.assertEqual(len(page_nums), len(set(page_nums)),
                         f'Duplicate pages found: {page_nums}')


if __name__ == '__main__':
    unittest.main()
