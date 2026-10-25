from __future__ import annotations

import argparse
import csv
import json
import os
import pymupdf
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1").rstrip("/")
API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
MODEL = os.getenv("LM_STUDIO_MODEL", "")
TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "25"))
MAILTO = os.getenv("OPENALEX_MAILTO", "").strip()
OUT = Path("outputs")
HEADERS = {"User-Agent": "AgentResearchLab/0.1 (local research workflow)"}


def get_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def normalize_title(title: str) -> str:
    return re.sub(r"\W+", " ", (title or "").lower()).strip()


def search_openalex(query: str, per_source: int) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "search": query,
        "per-page": min(per_source, 50),
        "sort": "relevance_score:desc",
        "select": "id,doi,title,publication_year,authorships,primary_location,biblio,cited_by_count,abstract_inverted_index,type",
    }
    if MAILTO:
        params["mailto"] = MAILTO
    data = get_json("https://api.openalex.org/works", params)
    results = []
    for item in data.get("results", []):
        inverted = item.get("abstract_inverted_index") or {}
        abstract = ""
        if inverted:
            words: dict[int, str] = {}
            for word, positions in inverted.items():
                for pos in positions:
                    words[int(pos)] = word
            abstract = " ".join(words[p] for p in sorted(words))
        loc = item.get("primary_location") or {}
        source = loc.get("source") or {}
        authors = [
            (a.get("author") or {}).get("display_name", "")
            for a in (item.get("authorships") or [])[:8]
        ]
        results.append({
            "source_db": "OpenAlex",
            "title": item.get("title") or "",
            "year": item.get("publication_year"),
            "authors": authors,
            "abstract": abstract,
            "doi": item.get("doi"),
            "url": loc.get("landing_page_url") or item.get("id") or "",
            "venue": source.get("display_name") or "",
            "citation_count": item.get("cited_by_count"),
            "openalex_id": item.get("id"),
            "arxiv_id": None,
        })
    return results



def search_arxiv(query: str, per_source: int) -> list[dict[str, Any]]:
    import xml.etree.ElementTree as ET

    # Search individual meaningful terms rather than requiring the entire
    # query to occur as one exact phrase.
    stopwords = {
        "a", "an", "and", "for", "in", "of", "on", "or", "the",
        "to", "with", "by", "from", "how", "do", "does",
    }
    terms = re.findall(r"[A-Za-z0-9-]+", query.lower())
    terms = list(dict.fromkeys(
        term for term in terms
        if term not in stopwords and len(term) > 1
    ))[:10]

    if not terms:
        return []

    search_query = " OR ".join(f"all:{term}" for term in terms)

    response = requests.get(
        "https://export.arxiv.org/api/query",
        params={
            "search_query": search_query,
            "start": 0,
            "max_results": min(per_source, 50),
            "sortBy": "relevance",
            "sortOrder": "descending",
        },
        headers=HEADERS,
        timeout=TIMEOUT,
    )
    response.raise_for_status()

    root = ET.fromstring(response.text)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    results = []

    for entry in root.findall("a:entry", ns):
        arxiv_url = (
            entry.findtext("a:id", default="", namespaces=ns).strip()
        )
        match = re.search(r"abs/([^v/?]+)", arxiv_url)
        arxiv_id = match.group(1) if match else None

        title = " ".join(
            entry.findtext("a:title", default="", namespaces=ns).split()
        )
        abstract = " ".join(
            entry.findtext("a:summary", default="", namespaces=ns).split()
        )
        published = entry.findtext(
            "a:published", default="", namespaces=ns
        )
        authors = [
            a.findtext("a:name", default="", namespaces=ns)
            for a in entry.findall("a:author", ns)
        ]

        results.append({
            "source_db": "arXiv",
            "title": title,
            "year": int(published[:4]) if published[:4].isdigit() else None,
            "authors": authors[:8],
            "abstract": abstract,
            "doi": None,
            "url": arxiv_url,
            "venue": "arXiv preprint",
            "citation_count": None,
            "openalex_id": None,
            "arxiv_id": arxiv_id,
        })

    return results


def dedupe(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    key_to_index: dict[str, int] = {}

    def keys(paper: dict[str, Any]) -> list[str]:
        result = []
        doi = (paper.get("doi") or "").lower().strip()
        doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
        if doi:
            result.append("doi:" + doi)

        arxiv_id = (paper.get("arxiv_id") or "").lower().strip()
        if arxiv_id:
            result.append("arxiv:" + arxiv_id)

        title = normalize_title(paper.get("title", ""))
        if title:
            result.append("title:" + title)

        return result

    for paper in papers:
        paper_keys = keys(paper)
        matches = list(dict.fromkeys(
            key_to_index[key] for key in paper_keys
            if key in key_to_index
        ))

        if not matches:
            index = len(unique)
            unique.append(paper.copy())
        else:
            index = matches[0]
            target = unique[index]

            # If identifiers connect multiple existing records, consolidate
            # those records into the same entry.
            for other_index in reversed(matches[1:]):
                other = unique[other_index]
                for field, value in other.items():
                    if field == "abstract":
                        if len(value or "") > len(target.get(field) or ""):
                            target[field] = value
                    elif field == "authors":
                        if len(value or []) > len(target.get(field) or []):
                            target[field] = value
                    elif not target.get(field) and value:
                        target[field] = value

                unique.pop(other_index)
                for key, mapped_index in list(key_to_index.items()):
                    if mapped_index == other_index:
                        key_to_index.pop(key)
                    elif mapped_index > other_index:
                        key_to_index[key] = mapped_index - 1

                if other_index < index:
                    index -= 1
                target = unique[index]

            # Merge useful fields from the duplicate into the retained record.
            for field, value in paper.items():
                if field == "abstract":
                    if len(value or "") > len(target.get(field) or ""):
                        target[field] = value
                elif field == "authors":
                    if len(value or []) > len(target.get(field) or []):
                        target[field] = value
                elif field == "source_db" and value:
                    sources = set((target.get(field) or "").split("; "))
                    sources.update(value.split("; "))
                    target[field] = "; ".join(sorted(sources))
                elif field == "venue":
                    if value and (
                        not target.get(field)
                        or target.get(field) == "arXiv preprint"
                    ):
                        target[field] = value
                elif not target.get(field) and value:
                    target[field] = value

        # Index the merged record by every available identifier.
        for key in keys(unique[index]):
            key_to_index[key] = index

    return unique


def get_arxiv_pdf_url(paper: dict[str, Any]) -> str | None:
    """Return a PDF URL only when an arXiv identifier is available."""
    arxiv_id = str(paper.get("arxiv_id") or "").strip()

    if not arxiv_id:
        url = str(paper.get("url") or "")
        match = re.search(r"arxiv\.org/(?:abs|pdf)/([^/?#]+)", url, re.I)
        if match:
            arxiv_id = re.sub(r"v\d+$", "", match.group(1))

    if not re.fullmatch(r"\d{4}\.\d{4,5}", arxiv_id):
        return None

    return f"https://arxiv.org/pdf/{arxiv_id}"


def extract_full_text(paper: dict[str, Any]) -> dict[str, Any]:
    """Download/cache an arXiv PDF and extract text with page markers."""
    pdf_url = get_arxiv_pdf_url(paper)
    if not pdf_url:
        return {"status": "unavailable", "reason": "No supported arXiv ID"}

    arxiv_id = pdf_url.rsplit("/", 1)[-1]
    pdf_dir = OUT / "pdfs"
    text_dir = OUT / "fulltext"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = pdf_dir / f"{arxiv_id}.pdf"
    text_path = text_dir / f"{arxiv_id}.txt"

    try:
        if text_path.exists() and text_path.stat().st_size > 1000:
            extracted = text_path.read_text(encoding="utf-8")
            page_count = len(re.findall(r"(?m)^\[PAGE \d+\]$", extracted))
            return {
                "status": "cached",
                "arxiv_id": arxiv_id,
                "pages": page_count,
                "characters": len(extracted),
                "text_path": str(text_path),
            }

        if not pdf_path.exists():
            response = requests.get(
                pdf_url, headers=HEADERS, timeout=60
            )
            response.raise_for_status()

            if not response.content.startswith(b"%PDF"):
                raise ValueError("The server did not return a valid PDF")

            pdf_path.write_bytes(response.content)

        page_texts = []
        with pymupdf.open(str(pdf_path)) as doc:
            page_count = len(doc)
            for page_number, page in enumerate(doc, start=1):
                page_texts.append(
                    f"[PAGE {page_number}]\n{page.get_text()}"
                )

        extracted = "\n\n".join(page_texts).strip()
        if len(extracted) < 1000:
            raise ValueError(
                "Extracted text is unexpectedly short; PDF may be scanned"
            )

        text_path.write_text(extracted, encoding="utf-8")

        return {
            "status": "downloaded",
            "arxiv_id": arxiv_id,
            "pages": page_count,
            "characters": len(extracted),
            "text_path": str(text_path),
        }

    except Exception as exc:
        return {
            "status": "failed",
            "arxiv_id": arxiv_id,
            "reason": f"{type(exc).__name__}: {exc}",
        }


FULL_TEXT_ANALYZE_CHARS = 15000  # Strict character budget per paper (includes headers + separators)


def _parse_pages(text: str) -> list[tuple[int, str]]:
    """Split page-marked text into (page_number, page_text) pairs."""
    pages: list[tuple[int, str]] = []
    current_page: int | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        m = re.match(r"^\[PAGE (\d+)\]$", line)
        if m and current_page is not None:
            pages.append((current_page, "\n".join(current_lines).strip()))
            current_page = int(m.group(1))
            current_lines = []
        elif current_page is None:
            m2 = re.match(r"^\[PAGE (\d+)\]$", line)
            if m2:
                current_page = int(m2.group(1))
                current_lines = []
            else:
                current_lines.append(line)
        else:
            current_lines.append(line)

    if current_page is not None:
        pages.append((current_page, "\n".join(current_lines).strip()))

    return pages


def _select_pages_with_budget(
    pages: list[tuple[int, str]], budget: int = FULL_TEXT_ANALYZE_CHARS
) -> tuple[list[tuple[int, str]], list[int]]:
    """Select pages to cover beginning, middle, and end within a strict character budget.

    Strategy: pick pages 1-2 (beginning), evenly-spaced middle pages, pages N-1 to N (end).
    Then enforce the budget on the selected pages only (including headers and separators).
    If the budget cannot cover all strategic pages, include as many as fit and report omitted.
    """
    if not pages:
        return [], []

    total_pages = len(pages)

    if total_pages <= 6:
        selected_indices = list(range(total_pages))
    else:
        first_chunk = 2  # pages 0, 1  (indices)
        last_chunk = 2   # last 2 pages
        middle_count = max(2, (total_pages - first_chunk - last_chunk) // 3)

        selected_indices: list[int] = []
        # First chunk (indices 0, 1)
        selected_indices.extend(range(first_chunk))
        # Middle chunk (evenly spaced)
        middle_start = first_chunk
        middle_end = total_pages - last_chunk
        if middle_end > middle_start:
            step = max(1, (middle_end - middle_start) // middle_count)
            selected_indices.extend(
                middle_start + i * step for i in range(middle_count)
                if middle_start + i * step < middle_end
            )
            # Ensure coverage of the middle range
            if middle_start + (middle_count - 1) * step < middle_end - 1:
                selected_indices.append(middle_start + (middle_count - 1) * step + 1)
        # Last chunk
        selected_indices.extend(range(total_pages - last_chunk, total_pages))

        # Deduplicate while preserving order
        seen: set[int] = set()
        deduped: list[int] = []
        for i in selected_indices:
            if i not in seen:
                seen.add(i)
                deduped.append(i)
        selected_indices = deduped

    # Enforce character budget on the strategically selected pages only.
    # Budget is strict: total excerpt characters (headers + page text + separators) <= budget.
    # Two-pass approach:
    #   Pass 1: greedily select pages in order, skipping overflows (no break).
    #   Pass 2: if beginning+middle+end coverage is not achieved, try swapping
    #           expensive pages for cheaper ones from the same region.
    selected: list[tuple[int, str]] = []
    chars_used = 0
    for idx in selected_indices:
        page_num, page_text = pages[idx]
        # Cost of adding this page: header line + separator + page text
        page_cost = len(f"[PAGE {page_num}]\n") + len(page_text) + 2  # +2 for "\n\n" separator
        if chars_used > 0 and chars_used + page_cost > budget:
            continue  # Skip this page; try later candidates
        selected.append((page_num, page_text))
        chars_used += page_cost

    # If the paper has >6 pages, verify that beginning, middle, and end are
    # all represented.  If not, try to swap expensive selected pages for cheaper
    # unselected ones from the same region.
    if total_pages > 6 and len(selected) > 0:
        included_page_nums = {pn for pn, _ in selected}
        # Check coverage
        has_beginning = 1 in included_page_nums and 2 in included_page_nums
        has_middle = any(5 <= pn <= 8 for pn in included_page_nums)
        has_end = 11 in included_page_nums or 12 in included_page_nums

        if not (has_beginning and has_middle and has_end):
            # Try to achieve full coverage by swapping expensive pages.
            # Build a candidate list: all strategically selected pages sorted by
            # cost (cheapest first), preserving original order for the final output.
            # Mark each page by region: 'beginning' (1-2), 'middle' (5-8), 'end' (11-12), 'other'
            def page_region(pn):
                if pn <= 2:
                    return 'beginning'
                elif 5 <= pn <= 8:
                    return 'middle'
                elif pn >= 11:
                    return 'end'
                return 'other'

            # Collect all candidate pages (selected + unselected from strategic set)
            all_candidates = [(idx, pages[idx]) for idx in selected_indices]
            # Sort by cost (cheapest first) for swapping
            all_candidates_sorted = sorted(all_candidates, key=lambda x: len(x[1][1]))

            # Try to swap: find the most expensive selected page from a region
            # that is already covered, and replace it with the cheapest unselected
            # page from a missing region.
            max_attempts = 5
            for attempt in range(max_attempts):
                current_selected_set = {pn for pn, _ in selected}
                current_regions_covered = set()
                for pn, _ in selected:
                    r = page_region(pn)
                    if r != 'other':
                        current_regions_covered.add(r)

                missing_regions = {'beginning', 'middle', 'end'} - current_regions_covered
                if not missing_regions:
                    break  # All regions covered

                # Find cheapest unselected page from a missing region
                best_swap_in = None
                for ci, (cidx, cpage) in enumerate(all_candidates_sorted):
                    cpn, cpt = cpage
                    if cpn not in current_selected_set and page_region(cpn) in missing_regions:
                        best_swap_in = (cidx, cpage, page_region(cpn))
                        break

                if best_swap_in is None:
                    break  # No page from missing region fits

                _, swap_in_page, swap_in_region = best_swap_in
                swap_in_cost = len(f'[PAGE {swap_in_page[0]}]\n') + len(swap_in_page[1]) + 2

                # Find most expensive selected page from a region that is also
                # covered by another selected page (so removing it won't lose coverage).
                best_swap_out = None
                best_swap_out_cost = 0
                for sn, st in selected:
                    st_region = page_region(sn)
                    if st_region == 'other':
                        continue
                    # Count how many selected pages cover this region
                    region_count = sum(1 for s, _ in selected if page_region(s) == st_region)
                    if region_count > 1:
                        st_cost = len(f'[PAGE {sn}]\n') + len(st) + 2
                        if st_cost > best_swap_out_cost:
                            best_swap_out = (sn, st, st_cost)

                if best_swap_out is None:
                    break  # Cannot swap without losing coverage

                out_sn, out_st, out_cost = best_swap_out
                delta = swap_in_cost - out_cost
                if delta <= 0:
                    # Swap is free or saves budget — do it
                    selected = [(sn, st) for sn, st in selected if sn != out_sn]
                    selected.append(swap_in_page)
                # else: swap would exceed budget; try next cheapest swap-in

    # Enforce final budget on the complete excerpt (headers + separators + text).
    # Strip pages from the end until the budget is respected.
    excerpt_candidate = _build_excerpt(selected)
    while selected and len(_build_excerpt(selected)) > budget:
        selected = selected[:-1]
        excerpt_candidate = _build_excerpt(selected)

    included_page_nums = {pn for pn, _ in selected}
    omitted = [pn for pn, _ in pages if pn not in included_page_nums]

    return selected, omitted


def _validate_fields(
    fields: dict[str, Any], pages_inspected: dict[int, str]
) -> dict[str, Any]:
    """Validate a single analysis field: check cited pages exist and evidence appears in source.

    On validation failure, preserves the original finding and evidence for auditability
    but sets status to 'uncertain'. Does not silently erase the original model output.
    """
    if not isinstance(fields, dict):
        return {"finding": None, "evidence": None, "pages": None, "status": "uncertain"}

    finding = fields.get("finding")
    evidence = fields.get("evidence")
    pages_cited = fields.get("pages")
    status = fields.get("status", "uncertain")

    # Validate page citations
    if pages_cited and isinstance(pages_cited, list):
        valid_pages = [
            pn for pn in pages_cited
            if isinstance(pn, int) and pn in pages_inspected
        ]
        if not valid_pages:
            # Cited pages don't exist in inspected text — mark uncertain but preserve
            status = "uncertain"
            pages_cited = None
        else:
            # Check evidence appears in source pages
            if evidence and isinstance(evidence, str):
                verified = any(
                    evidence in pages_inspected.get(pn, "") for pn in valid_pages
                )
                if not verified:
                    # Evidence snippet not found in cited pages — mark uncertain
                    # but keep finding/evidence for auditability
                    status = "uncertain"
            pages_cited = valid_pages
    elif pages_cited is None:
        # Model did not cite pages — mark as uncertain
        status = "uncertain"
        pages_cited = None

    return {
        "finding": finding,
        "evidence": evidence,
        "pages": pages_cited,
        "status": status,
    }


def _build_excerpt(pages: list[tuple[int, str]]) -> str:
    """Reconstruct page-marked excerpt from page tuples."""
    parts = []
    for page_num, page_text in pages:
        parts.append(f"[PAGE {page_num}]\n{page_text}")
    return "\n\n".join(parts)


def analyze_full_text(client: OpenAI, paper: dict[str, Any]) -> dict[str, Any]:
    """Analyze a paper's cached full text with page-referenced, bounded queries.

    Returns structured JSON analysis with evidence snippets and page citations.
    Exactly one model call per paper. No retries or follow-ups.
    """
    ft = paper.get("full_text") or {}
    text_path = ft.get("text_path")

    if not text_path or not Path(text_path).exists():
        return {
            "title": paper.get("title", "Unknown"),
            "arxiv_id": ft.get("arxiv_id") or paper.get("arxiv_id"),
            "_status": "cache_unavailable",
            "_reason": "No cached full text found",
        }

    try:
        raw_text = Path(text_path).read_text(encoding="utf-8")
    except Exception as exc:
        return {
            "title": paper.get("title", "Unknown"),
            "arxiv_id": ft.get("arxiv_id") or paper.get("arxiv_id"),
            "_status": "cache_read_error",
            "_reason": f"{type(exc).__name__}: {exc}",
        }

    if len(raw_text) < 1000:
        return {
            "title": paper.get("title", "Unknown"),
            "arxiv_id": ft.get("arxiv_id") or paper.get("arxiv_id"),
            "_status": "cache_too_short",
            "_reason": f"Extracted text is {len(raw_text)} chars (minimum 1000)",
        }

    # Parse pages
    pages = _parse_pages(raw_text)
    if not pages:
        return {
            "title": paper.get("title", "Unknown"),
            "arxiv_id": ft.get("arxiv_id") or paper.get("arxiv_id"),
            "_status": "no_page_markers",
            "_reason": "Text does not contain [PAGE N] markers",
        }

    total_pages = len(pages)

    # Select pages within character budget
    selected, omitted = _select_pages_with_budget(pages)
    if not selected:
        return {
            "title": paper.get("title", "Unknown"),
            "arxiv_id": ft.get("arxiv_id") or paper.get("arxiv_id"),
            "_status": "selection_failed",
            "_reason": "Could not select any pages within budget",
            "_pages_total": total_pages,
        }

    excerpt = _build_excerpt(selected)
    included_pages = [pn for pn, _ in selected]
    pages_inspected = dict(selected)  # page_num -> text for validation

    # Build prompt
    title = paper.get("title", "Unknown")
    prompt = f"""You are a careful research analyst. Analyze this paper's full text
and return structured findings. The text below has been excerpted from the full
paper using a bounded page-selection strategy.

IMPORTANT RULES:
- Attribute all claims to the authors. Use wording like "The authors report..."
  Do NOT present author-reported results as independently verified facts.
- For main_results, explicitly label them as "author-reported".
  This pipeline does NOT independently reproduce experiments.
- If evidence is not found in the inspected pages, set status to
  "not_found_in_inspected_text" and leave finding/evidence/pages as empty/None.
  Do NOT imply the entire paper lacks the information.
- Cite only page numbers that appear in the supplied excerpt.
  Validate that each evidence snippet actually appears in the corresponding page.
- Return valid JSON only. No reasoning or text outside the JSON.

Return exactly this JSON structure:
{{
  "research_question": {{
    "finding": "string or null",
    "evidence": "short verbatim excerpt or null",
    "pages": [integer page numbers] or null,
    "status": "supported" | "not_found_in_inspected_text" | "uncertain"
  }},
  "claimed_contribution": {{ ... same structure ... }},
  "method": {{ ... same structure ... }},
  "datasets": {{ ... same structure ... }},
  "baselines": {{ ... same structure ... }},
  "metrics": {{ ... same structure ... }},
  "main_results": {{
    "finding": "string (explicitly author-reported) or null",
    "evidence": "short verbatim excerpt or null",
    "pages": [integer page numbers] or null,
    "status": "supported" | "not_found_in_inspected_text" | "uncertain"
  }},
  "limitations": {{ ... same structure ... }},
  "computational_cost": {{ ... same structure ... }},
  "reproducibility": {{ ... same structure ... }}
}}

PAPER TITLE: {title}

EXCERPT (page-selected, {len(selected)}/{total_pages} pages, {len(excerpt)} chars):
{excerpt}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            max_tokens=1500,
            reasoning_effort="none",
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False}
            },
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a conservative research analyst. "
                        "Return one valid JSON object only. No reasoning or "
                        "text outside the JSON. Validate all cited page numbers "
                        "exist in the supplied text."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )

        content = response.choices[0].message.content or "{}"
        content = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.I
        )
        result = json.loads(content)

        # Validate: check cited pages exist and evidence appears in source
        validated_fields: dict[str, Any] = {}
        for field_name, field_data in result.items():
            if field_name.startswith("_"):
                validated_fields[field_name] = field_data
                continue
            validated_fields[field_name] = _validate_fields(
                field_data, pages_inspected
            )

        supported_count = sum(
            1 for v in validated_fields.values()
            if isinstance(v, dict) and v.get("status") == "supported"
        )

        return {
            "title": paper.get("title", "Unknown"),
            "arxiv_id": ft.get("arxiv_id") or paper.get("arxiv_id"),
            "_status": "complete" if supported_count >= 3 else "partial",
            "_pages_total": total_pages,
            "_pages_inspected": included_pages,
            "_pages_omitted": omitted,
            "_excerpt_chars": len(excerpt),
            "_analysis": validated_fields,
        }

    except RuntimeError as exc:
        # MODEL not configured
        return {
            "title": paper.get("title", "Unknown"),
            "arxiv_id": ft.get("arxiv_id") or paper.get("arxiv_id"),
            "_status": "model_not_configured",
            "_reason": str(exc),
        }
    except requests.exceptions.ConnectionError as exc:
        return {
            "title": paper.get("title", "Unknown"),
            "arxiv_id": ft.get("arxiv_id") or paper.get("arxiv_id"),
            "_status": "connection_error",
            "_reason": f"Could not connect to LM Studio: {exc}",
        }
    except requests.exceptions.Timeout as exc:
        return {
            "title": paper.get("title", "Unknown"),
            "arxiv_id": ft.get("arxiv_id") or paper.get("arxiv_id"),
            "_status": "timeout",
            "_reason": f"Model request timed out: {exc}",
        }
    except json.JSONDecodeError as exc:
        return {
            "title": paper.get("title", "Unknown"),
            "arxiv_id": ft.get("arxiv_id") or paper.get("arxiv_id"),
            "_status": "parse_error",
            "_reason": f"Model output was not valid JSON: {exc}",
            "_raw_excerpt": content[:2000] if 'content' in dir() else None,
        }
    except Exception as exc:
        return {
            "title": paper.get("title", "Unknown"),
            "arxiv_id": ft.get("arxiv_id") or paper.get("arxiv_id"),
            "_status": "error",
            "_reason": f"{type(exc).__name__}: {exc}",
        }


def local_client() -> OpenAI:
    return OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=120.0, max_retries=0)


def triage_paper(client: OpenAI, paper: dict[str, Any]) -> dict[str, Any]:
    if not MODEL or MODEL.startswith("replace-with"):
        raise RuntimeError("Set LM_STUDIO_MODEL in .env to the model ID returned by /v1/models.")
    
    prompt = f"""
You are a careful research librarian for agent systems. Assess this paper
only from the supplied metadata and abstract. You have NOT read the full
paper. Do not invent details, results, limitations, or novelty.

Return valid JSON only with these keys:
relevance (integer 1-5),
themes (array of short strings),
contribution (one sentence describing what the authors claim),
method (one sentence, or 'not clear from abstract'),
evidence_limits (one sentence),
follow_up_questions (array of up to 3 strings).

Evidence rules:
- Attribute claims to the authors: use wording such as "The abstract reports..."
  rather than presenting reported results as independently verified facts.
- If the abstract does not mention a baseline, statistical test, dataset detail,
  cost, or limitation, say "Not stated in the abstract; verify in full text."
  Do NOT conclude that the full paper lacks it.
- Distinguish a limitation explicitly acknowledged by the authors from a
  concern you infer. Label inferred concerns as "Potential concern to check".
- Do not invent benchmark names, numerical results, comparisons, or methods.
- If the abstract is too vague to support a claim, say so.
- Follow-up questions must be answerable by inspecting the paper or its
  experimental artifacts. Prioritize baselines, evaluation design,
  reproducibility, generalization, and computational cost when relevant.

A relevance of 5 means directly useful for research on agent reliability,
evaluation, inference-time efficiency, adaptive harnesses, test-time learning,
or theoretical principles of agents. Be selective; topical similarity alone
does not establish research quality.

PAPER:
Title: {paper.get('title')}
Year: {paper.get('year')}
Authors: {', '.join(paper.get('authors') or [])}
Venue: {paper.get('venue')}
Abstract: {(paper.get('abstract') or '')[:9000]}
"""

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=1200,
        reasoning_effort="none",
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        },
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a conservative research assistant. "
                    "Return one valid JSON object only. No reasoning or "
                    "text outside the JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content or "{}"
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.I)
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        result = {
            "relevance": 1,
            "themes": ["parse_error"],
            "contribution": "Could not parse local-model output; inspect raw response.",
            "method": "not clear from abstract",
            "evidence_limits": "Triage output was not valid JSON.",
            "follow_up_questions": [],
            "raw_model_output": content[:2000],
        }
    return result


def write_outputs(papers: list[dict[str, Any]], queries: list[str]) -> None:
    OUT.mkdir(exist_ok=True)
    with (OUT / "papers.jsonl").open("w", encoding="utf-8") as f:
        for p in papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    columns = [
        "title", "year", "authors", "venue", "source_db", "relevance", "themes",
        "contribution", "method", "evidence_limits", "citation_count", "doi", "arxiv_id", "url"
    ]
    with (OUT / "paper_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for p in sorted(papers, key=lambda x: (x.get("relevance") or 0, x.get("year") or 0), reverse=True):
            row = dict(p)
            row["authors"] = "; ".join(row.get("authors") or [])
            row["themes"] = "; ".join((row.get("triage") or {}).get("themes") or [])
            row["relevance"] = (row.get("triage") or {}).get("relevance", "")
            for key in ("contribution", "method", "evidence_limits"):
                row[key] = (row.get("triage") or {}).get(key, "")
            writer.writerow(row)

    ranked = sorted(papers, key=lambda x: (x.get("triage", {}).get("relevance", 0), x.get("year") or 0), reverse=True)
    lines = [
        "# Agent-systems research map",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "> **Evidence warning:** This report uses public metadata/abstracts and local-model triage. It is not a full-paper review, novelty proof, or peer-reviewed conclusion. Verify primary sources before relying on any claim.",
        "",
        "## Search queries",
        *[f"- {q}" for q in queries],
        "",
        "## Highest-ranked papers",
    ]
    for p in ranked[:20]:
        t = p.get("triage") or {}
        lines += [
            "",
            f"### {p.get('title','Untitled')} ({p.get('year') or 'year unknown'})",
            f"- **Relevance:** {t.get('relevance', 'unrated')}/5",
            f"- **Source:** {p.get('url') or 'URL unavailable'}",
            f"- **Authors:** {', '.join(p.get('authors') or []) or 'not listed'}",
            f"- **Venue/database:** {p.get('venue') or p.get('source_db') or 'unknown'}",
            f"- **Contribution (abstract-level):** {t.get('contribution', 'not extracted')}",
            f"- **Method (abstract-level):** {t.get('method', 'not clear from abstract')}",
            f"- **Limits:** {t.get('evidence_limits', 'not extracted')}",
            f"- **Themes:** {', '.join(t.get('themes') or [])}",
            f"- **Questions to check in the full paper:** {'; '.join(t.get('follow_up_questions') or []) or 'none extracted'}",
        ]
    lines += [
        "",
        "## Next manual verification steps",
        "1. Open the highest-relevance primary papers and verify their actual methods, baselines, and limitations.",
        "2. Check related work and papers that cite each candidate; search for contrary findings.",
        "3. Mark which results are author-reported versus independently reproduced.",
        "4. Do not claim a novelty gap until near-neighbor work has been checked.",
        "5. Record exclusions and reasons, not just selected papers.",
        "",
    ]
    (OUT / "research_map.md").write_text("\n".join(lines), encoding="utf-8")



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bounded local-first agent-systems literature triage."
    )
    parser.add_argument(
        "--query",
        action="append",
        required=True,
        help="Repeat to search multiple queries.",
    )
    parser.add_argument(
        "--per-source",
        type=int,
        default=10,
        help="Max results per query per database (1-50).",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Collect metadata without local-model triage.",
    )
    parser.add_argument(
        "--full-text",
        action="store_true",
        help="Download and extract full text for arXiv papers.",
    )
    parser.add_argument(
        "--full-text-analyze",
        action="store_true",
        help="Analyze cached full text with page-referenced, bounded queries.",
    )

    args = parser.parse_args()
    per_source = max(1, min(args.per_source, 50))

    all_papers: list[dict[str, Any]] = []

    # 1. Search both sources.
    for query in args.query:
        print(f"[search] {query}")

        for fn in (search_openalex, search_arxiv):
            try:
                results = fn(query, per_source)
                print(f"  {fn.__name__}: {len(results)} results")

                for paper in results:
                    paper["matched_queries"] = [query]

                all_papers.extend(results)

            except Exception as exc:
                print(
                    f"  WARNING: {fn.__name__} failed: "
                    f"{type(exc).__name__}: {exc}"
                )

            time.sleep(0.5)

    # 2. Deduplicate before processing.
    papers = dedupe(all_papers)
    print(f"[dedupe] {len(papers)} unique papers")

    if not papers:
        raise SystemExit(
            "No papers collected. Check internet access/API availability "
            "and try fewer queries."
        )

    # 3. Optionally download and extract full text.
    if args.full_text:
        manifest_path = OUT / "fulltext_manifest.jsonl"
        OUT.mkdir(parents=True, exist_ok=True)

        with manifest_path.open("w", encoding="utf-8") as manifest:
            for i, paper in enumerate(papers, 1):
                result = extract_full_text(paper)
                paper["full_text"] = result

                print(
                    f"[full-text {i}/{len(papers)}] "
                    f"{result['status']} — "
                    f"{paper.get('title', '')[:70]}"
                )

                manifest.write(
                    json.dumps(
                        {
                            "title": paper.get("title"),
                            "url": paper.get("url"),
                            "full_text": result,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        print(f"[full-text] Manifest: {manifest_path}")

    # 4. Run the existing abstract-level triage unless skipped.
    client = None if args.skip_model else local_client()

    for i, paper in enumerate(papers, 1):
        if client:
            try:
                paper["triage"] = triage_paper(client, paper)

                print(
                    f"[triage {i}/{len(papers)}] "
                    f"{paper.get('triage', {}).get('relevance', '?')}/5 — "
                    f"{paper['title'][:80]}"
                )

            except Exception as exc:
                paper["triage"] = {
                    "relevance": 0,
                    "themes": [],
                    "contribution": "Triage failed.",
                    "method": "unknown",
                    "evidence_limits": str(exc),
                    "follow_up_questions": [],
                }

                print(
                    f"[triage {i}/{len(papers)}] FAILED: "
                    f"{type(exc).__name__}: {exc}"
                )
        else:
            paper["triage"] = {
                "relevance": 0,
                "themes": [],
                "contribution": "",
                "method": "",
                "evidence_limits": "Skipped",
                "follow_up_questions": [],
            }

    # 5. Optionally analyze full text.
    if args.full_text_analyze:
        ft_client = None if args.skip_model else local_client()
        analysis_path = OUT / "fulltext_analysis.jsonl"
        OUT.mkdir(parents=True, exist_ok=True)

        analyzed_count = 0
        with analysis_path.open("w", encoding="utf-8") as analysis_file:
            for i, paper in enumerate(papers, 1):
                ft = paper.get("full_text")
                if not ft or ft.get("status") not in ("cached", "downloaded"):
                    print(
                        f"[full-text-analyze {i}/{len(papers)}] "
                        f"skipped (no cache) — {paper.get('title', '')[:60]}"
                    )
                    continue

                if ft_client:
                    try:
                        analysis = analyze_full_text(ft_client, paper)
                    except Exception as exc:
                        analysis = {
                            "title": paper.get("title", "Unknown"),
                            "arxiv_id": ft.get("arxiv_id"),
                            "_status": "error",
                            "_reason": f"{type(exc).__name__}: {exc}",
                        }
                else:
                    analysis = {
                        "title": paper.get("title", "Unknown"),
                        "arxiv_id": ft.get("arxiv_id"),
                        "_status": "skipped",
                        "_reason": "Model triage was skipped",
                    }

                paper["full_text_analysis"] = analysis
                print(
                    f"[full-text-analyze {i}/{len(papers)}] "
                    f"{analysis.get('_status', '?')} — "
                    f"{paper.get('title', '')[:60]}"
                )

                analysis_file.write(
                    json.dumps(analysis, ensure_ascii=False) + "\n"
                )
                analyzed_count += 1

        print(f"[full-text-analyze] Manifest: {analysis_path}")

    # 6. Write the existing research outputs.
    write_outputs(papers, args.query)

    print("\nDone. Review outputs/research_map.md and outputs/paper_matrix.csv.")

    if args.full_text:
        print(
            "Full-text extraction was attempted for each paper. "
            "Check outputs/fulltext_manifest.jsonl for individual statuses."
        )

    if args.full_text_analyze:
        print(
            "Full-text analysis was attempted for each paper with cached text. "
            "Check outputs/fulltext_analysis.jsonl for individual results."
        )

    print(
        "Remember: abstract-level ranking is a filter, "
        "not a substitute for reading primary papers."
    )


if __name__ == "__main__":
    main()

