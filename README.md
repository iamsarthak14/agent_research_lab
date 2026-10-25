# Agent Research Lab — local-first research assistant

A small, bounded starter pipeline for discovering AI/agent-systems papers, ranking them with your local Qwen model in LM Studio, and producing a source-linked research brief.

**Design goal:** automate literature-search breadth while keeping claims traceable. This is a starter system, not an autonomous scientist and not a guarantee of novelty.

## Architecture

1. **Discovery:** OpenAlex + arXiv search APIs (no paid LLM API).
2. **Deduplication:** DOI/arXiv ID/title.
3. **Local triage:** OpenAI-compatible LM Studio endpoint ranks relevance and extracts structured notes.
4. **Artifacts:** JSONL records, a paper matrix CSV, and a Markdown report.
5. **Human checkpoint:** inspect primary papers and citations before accepting any gap claim.

## Requirements

- macOS, Python 3.11+
- LM Studio with your Qwen 3.6 35B A3B model loaded
- LM Studio local server running (usually `http://localhost:1235/v1`)
- Internet access for public scholarly metadata and paper pages

No paid model API is used. Public APIs/websites may rate-limit requests. Local inference is private to your machine, but search queries and HTTP requests go to external services.

## Setup

```bash
cd agent_research_lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` if your LM Studio port/model differs. Start the LM Studio server before running the pipeline.

Check your local endpoint:

```bash
curl http://localhost:1235/v1/models
```

The `model` setting in `.env` should match an ID returned by that command. If the server accepts any model name, use the model's served ID.

## First run

```bash
python -m src.research_agent \
  --query "LLM agent reliability failure analysis evaluation" \
  --query "inference-time compute allocation for tool-using agents" \
  --query "test-time adaptation agent harness learning" \
  --per-source 15
```

Start with a small run. The script limits result counts and uses one local-model triage call per paper. If local inference is slow, lower `--per-source` to 5.

Outputs are written to `outputs/`:
- `papers.jsonl`: raw metadata + model triage
- `paper_matrix.csv`: sortable comparison table
- `research_map.md`: source-linked synthesis with explicit uncertainty

## What this first version does / does not do

**Does:** search public metadata, collect abstracts/links, deduplicate, use local Qwen for triage, save a repeatable evidence trail.

**Does not yet:** download and parse every PDF, follow full citation graphs, run experiments, or prove a research gap. Those should be added only after this small pipeline works reliably.

## Guardrails

- It never edits your source code or runs shell commands through the model.
- It does not automatically execute generated code.
- Search count is bounded; HTTP calls have timeouts.
- Treat all model-generated summaries as hypotheses until checked against source papers.
- Do not feed confidential employer material into public search or public repositories.
