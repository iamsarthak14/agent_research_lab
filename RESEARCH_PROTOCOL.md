# Research protocol (starter)

## Research identity
Working scope: agent systems, especially reliability/failure analysis, inference-time efficiency, test-time adaptation, and explanatory principles.

## Evidence hierarchy
1. Primary paper and supplementary material
2. Official repository, benchmark documentation, and artifacts
3. Independent reproduction or replication
4. Credible secondary analysis
5. Social posts / summaries (discovery only)

## Paper matrix fields
- Bibliographic identity, date/version, venue/status
- Research question and claim
- Method and assumptions
- Baselines and controls
- Tasks/datasets and leakage risks
- Metrics and resource budget
- Main results as reported by authors
- Limitations / negative findings
- Code, data, reproducibility status
- Closest prior work and contradictory evidence
- What must be verified in full text

## Novelty gate
Do not call a question novel until:
- Search synonyms and adjacent terminology.
- Inspect related work in the closest papers.
- Follow both backward and forward citations.
- Search for code, benchmarks, workshop papers, and preprints.
- Write the strongest argument that the contribution is not novel.
- State the exact difference from the closest three works.

## Experiment gate
- Pre-register hypothesis, conditions, primary metric, exclusions, and stopping rule.
- Use held-out tasks where possible.
- Keep model, prompt, tools, task set, and resource budgets controlled.
- Repeat stochastic trials where feasible.
- Log model identifier, quantization, context size, sampling parameters, timestamps, token usage, latency, tool errors, and exact code revision.
- Never fabricate missing measurements. Mark unavailable values explicitly.
