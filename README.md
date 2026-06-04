---
title: LLM Dataset QA Inspector
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.0"
app_file: gradio_app.py
pinned: false
license: mit
---

# LLM Dataset QA Inspector

An internal tool for auditing whether a dataset has been properly cleaned and
prepared for LLM training. Paste a dataset link (or local path); the tool
downloads it, runs a deterministic inspection pipeline, and produces a strict
QA scorecard with row-level evidence and remediation snippets.

## What it checks

- **Source and structure**: source type, format, splits, rows/columns, memory,
  schema, dtype consistency, missingness (including correlated-missingness
  signals), cardinality.
- **Cleaning**: exact duplicates, near-duplicates (whitespace/case normalised),
  HTML/XML bleed, encoding artifacts (mojibake), empty/whitespace rows, noise
  and boilerplate signals.
- **Formatting**: detects Alpaca, ShareGPT, ChatML, prompt-completion, and RLHF
  preference-pair layouts; validates chat turn ordering and empty turns.
- **LLM-readiness**: token distribution and percentiles (p5..p99), token-budget
  checks, lexical diversity (type-token ratio).
- **Safety**: PII regex scan (email, phone, SSN, IPv4, Luhn-checked card
  numbers) and a configurable toxicity keyword surface scan.
- **Integrity**: ID/row uniqueness and train/eval split-leakage detection.
- **Profiling**: numeric outliers (IQR and z-score), class imbalance with a
  recommended resampling strategy, numeric correlation.

Each check is reported as PASS / WARN / FAIL with the count of affected rows,
example row indices, a severity (CRITICAL/HIGH/MEDIUM/LOW/INFO), a remediation
snippet, and an effort estimate. The results roll up into a weighted
LLM-Readiness Score (Completeness 20, Quality 25, Format 20, Safety 20,
Diversity 15).

## Setup

    pip install -r requirements.txt

## Use the GUI

    streamlit run app.py

Paste a HuggingFace Hub URL, a GitHub raw URL, or a direct
CSV/TSV/JSON/JSONL/Parquet URL (optionally .gz), then click Run audit. Export
the HTML report, JSON metadata, or `data_card.md` from the Export section.

## Use the CLI (for CI gates)

    python cli.py data/train.jsonl --html report.html --json meta.json --card data_card.md
    python cli.py data/train.csv --fail-under 80

The CLI exits non-zero if the overall score is below `--fail-under` or any
CRITICAL test fails, so it can gate a training pipeline.

## Use as a library

    from inspector import audit, report_to_html
    report = audit("https://huggingface.co/datasets/owner/name", row_limit=5000)
    print(report.score["overall"], report.score["overall_rating"])
    open("report.html", "w").write(report_to_html(report))

## Honest limitations

- **Token counts** use `tiktoken` (cl100k_base) when available; on first use it
  fetches its vocabulary over the network. If that is unavailable it falls back
  to a `len(text)/4` estimate, which the report labels explicitly. For exact
  LLaMA counts, swap in the relevant tokenizer in `inspector/readiness.py`.
- **Toxicity** is a keyword *surface* flag only. It both over- and under-flags;
  route flagged rows to a trained classifier (Detoxify, Perspective) or human
  review. Slur lists are intentionally not hard-coded; supply your own policy
  list via `TOXICITY_KEYWORDS`.
- **PII** is regex-based and approximate. Use it to triage, not to certify.
- **Near-duplicate** detection is normalisation-based, not embedding-based; for
  true semantic dedup, add MinHash/embeddings.
- Private or gated sources require local credentials.

## Project layout

    inspector/
      sources.py       source detection and loading
      structural.py    schema, missingness, cardinality, duplicates, leakage
      readiness.py     tokens, text quality, format detection, PII, toxicity
      profiling.py     outliers, imbalance, vocabulary, correlation
      tests_suite.py   PASS/WARN/FAIL checks with fixes
      scoring.py       weighted scorecard
      reporting.py     HTML / JSON / data_card.md
      pipeline.py      orchestration -> AuditReport
    app.py             Streamlit GUI
    cli.py             headless CLI / CI gate
