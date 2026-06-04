# LLM Dataset QA Inspector — Documentation

Complete reference for setup, running, understanding, and extending the inspector.

---

## 1. What this project is

An internal QA tool for ML/data teams. Paste a dataset link or local path; the
tool streams the data, runs a deterministic inspection pipeline, and produces a
strict QA scorecard with row-level evidence and copy-paste remediation snippets.

The intended workflow: a team lead audits whether the data team has properly
cleaned and formatted a dataset before it goes into LLM training or fine-tuning.

The tool ships in three forms that all share one engine:

| Interface | File | How to run |
|---|---|---|
| Streamlit GUI | `app.py` | `streamlit run app.py` |
| Gradio GUI | `gradio_app.py` | `python gradio_app.py` |
| CLI / CI gate | `cli.py` | `python cli.py <path-or-url>` |
| Python library | `inspector/` | `from inspector import audit` |

---

## 2. Project structure

    LLM-Dataset-Inspection/
    ├── app.py                  Streamlit GUI
    ├── gradio_app.py           Gradio GUI (shareable / HF Spaces)
    ├── cli.py                  Headless CLI and CI gate
    ├── requirements.txt        Python dependencies
    ├── README.md               HuggingFace Spaces metadata + short intro
    ├── DOCUMENTATION.md        This file
    └── inspector/              Inspection engine (Python package)
        ├── __init__.py         Public API: audit, report_to_html, etc.
        ├── sources.py          Source detection + streaming dataset loader
        ├── structural.py       Schema, missingness, duplicates, leakage
        ├── readiness.py        Tokens, text quality, PII, toxicity, format
        ├── profiling.py        Outliers, class imbalance, vocabulary, correlation
        ├── tests_suite.py      PASS/WARN/FAIL checks with fix snippets
        ├── scoring.py          Weighted readiness scorecard
        ├── reporting.py        HTML / JSON / data_card.md generators
        └── pipeline.py         Orchestration → AuditReport dataclass

---

## 3. Prerequisites

- Python 3.10 or newer (developed on 3.12)
- Internet access for the first run (tiktoken fetches its vocabulary once)

---

## 4. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---|---|
| `pandas >= 2.0` | Core dataframe engine |
| `numpy >= 1.24` | Numeric operations |
| `requests >= 2.28` | HTTP streaming for remote files |
| `streamlit >= 1.30` | Streamlit GUI |
| `gradio >= 4.0` | Gradio GUI and HF Spaces deployment |
| `tiktoken >= 0.5` | Accurate token counts (falls back to char/4 if absent) |
| `datasets >= 2.14` | HuggingFace Hub streaming |
| `pyarrow >= 12.0` | Parquet file support |
| `ftfy >= 6.0` | Encoding-artifact fix snippets |

---

## 5. Running the tool

### 5.1 Streamlit GUI (local)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Paste a dataset link in the sidebar, set a
row limit, and click **Run audit**. Results appear across five sections:
Dataset identity, Scorecard, Critical issues, Test suite, Column analysis.
Export as HTML report, JSON metadata, or `data_card.md`.

### 5.2 Gradio GUI (local or shared)

```bash
python gradio_app.py            # local — http://localhost:7860
python gradio_app.py --share    # public URL valid for 72 hours
```

Tabs: Summary · Test Suite · Column Analysis · Export.
Progress bar updates at each pipeline stage.

### 5.3 Gradio GUI on HuggingFace Spaces (permanent URL)

Deployed at:
```
https://mkd-hika-llm-dataset-inspector.hf.space
```

To update after code changes:
```bash
git add .
git commit -m "your message"
git push hf main
```

HF Spaces rebuilds automatically on every push.

### 5.4 CLI (headless / CI gate)

```bash
python cli.py path/or/url
python cli.py data/train.jsonl --html report.html --json meta.json --card data_card.md
python cli.py data/train.csv --limit 5000 --fail-under 80
```

`--fail-under N` exits with code 1 if the overall score is below N **or** any
CRITICAL check fails. Use this to gate a training job in CI.

### 5.5 Python library

```python
from inspector import audit, report_to_html

report = audit("https://huggingface.co/datasets/owner/name", row_limit=5000)
print(report.score["overall"], report.score["overall_rating"])

for t in report.tests:
    if t.status != "PASS":
        print(t.severity, t.category, t.name, t.affected_rows)

open("report.html", "w").write(report_to_html(report))
```

---

## 6. Supported sources

| Source | How it is detected |
|---|---|
| HuggingFace Hub | `huggingface.co/datasets/` in the URL |
| GitHub raw | `raw.githubusercontent.com` or `/blob/` GitHub URL |
| Direct file URL | Ends in `.csv` `.tsv` `.json` `.jsonl` `.parquet` (optionally `.gz`) |
| Cloud storage | `s3://`, `gs://`, `amazonaws.com`, `storage.googleapis.com` |
| Local path | File exists on disk |

HuggingFace datasets are loaded with `streaming=True` — no full shard download.
JSONL files over HTTP are streamed line-by-line and stop at `row_limit`.

---

## 7. Detected instruction formats

The format detector checks column names and value shapes:

| Label | Required columns / structure |
|---|---|
| `alpaca` | `instruction` + `output` |
| `prompt_completion` | `prompt` + `completion` |
| `prompt_response` | `prompt` + `response` |
| `prompt_answer` | `prompt` + `answer` |
| `qa` | `question` + `answer` or `answers` |
| `query_response` | `query` + `response` |
| `query_answer` | `query` + `answer` |
| `input_output` | `input` + `output` |
| `input_target` | `input` + `target` |
| `source_target` | `source` + `target` |
| `rlhf_preference_pairs` | `chosen` + `rejected` |
| `sharegpt` | `conversations` column with `{from, value}` objects |
| `chatml` | `messages` column with `{role, content}` objects |
| `plain_text` | Single `text`, `content`, `document`, `passage`, or `body` column |
| `generic_tabular` | Fallback — no known format matched |

---

## 8. The checks

Results are grouped into seven categories. Each check returns PASS, WARN, or
FAIL with the count of affected rows, example row indices, a severity
(CRITICAL / HIGH / MEDIUM / LOW / INFO), a remediation snippet, and an effort
estimate.

| Category | What is checked |
|---|---|
| **SCHEMA** | Required columns present; dtype consistency (mixed Python types) |
| **INTEGRITY** | Exact duplicate rows; near-duplicate text; train/eval split leakage |
| **QUALITY** | Empty/whitespace text; HTML/XML bleed; encoding artifacts (mojibake); token-budget violations |
| **FORMAT** | Recognised instruction format; chat turn ordering; empty conversation turns |
| **SAFETY** | PII (email, phone, SSN, IPv4, Luhn-checked card numbers); toxicity keyword surface flags |
| **CONSISTENCY** | Empty responses; instruction/response length-ratio anomalies |
| **COVERAGE** | Lexical diversity (type-token ratio); class balance |

The engine also computes token distribution percentiles (p5–p99), numeric
outliers (IQR and z-score), numeric correlation, and missingness co-occurrence
signals.

---

## 9. The scorecard

Test results roll up into five weighted dimensions, each scored 0–100:

| Dimension | Weight | Fed by categories |
|---|---|---|
| Completeness | 20% | SCHEMA, INTEGRITY |
| Quality | 25% | QUALITY, CONSISTENCY |
| Format | 20% | FORMAT |
| Safety | 20% | SAFETY |
| Diversity | 15% | COVERAGE |

Each dimension starts at 100 and loses points per failing/warning check,
weighted by severity:

| Severity | FAIL penalty | WARN penalty |
|---|---|---|
| CRITICAL | 60 | 30 |
| HIGH | 30 | 15 |
| MEDIUM | 15 | 7.5 |
| LOW | 6 | 3 |

Overall score → rating: **Excellent** (≥ 90) · **Good** (≥ 75) · **Needs Work** (≥ 50) · **Critical** (< 50)

Tune weights and penalties in `inspector/scoring.py` to match your team's standards.

---

## 10. Module reference

| Module | Responsibility |
|---|---|
| `sources.py` | `detect_source(link)` classifies input without downloading. `load_dataset(info, row_limit)` streams into `{split: DataFrame}`. |
| `structural.py` | Schema, missingness (including correlated-missingness), cardinality, dtype consistency, exact/near duplicates, split leakage. |
| `readiness.py` | Token counting (tiktoken with char/4 fallback), text quality, noise/boilerplate, PII scan, toxicity surface scan, format detection, chat turn validation. |
| `profiling.py` | Numeric outliers (IQR + z-score), class imbalance with resampling strategy, vocabulary richness (TTR), numeric correlation. |
| `tests_suite.py` | Converts analysis into flat list of PASS/WARN/FAIL `TestResult` objects. Check thresholds and fix snippets live here. |
| `scoring.py` | Weighted scorecard (`DIMENSIONS`, `SEVERITY_PENALTY`). |
| `reporting.py` | `report_to_html`, `report_to_json`, `data_card_md`. |
| `pipeline.py` | `audit(link, row_limit)` — ties everything together, returns `AuditReport`. |
| `__init__.py` | Public exports: `audit`, `AuditReport`, `report_to_html`, `report_to_json`, `data_card_md`. |

---

## 11. How to extend

**Add a new check.** Open `inspector/tests_suite.py`, compute your signal from
the `analysis` dict or DataFrame, and append a `TestResult` with a category,
status, severity, affected-row count, example rows, and a fix snippet.

**Add a new instruction format.** Open `inspector/readiness.py`, add a
column-name pattern (or struct validator) to `detect_format()`.

**Change scoring.** Edit `DIMENSIONS` (weights and contributing categories) and
`SEVERITY_PENALTY` in `inspector/scoring.py`.

**Swap the tokenizer.** Replace the `tiktoken` encoder in `inspector/readiness.py`
with a LLaMA/SentencePiece tokenizer for exact counts for your target model.

**Customize toxicity / PII.** Edit `TOXICITY_KEYWORDS` and `PII_PATTERNS` in
`inspector/readiness.py`. The toxicity scan is keyword-only; route flagged rows
to a trained classifier (Detoxify, Perspective) for production use.

**Add a source type.** Extend `detect_source` / `load_dataset` in
`inspector/sources.py`.

---

## 12. Deployment

### HuggingFace Spaces (live)

```
https://mkd-hika-llm-dataset-inspector.hf.space
```

Push updates:
```bash
git add .
git commit -m "description of change"
git push hf main
```

### GitHub (source)

```
https://github.com/mkd-hika/llm-dataset-inspector
```

Push updates:
```bash
git push origin main
```

### Both at once

```bash
git push origin main && git push hf main
```

---

## 13. Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'inspector'` | Run from inside `LLM-Dataset-Inspection/` — the `inspector/` package must be in the current directory |
| `datasets` library required | `pip install datasets` — only needed for HuggingFace Hub URLs |
| Token counts show `estimate(char/4)` | `tiktoken` not installed or no network on first run. `pip install tiktoken` and allow one networked run |
| `.parquet` fails | `pip install pyarrow` |
| Gated / private dataset error | `huggingface-cli login` or download the file and pass its local path |
| `streamlit: command not found` | Activate the virtual environment first |
| HF Spaces Runtime error | Check the Logs tab on the Space page for the Python traceback |

---

## 14. Honest limitations

- **Token counts** are exact only with a matching tokenizer. The report labels
  the source so reviewers know which was used.
- **Toxicity scan** is a keyword surface flag — it over- and under-flags. Treat
  it as triage, not a verdict, and back it with a real classifier.
- **PII detection** is regex-based and approximate. Use it to triage candidates,
  not to certify a dataset as clean.
- **Near-duplicate detection** is normalisation-based, not embedding-based. For
  true semantic deduplication, add MinHash or embedding similarity.
- **Scorecard** is a heuristic aid for review, not an objective measure of
  training quality.
