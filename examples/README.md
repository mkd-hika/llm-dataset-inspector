# Reproducible audit example

From the repository root:

```bash
python -m pip install pandas numpy requests
python examples/quickstart.py
```

This minimal path audits a local JSONL file. The full UI and remote dataset loaders use the dependencies in `requirements.txt`.

The [six-row synthetic fixture](sample.jsonl) intentionally includes a duplicate, an empty response, HTML markup, and a reserved `demo@example.com` address. It contains no real personal data.

## Observed output

The local run produced **85 / 100**, **12 PASS**, **2 FAIL**, and **1 WARN** checks, with zero critical findings. Token counts used **estimate(char/4)** because `tiktoken` was absent. Installing a tokenizer can change token-dependent checks; this is not a benchmark or certification.

![Actual fixture audit counts](../docs/images/example-audit.png)

Generated artifacts:

- [JSON report](report.json): machine-readable findings.
- [Data card](data_card.md): inspectable Markdown summary.
- [HTML report](report.html): download and open locally in a browser.

Outputs come from the current inspector pipeline. Regenerate them with `quickstart.py` after changes.
