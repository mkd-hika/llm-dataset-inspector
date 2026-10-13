"""Generate a real audit of deliberately imperfect synthetic data."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from inspector import audit, data_card_md, report_to_html, report_to_json
from inspector.readiness import TOKENIZER_NAME

directory = ROOT / "examples"
report = audit("examples/sample.jsonl")
for name, content in {
    "report.html": report_to_html(report),
    "report.json": report_to_json(report),
    "data_card.md": data_card_md(report),
}.items():
    (directory / name).write_text(content, encoding="utf-8")
print(f"Tokenizer: {TOKENIZER_NAME}")
print(f"Readiness: {report.score['overall']} / 100")
print(f"Counts: {report.score['counts']}")
print(
    "This synthetic fixture contains intentional defects; its score is not a benchmark."
)
