"""Headless CLI for the LLM Dataset Inspector.

Examples:
  python cli.py https://huggingface.co/datasets/owner/name
  python cli.py data/train.jsonl --html report.html --json meta.json --card data_card.md
  python cli.py data/train.csv --fail-under 75      # non-zero exit if score < 75

Designed for CI gates: exits 1 when the score is below --fail-under or any
CRITICAL test fails.
"""

from __future__ import annotations

import argparse
import sys

from inspector import audit, report_to_html, report_to_json, data_card_md
from inspector.tests_suite import FAIL


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit an LLM training dataset.")
    ap.add_argument("link", help="Dataset link or local path")
    ap.add_argument("--limit", type=int, default=None, help="Row sample limit per split")
    ap.add_argument("--html", help="Write HTML report to this path")
    ap.add_argument("--json", help="Write JSON metadata to this path")
    ap.add_argument("--card", help="Write data_card.md to this path")
    ap.add_argument("--fail-under", type=float, default=None,
                    help="Exit 1 if overall score is below this value")
    args = ap.parse_args()

    try:
        report = audit(args.link, row_limit=args.limit)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    s = report.score
    print(f"\nLLM-Readiness: {s['overall']} / 100  ({s['overall_rating']})")
    print(f"FAIL={s['counts']['fail']}  WARN={s['counts']['warn']}  "
          f"PASS={s['counts']['pass']}  CRITICAL={s['counts']['critical']}\n")
    for t in sorted(report.tests, key=lambda t: t.status != FAIL):
        if t.status != "PASS":
            print(f"  [{t.severity:8}] {t.status:4} {t.category:11} {t.name} "
                  f"({t.affected_rows} rows)")

    if args.html:
        open(args.html, "w").write(report_to_html(report))
        print(f"\nWrote {args.html}")
    if args.json:
        open(args.json, "w").write(report_to_json(report))
        print(f"Wrote {args.json}")
    if args.card:
        open(args.card, "w").write(data_card_md(report))
        print(f"Wrote {args.card}")

    critical_fail = s["counts"]["critical"] > 0
    below = args.fail_under is not None and s["overall"] < args.fail_under
    if critical_fail or below:
        print("\nQA GATE: FAILED", file=sys.stderr)
        return 1
    print("\nQA GATE: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
