"""Report generation: a self-contained HTML review report, a JSON metadata
file, and a HuggingFace-style data_card.md. No emoji is used anywhere in the
generated output."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from typing import List

from .pipeline import AuditReport
from .tests_suite import TestResult, FAIL, WARN

SEV_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
SEV_COLOR = {"CRITICAL": "#b00020", "HIGH": "#d35400", "MEDIUM": "#b8860b",
             "LOW": "#3a6ea5", "INFO": "#777"}
STATUS_COLOR = {"PASS": "#1b7a3d", "WARN": "#b8860b", "FAIL": "#b00020"}


def report_to_json(report: AuditReport) -> str:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "link": report.link,
        "source": {
            "type": report.source.source_type,
            "format": report.source.file_format,
            "resolved_url": report.source.resolved_url,
        },
        "splits": report.splits,
        "primary_split": report.primary_split,
        "scorecard": report.score,
        "format_detected": report.analysis["format"]["detected_formats"],
        "tests": [t.as_dict() for t in report.tests],
        "leakage": report.leakage,
    }
    return json.dumps(payload, indent=2, default=str)


def data_card_md(report: AuditReport) -> str:
    s = report.score
    fmt = ", ".join(report.analysis["format"]["detected_formats"])
    lines = [
        "---",
        "tags:",
        "- llm-training",
        "- qa-audited",
        "---",
        "",
        f"# Dataset Card",
        "",
        f"- Source type: {report.source.source_type}",
        f"- Detected format: {fmt}",
        f"- Splits: " + ", ".join(f"{k} ({v['rows']} rows)" for k, v in report.splits.items()),
        f"- LLM-Readiness Score: {s['overall']} / 100 ({s['overall_rating']})",
        "",
        "## Readiness dimensions",
        "",
        "| Dimension | Score | Rating |",
        "|---|---|---|",
    ]
    for dim, d in s["dimensions"].items():
        lines.append(f"| {dim} | {d['score']} | {d['rating']} |")
    lines += ["", "## Outstanding issues", ""]
    issues = [t for t in report.tests if t.status in (FAIL, WARN)]
    if not issues:
        lines.append("No failing or warning checks.")
    for t in sorted(issues, key=lambda t: SEV_ORDER[t.severity]):
        lines.append(f"- [{t.severity}] {t.category} / {t.name}: {t.detail} "
                     f"({t.affected_rows} rows)")
    return "\n".join(lines)


def report_to_html(report: AuditReport) -> str:
    s = report.score
    e = html.escape
    gen = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    crit = [t for t in report.tests if t.severity in ("CRITICAL", "HIGH") and t.status in (FAIL, WARN)]
    crit_html = ""
    if crit:
        items = "".join(
            f"<li><b style='color:{SEV_COLOR[t.severity]}'>[{t.severity}]</b> "
            f"{e(t.category)} / {e(t.name)} - {e(t.detail)} "
            f"<span class='muted'>({t.affected_rows} rows)</span></li>"
            for t in sorted(crit, key=lambda t: SEV_ORDER[t.severity]))
        crit_html = f"<div class='card alert'><h2>Critical and high-severity issues</h2><ul>{items}</ul></div>"

    dim_rows = "".join(
        f"<tr><td>{e(dim)}</td><td>{d['score']}</td><td>{int(d['weight']*100)}%</td>"
        f"<td><span class='rating r-{d['rating'].split()[0].lower()}'>{e(d['rating'])}</span></td></tr>"
        for dim, d in s["dimensions"].items())

    def test_row(t: TestResult):
        ex = ", ".join(str(x) for x in t.example_rows[:5])
        ex_html = f"<div class='muted'>example rows: {e(ex)}</div>" if ex else ""
        fix_html = f"<pre>{e(t.fix)}</pre>" if t.fix else ""
        return (f"<tr class='s-{t.status.lower()}'>"
                f"<td>{e(t.category)}</td><td>{e(t.name)}</td>"
                f"<td><b style='color:{STATUS_COLOR[t.status]}'>{t.status}</b></td>"
                f"<td style='color:{SEV_COLOR[t.severity]}'>{t.severity}</td>"
                f"<td>{t.affected_rows}</td>"
                f"<td>{e(t.detail)}{ex_html}{fix_html}</td></tr>")

    test_rows = "".join(test_row(t) for t in sorted(
        report.tests, key=lambda t: (SEV_ORDER[t.severity], t.category)))

    split_rows = "".join(
        f"<tr><td>{e(k)}</td><td>{v['rows']}</td><td>{v['cols']}</td><td>{v['memory_mb']}</td></tr>"
        for k, v in report.splits.items())

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Dataset QA Report</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:0;background:#f4f5f7;color:#1f2328}}
 .wrap{{max-width:1000px;margin:0 auto;padding:28px}}
 h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:16px;margin:0 0 12px}}
 .muted{{color:#6b7280;font-size:12px}}
 .card{{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:18px;margin-bottom:18px}}
 .alert{{border-left:5px solid #b00020}}
 .score-big{{font-size:44px;font-weight:700}}
 table{{width:100%;border-collapse:collapse;font-size:13px}}
 th,td{{text-align:left;padding:7px 9px;border-bottom:1px solid #eee;vertical-align:top}}
 th{{background:#fafafa;font-size:12px;text-transform:uppercase;letter-spacing:.03em;color:#555}}
 pre{{background:#0f172a;color:#e2e8f0;padding:8px;border-radius:6px;font-size:12px;overflow:auto;margin:6px 0 0}}
 .rating{{padding:2px 8px;border-radius:12px;font-size:12px;font-weight:600}}
 .r-excellent{{background:#d1fae5;color:#065f46}} .r-good{{background:#dbeafe;color:#1e40af}}
 .r-needs{{background:#fef3c7;color:#92400e}} .r-critical{{background:#fee2e2;color:#991b1b}}
 tr.s-fail td{{background:#fff6f6}} tr.s-warn td{{background:#fffdf3}}
</style></head><body><div class="wrap">
 <div class="card">
   <h1>LLM Dataset QA Review</h1>
   <div class="muted">{e(report.link)} &middot; source: {e(report.source.source_type)} &middot; generated {gen}</div>
 </div>
 {crit_html}
 <div class="card">
   <h2>LLM-Readiness Scorecard</h2>
   <div class="score-big">{s['overall']}<span class="muted"> / 100 &middot; {e(s['overall_rating'])}</span></div>
   <div class="muted">FAIL {s['counts']['fail']} &middot; WARN {s['counts']['warn']} &middot; PASS {s['counts']['pass']}</div>
   <table><tr><th>Dimension</th><th>Score</th><th>Weight</th><th>Rating</th></tr>{dim_rows}</table>
 </div>
 <div class="card">
   <h2>Splits</h2>
   <table><tr><th>Split</th><th>Rows</th><th>Columns</th><th>Memory (MB)</th></tr>{split_rows}</table>
   <div class="muted" style="margin-top:6px">Format detected: {e(', '.join(report.analysis['format']['detected_formats']))}</div>
 </div>
 <div class="card">
   <h2>Test suite ({len(report.tests)} checks on split '{e(report.primary_split)}')</h2>
   <table><tr><th>Category</th><th>Check</th><th>Status</th><th>Severity</th><th>Rows</th><th>Detail / fix</th></tr>{test_rows}</table>
 </div>
 <div class="muted">Toxicity checks are keyword surface flags for human/classifier review, not a measure of toxicity. PII checks are regex-based and may over- or under-match.</div>
</div></body></html>"""
