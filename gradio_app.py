"""Gradio GUI for the LLM Dataset Inspector.

Run locally:
  python gradio_app.py

Share publicly (instant URL, no deploy needed):
  python gradio_app.py --share
"""

from __future__ import annotations

import argparse
import os
import tempfile

import gradio as gr
import pandas as pd

from inspector.sources import detect_source, load_dataset
from inspector.pipeline import analyse_split, AuditReport
from inspector.structural import split_leakage
from inspector.tests_suite import run_tests, FAIL, WARN
from inspector.scoring import scorecard
from inspector import report_to_html, report_to_json, data_card_md

# ── Colour maps ───────────────────────────────────────────────────────────────
_SEV_COLOR = {
    "CRITICAL": "#c0392b", "HIGH": "#d35400",
    "MEDIUM": "#b8860b", "LOW": "#3a6ea5", "INFO": "#777",
}
_STATUS_COLOR = {"PASS": "#1b7a3d", "WARN": "#b8860b", "FAIL": "#c0392b"}
_RATING_COLOR = {
    "Excellent": "#1b7a3d", "Good": "#3a6ea5",
    "Needs Work": "#b8860b", "Critical": "#c0392b",
}


def _badge(text: str, color: str) -> str:
    return (
        f'<span style="background:{color};color:#fff;padding:2px 8px;'
        f'border-radius:4px;font-size:0.82em;font-weight:700">{text}</span>'
    )


# ── HTML builders ─────────────────────────────────────────────────────────────
def _summary_html(report: AuditReport) -> str:
    s = report.score
    cnt = s["counts"]
    fmt = ", ".join(report.analysis["format"]["detected_formats"])
    total_rows = sum(v["rows"] for v in report.splits.values())
    rc = _RATING_COLOR.get(s["overall_rating"], "#777")

    cards = "".join(
        '<div style="background:#1e1e2e;border-radius:8px;padding:14px;text-align:center">'
        '<div style="font-size:0.78em;color:#888;margin-bottom:4px">' + label + '</div>'
        '<div style="font-size:1.05em;font-weight:700;color:#e0e0e0">' + str(value) + '</div>'
        '</div>'
        for label, value in [
            ("Source", report.source.source_type),
            ("Format", fmt),
            ("Total rows", f"{total_rows:,}"),
            ("Splits", len(report.splits)),
            ("Columns", next(iter(report.splits.values()))["cols"]),
        ]
    )

    dim_rows = []
    for d, v in s["dimensions"].items():
        dim_color = _RATING_COLOR.get(v["rating"], "#aaa")
        dim_badge = _badge(v["rating"], _RATING_COLOR.get(v["rating"], "#777"))
        dim_rows.append(
            '<tr style="border-bottom:1px solid #222">'
            '<td style="padding:7px 6px">' + d + '</td>'
            '<td style="padding:7px 6px;text-align:right;font-weight:700;color:' + dim_color + '">' + str(v["score"]) + '</td>'
            '<td style="padding:7px 6px;text-align:right;color:#888">' + str(int(v["weight"] * 100)) + '%</td>'
            '<td style="padding:7px 6px">' + dim_badge + '</td>'
            '</tr>'
        )
    dim_rows_html = "".join(dim_rows)

    return (
        '<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:18px">'
        + cards +
        '</div>'
        '<div style="display:grid;grid-template-columns:180px 1fr;gap:16px">'
        '<div style="background:#1e1e2e;border-radius:8px;padding:20px;text-align:center">'
        '<div style="font-size:3em;font-weight:800;color:' + rc + ';line-height:1">' + str(s["overall"]) + '</div>'
        '<div style="color:#666;font-size:0.8em">/ 100</div>'
        '<div style="font-size:1.1em;font-weight:600;color:' + rc + ';margin:8px 0">' + s["overall_rating"] + '</div>'
        '<div style="font-size:0.85em;line-height:2">'
        '<span style="color:#c0392b">' + str(cnt["fail"]) + ' FAIL</span>&nbsp;'
        '<span style="color:#b8860b">' + str(cnt["warn"]) + ' WARN</span>&nbsp;'
        '<span style="color:#1b7a3d">' + str(cnt["pass"]) + ' PASS</span>&nbsp;'
        '<span style="color:#c0392b">' + str(cnt["critical"]) + ' CRITICAL</span>'
        '</div></div>'
        '<div style="background:#1e1e2e;border-radius:8px;padding:16px">'
        '<table style="width:100%;border-collapse:collapse;font-size:0.9em">'
        '<thead><tr style="color:#888;font-size:0.8em;border-bottom:1px solid #333">'
        '<th style="padding:6px;text-align:left">Dimension</th>'
        '<th style="padding:6px;text-align:right">Score</th>'
        '<th style="padding:6px;text-align:right">Weight</th>'
        '<th style="padding:6px;text-align:left">Rating</th>'
        '</tr></thead>'
        '<tbody>' + dim_rows_html + '</tbody>'
        '</table></div></div>'
    )


def _issues_html(report: AuditReport, primary_df: pd.DataFrame | None) -> str:
    by_cat: dict = {}
    for t in report.tests:
        by_cat.setdefault(t.category, []).append(t)

    sections = []
    for cat, items in by_cat.items():
        n_fail = sum(1 for t in items if t.status == FAIL)
        n_warn = sum(1 for t in items if t.status == WARN)
        color = "#c0392b" if n_fail else ("#b8860b" if n_warn else "#1b7a3d")

        rows = []
        for t in items:
            sc = _STATUS_COLOR[t.status]
            sevc = _SEV_COLOR.get(t.severity, "#777")

            sample = ""
            if t.status != "PASS" and t.example_rows and primary_df is not None:
                idx = [i for i in t.example_rows[:3] if i < len(primary_df)]
                if idx:
                    tbl = primary_df.iloc[idx].to_html(
                        border=0, classes="stbl", max_rows=3,
                    )
                    sample = (
                        f"<details style='margin-top:6px'><summary style='cursor:pointer;"
                        f"font-size:0.8em;color:#888'>Sample flagged rows</summary>"
                        f"<div style='overflow-x:auto'>{tbl}</div></details>"
                    )

            fix = ""
            if t.fix and t.status != "PASS":
                fix = (
                    f"<details style='margin-top:4px'><summary style='cursor:pointer;"
                    f"font-size:0.8em;color:#888'>Fix snippet</summary>"
                    f"<pre style='background:#111;color:#ccc;padding:10px;border-radius:6px;"
                    f"overflow-x:auto;font-size:0.78em;margin:6px 0'>{t.fix}</pre></details>"
                )

            rows.append(
                f"<tr style='border-bottom:1px solid #222'>"
                f"<td style='padding:8px 6px;white-space:nowrap'>{_badge(t.status, sc)}</td>"
                f"<td style='padding:8px 6px'><strong>{t.name}</strong>"
                f"<br><span style='font-size:0.82em;color:#aaa'>{t.detail}</span>{sample}{fix}</td>"
                f"<td style='padding:8px 6px;white-space:nowrap'>{_badge(t.severity, sevc)}</td>"
                f"<td style='padding:8px 6px;text-align:right'>{t.affected_rows:,}</td>"
                f"</tr>"
            )

        sections.append(
            f"<details {'open' if n_fail else ''}>"
            f"<summary style='cursor:pointer;padding:10px 14px;background:#1e1e2e;"
            f"border-radius:6px;margin-bottom:6px;font-weight:600;color:{color}'>"
            f"{cat} &nbsp;—&nbsp; {n_fail} fail · {n_warn} warn</summary>"
            f"<table style='width:100%;border-collapse:collapse;font-size:0.88em'>"
            f"<thead><tr style='color:#777;font-size:0.8em;border-bottom:2px solid #333'>"
            f"<th style='padding:6px;text-align:left'>Status</th>"
            f"<th style='padding:6px;text-align:left'>Check</th>"
            f"<th style='padding:6px;text-align:left'>Severity</th>"
            f"<th style='padding:6px;text-align:right'>Rows</th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table></details>"
        )

    style = (
        "<style>.stbl{width:100%;font-size:0.76em;border-collapse:collapse}"
        ".stbl td,.stbl th{padding:3px 6px;border:1px solid #333}</style>"
    )
    return style + "".join(sections)


def _columns_html(report: AuditReport) -> str:
    cols = report.analysis["text_cols"]
    if not cols:
        return "<p style='color:#aaa'>No text columns detected.</p>"

    parts = []
    for col in cols:
        miss_pct = (
            report.analysis["missingness"]
            .get("per_column", {}).get(col, {}).get("null_pct", 0)
        )
        tok = report.analysis["tokens"].get(col, {})
        tq  = report.analysis["text_quality"].get(col, {})
        pii = report.analysis["pii"].get(col, {})
        pii_total = sum(v.get("rows", 0) for v in pii.values()) if pii else 0

        metrics = [
            ("Missing",          f"{miss_pct:.1f}%"),
            ("Median tokens",    int(tok["p50"]) if tok.get("count") else "—"),
            ("Max tokens",       tok.get("max", "—") if tok.get("count") else "—"),
            ("Total tokens",     f"{tok.get('total_tokens', 0):,}" if tok.get("count") else "—"),
            ("Empty cells",      tq.get("empty_or_whitespace", {}).get("rows", 0)),
            ("HTML bleed",       tq.get("html_bleed", {}).get("rows", 0)),
            ("Encoding errors",  tq.get("encoding_artifacts", {}).get("rows", 0)),
            ("PII hits",         pii_total),
        ]

        grid = "".join(
            f'<div style="background:#16213e;border-radius:6px;padding:12px;text-align:center">'
            f'<div style="font-size:1.25em;font-weight:700;color:#e0e0e0">{v}</div>'
            f'<div style="font-size:0.75em;color:#888;margin-top:3px">{k}</div>'
            f'</div>'
            for k, v in metrics
        )

        pii_rows = "".join(
            f"<tr><td style='padding:3px 8px'>{k}</td>"
            f"<td style='padding:3px 8px;text-align:right;color:#c0392b'>{v.get('rows',0)}</td></tr>"
            for k, v in pii.items() if v.get("rows", 0) > 0
        ) if pii_total > 0 else ""

        pii_section = (
            f"<p style='font-size:0.82em;color:#c0392b;margin:10px 0 4px'>PII breakdown:</p>"
            f"<table style='font-size:0.83em;border-collapse:collapse'>{pii_rows}</table>"
        ) if pii_rows else ""

        parts.append(
            f"<details open style='margin-bottom:12px'>"
            f"<summary style='cursor:pointer;padding:10px 14px;background:#1e1e2e;"
            f"border-radius:6px;font-weight:600;margin-bottom:8px'>{col}</summary>"
            f"<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:8px'>{grid}</div>"
            f"{pii_section}</details>"
        )

    return "".join(parts)


# ── Core audit function ───────────────────────────────────────────────────────
def run_audit(link: str, row_start: int, row_end: int, progress=gr.Progress()):
    if not link.strip():
        raise gr.Error("Enter a dataset URL or local path.")

    start = int(row_start) if row_start else 0
    end   = int(row_end)   if row_end   else 0
    limit = (end - start)  if end > start else None

    try:
        progress(0.10, desc="Detecting source…")
        info = detect_source(link.strip())

        progress(0.20, desc=f"Streaming dataset ({info.source_type})…")
        splits = load_dataset(info, row_limit=limit, row_start=start)

        split_meta = {
            name: {
                "rows": int(len(df)), "cols": int(df.shape[1]),
                "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
            }
            for name, df in splits.items()
        }
        primary_split = next(
            (s for s in splits if "train" in s.lower()), next(iter(splits))
        )
        primary_df = splits[primary_split]

        progress(0.45, desc=f"Analysing {len(primary_df):,} rows, {primary_df.shape[1]} cols…")
        analysis = analyse_split(primary_df)

        progress(0.65, desc="Checking leakage…")
        leakage = split_leakage(splits, analysis["text_cols"])

        progress(0.78, desc="Running test suite…")
        tests = run_tests(primary_df, analysis, leakage)

        progress(0.92, desc="Scoring…")
        score = scorecard(tests)

        report = AuditReport(
            link=link, source=info, splits=split_meta,
            primary_split=primary_split, analysis=analysis,
            tests=tests, leakage=leakage, score=score,
        )
        progress(1.0, desc="Done!")

    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(str(exc)) from exc

    # Write downloadable files
    tmp = tempfile.mkdtemp()
    html_path = os.path.join(tmp, "dataset_qa_report.html")
    json_path = os.path.join(tmp, "dataset_qa_metadata.json")
    card_path = os.path.join(tmp, "data_card.md")

    open(html_path, "w").write(report_to_html(report))
    open(json_path, "w").write(report_to_json(report))
    open(card_path, "w").write(data_card_md(report))

    splits_df = (
        pd.DataFrame(report.splits).T.rename_axis("split").reset_index()
    )

    return (
        _summary_html(report),
        splits_df,
        _issues_html(report, primary_df),
        _columns_html(report),
        html_path,
        json_path,
        card_path,
    )


# ── Layout ────────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="LLM Dataset QA Inspector",
    theme=gr.themes.Base(primary_hue="blue"),
    css=".contain{max-width:1100px} footer{display:none}",
) as demo:

    gr.Markdown("# LLM Dataset QA Inspector")
    gr.Markdown("**MKD Co. Ltd.**")
    gr.Markdown(
        "Paste a dataset link to inspect cleaning, formatting, safety, and LLM-readiness."
    )

    with gr.Row():
        link_box = gr.Textbox(
            label="Dataset link or local path",
            placeholder="https://huggingface.co/datasets/… or local .jsonl/.csv/.parquet",
            scale=4,
        )
        start_box = gr.Number(label="From row", value=0, minimum=0, precision=0, scale=1)
        end_box   = gr.Number(label="To row (0 = all)", value=0, minimum=0, precision=0, scale=1)
        run_btn    = gr.Button("Inspect", variant="primary", scale=1)
        cancel_btn = gr.Button("Cancel",    variant="stop",    scale=1)
        clear_btn  = gr.Button("New Inspection", scale=1)

    with gr.Tabs():
        with gr.Tab("Summary"):
            summary_out = gr.HTML()
            splits_out  = gr.Dataframe(
                label="Splits", interactive=False, wrap=True,
            )

        with gr.Tab("Test Suite"):
            issues_out = gr.HTML()

        with gr.Tab("Column Analysis"):
            columns_out = gr.HTML()

        with gr.Tab("Export"):
            gr.Markdown("Download the full audit as a report, raw metadata, or a model card.")
            with gr.Row():
                html_file = gr.File(label="HTML report",    file_count="single")
                json_file = gr.File(label="JSON metadata",  file_count="single")
                card_file = gr.File(label="data_card.md",   file_count="single")

    _outputs = [summary_out, splits_out, issues_out, columns_out,
                html_file, json_file, card_file]

    run_event = run_btn.click(
        fn=run_audit,
        inputs=[link_box, start_box, end_box],
        outputs=_outputs,
    )
    submit_event = link_box.submit(
        fn=run_audit,
        inputs=[link_box, start_box, end_box],
        outputs=_outputs,
    )
    cancel_btn.click(fn=None, cancels=[run_event, submit_event])
    clear_btn.click(
        fn=lambda: ("", 0, 0, None, None, None, None, None, None, None),
        inputs=[],
        outputs=[link_box, start_box, end_box, summary_out, splits_out,
                 issues_out, columns_out, html_file, json_file, card_file],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo.launch(
        server_name="0.0.0.0",   # required for HF Spaces
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
