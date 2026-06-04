"""Streamlit GUI for the LLM Dataset Inspector."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from inspector.sources import detect_source, load_dataset
from inspector.pipeline import analyse_split, AuditReport
from inspector.structural import split_leakage
from inspector.tests_suite import run_tests, FAIL, WARN
from inspector.scoring import scorecard
from inspector import report_to_html, report_to_json, data_card_md

SEV_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
STATUS_BADGE = {"PASS": ":green[PASS]", "WARN": ":orange[WARN]", "FAIL": ":red[FAIL]"}
RATING_COLOR = {"Excellent": "green", "Good": "blue", "Needs Work": "orange", "Critical": "red"}


def _sev_color(sev: str) -> str:
    return {"CRITICAL": "red", "HIGH": "orange", "MEDIUM": "orange",
            "LOW": "blue", "INFO": "gray"}.get(sev, "gray")


st.set_page_config(page_title="LLM Dataset QA Inspector — MKD Co. Ltd.", layout="wide")

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "active_report" not in st.session_state:
    st.session_state.active_report = None
if "active_df" not in st.session_state:
    st.session_state.active_df = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Inspect")
    link = st.text_input(
        "Dataset link or local path",
        placeholder="https://huggingface.co/datasets/… or .jsonl/.csv/.parquet",
    )
    row_limit = st.number_input(
        "Row sample limit (0 = all rows)", min_value=0, value=5000, step=1000,
        help="Caps rows per split for a faster pass on large datasets.",
    )
    run_btn = st.button("Run audit", type="primary", use_container_width=True)

    st.markdown("---")

    if st.session_state.history:
        st.subheader("Recent audits")
        for entry in reversed(st.session_state.history[-5:]):
            color = RATING_COLOR.get(entry["rating"], "gray")
            label = f":{color}[{entry['score']}/100] · {entry['rating']} · {entry['time']}"
            if st.button(label, key=entry["key"], use_container_width=True, help=entry["link"]):
                st.session_state.active_report = entry["report"]
                st.session_state.active_df = entry["df"]

    st.markdown("---")
    st.caption(
        "Supports HuggingFace Hub, GitHub raw, and direct "
        "CSV/TSV/JSON/JSONL/Parquet URLs (optionally .gz)."
    )


# ── Audit runner ──────────────────────────────────────────────────────────────
def _run_audit(link: str, limit: int | None) -> tuple[AuditReport, pd.DataFrame]:
    with st.status("Running audit…", expanded=True) as status:
        try:
            st.write("Detecting source…")
            info = detect_source(link)

            st.write(f"Streaming dataset ({info.source_type})…")
            splits = load_dataset(info, row_limit=limit)

            split_meta = {
                name: {
                    "rows": int(len(df)),
                    "cols": int(df.shape[1]),
                    "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
                }
                for name, df in splits.items()
            }
            primary_split = next(
                (s for s in splits if "train" in s.lower()), next(iter(splits))
            )
            primary_df = splits[primary_split]

            st.write(
                f"Analysing {len(primary_df):,} rows "
                f"across {primary_df.shape[1]} columns…"
            )
            analysis = analyse_split(primary_df)

            st.write("Checking split leakage…")
            leakage = split_leakage(splits, analysis["text_cols"])

            st.write("Running test suite…")
            tests = run_tests(primary_df, analysis, leakage)

            st.write("Scoring…")
            score = scorecard(tests)

            report = AuditReport(
                link=link, source=info, splits=split_meta,
                primary_split=primary_split, analysis=analysis,
                tests=tests, leakage=leakage, score=score,
            )
            status.update(label="Audit complete", state="complete", expanded=False)
            return report, primary_df

        except Exception as exc:
            status.update(label="Audit failed", state="error", expanded=True)
            raise exc


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("LLM Dataset QA Inspector")
st.caption("MKD Co. Ltd. — Paste a dataset link to audit cleaning, formatting, safety, and LLM-readiness.")

if run_btn and link:
    try:
        report, primary_df = _run_audit(link, row_limit or None)
        st.session_state.active_report = report
        st.session_state.active_df = primary_df
        st.session_state.history.append({
            "key": f"hist_{len(st.session_state.history)}",
            "link": link,
            "time": datetime.now().strftime("%H:%M"),
            "score": report.score["overall"],
            "rating": report.score["overall_rating"],
            "report": report,
            "df": primary_df,
        })
    except Exception as exc:
        st.error(f"Audit failed: {exc}")
        st.info(
            "Check the dataset URL and ensure you have access. "
            "For gated HuggingFace datasets, run `huggingface-cli login` first."
        )
        st.stop()
elif run_btn:
    st.warning("Enter a dataset link or local path first.")

# ── Display ───────────────────────────────────────────────────────────────────
report: AuditReport | None = st.session_state.active_report
primary_df: pd.DataFrame | None = st.session_state.active_df

if report is None:
    st.info("Enter a dataset link in the sidebar and click Run audit.")
    st.stop()

s = report.score

# 1 · Dataset identity
st.subheader("Dataset identity")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Source", report.source.source_type)
fmt_label = ", ".join(report.analysis["format"]["detected_formats"])
c2.metric("Format", fmt_label, help=fmt_label)
total_rows = sum(v["rows"] for v in report.splits.values())
c3.metric("Total rows", f"{total_rows:,}")
c4.metric("Splits", str(len(report.splits)))
st.dataframe(
    pd.DataFrame(report.splits).T.rename_axis("split").reset_index(),
    use_container_width=True, hide_index=True,
)

# 2 · Scorecard
st.subheader("LLM-Readiness scorecard")
sc_left, sc_right = st.columns([1, 3])
rating_color = RATING_COLOR.get(s["overall_rating"], "gray")
sc_left.metric("Overall", f"{s['overall']} / 100")
sc_left.markdown(f":{rating_color}[**{s['overall_rating']}**]")
cnt = s["counts"]
sc_left.markdown(
    f":red[**{cnt['fail']}** FAIL]&nbsp; "
    f":orange[**{cnt['warn']}** WARN]&nbsp; "
    f":green[**{cnt['pass']}** PASS]&nbsp; "
    f":red[**{cnt['critical']}** CRITICAL]"
)
dim_df = pd.DataFrame([
    {"Dimension": d, "Score": v["score"], "Weight": f"{int(v['weight'] * 100)}%", "Rating": v["rating"]}
    for d, v in s["dimensions"].items()
])
sc_right.dataframe(dim_df, use_container_width=True, hide_index=True)

# 3 · Critical callout
crit = [t for t in report.tests if t.severity in ("CRITICAL", "HIGH") and t.status in (FAIL, WARN)]
if crit:
    st.subheader("Critical and high-severity issues")
    for t in sorted(crit, key=lambda t: SEV_ORDER[t.severity]):
        st.error(
            f"[{t.severity}] {t.category} / {t.name} — {t.detail} ({t.affected_rows} rows)"
        )

# 4 · Test suite
st.subheader(f"Test suite — {len(report.tests)} checks on split '{report.primary_split}'")
by_cat: dict = {}
for t in report.tests:
    by_cat.setdefault(t.category, []).append(t)

for cat, items in by_cat.items():
    n_fail = sum(1 for t in items if t.status == FAIL)
    n_warn = sum(1 for t in items if t.status == WARN)
    header = f"{cat}  —  {n_fail} fail · {n_warn} warn"
    with st.expander(header, expanded=(n_fail > 0)):
        for t in items:
            st.markdown(
                f"{STATUS_BADGE[t.status]} **{t.name}**"
                f"&nbsp; :{_sev_color(t.severity)}[{t.severity}]"
                f"&nbsp; {t.affected_rows} rows"
            )
            st.write(t.detail)

            # Sample flagged rows
            if t.example_rows and primary_df is not None:
                valid_idx = [i for i in t.example_rows[:5] if i < len(primary_df)]
                if valid_idx:
                    st.caption("Sample flagged rows:")
                    st.dataframe(
                        primary_df.iloc[valid_idx],
                        use_container_width=True,
                        hide_index=False,
                    )
            elif t.example_rows:
                st.caption("Row indices: " + ", ".join(str(x) for x in t.example_rows[:10]))

            if t.fix:
                with st.expander("Fix snippet", expanded=False):
                    st.code(t.fix, language="python")

# 5 · Column analysis
if report.analysis["text_cols"]:
    st.subheader("Column analysis")
    tabs = st.tabs(report.analysis["text_cols"])
    for tab, col in zip(tabs, report.analysis["text_cols"]):
        with tab:
            miss_pct = (
                report.analysis["missingness"]
                .get("per_column", {})
                .get(col, {})
                .get("null_pct", 0)
            )
            tok = report.analysis["tokens"].get(col, {})
            tq = report.analysis["text_quality"].get(col, {})
            pii = report.analysis["pii"].get(col, {})
            pii_total = sum(v.get("rows", 0) for v in pii.values()) if pii else 0

            # Top metrics row
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Missing", f"{miss_pct:.1f}%")
            if tok.get("count"):
                m2.metric("Median tokens", int(tok["p50"]))
                m3.metric("Max tokens", tok["max"])
            m4.metric("PII hits", pii_total,
                      delta=None if pii_total == 0 else f"{pii_total} rows",
                      delta_color="inverse")
            m5.metric("Empty cells", tq.get("empty_or_whitespace", {}).get("rows", 0))

            # Token histogram
            if tok.get("count"):
                st.caption(f"Tokenizer: {tok['tokenizer']}")
                pct_df = pd.DataFrame({
                    "percentile": ["p5", "p25", "p50", "p75", "p95", "p99"],
                    "tokens": [tok["p5"], tok["p25"], tok["p50"],
                               tok["p75"], tok["p95"], tok["p99"]],
                }).set_index("percentile")
                st.bar_chart(pct_df)

            # Quality breakdown
            q1, q2, q3 = st.columns(3)
            q1.metric("HTML bleed", tq.get("html_bleed", {}).get("rows", 0))
            q2.metric("Encoding artifacts", tq.get("encoding_artifacts", {}).get("rows", 0))
            q3.metric("Whitespace anomalies", tq.get("whitespace_anomalies", {}).get("rows", 0))

            # PII breakdown
            if pii_total > 0:
                st.caption("PII breakdown:")
                pii_df = pd.DataFrame([
                    {"Type": k, "Rows": v.get("rows", 0)}
                    for k, v in pii.items() if v.get("rows", 0) > 0
                ])
                st.dataframe(pii_df, use_container_width=True, hide_index=True)

# 6 · Export
st.subheader("Export")
e1, e2, e3 = st.columns(3)
e1.download_button(
    "HTML report", report_to_html(report),
    file_name="dataset_qa_report.html", mime="text/html",
    use_container_width=True,
)
e2.download_button(
    "JSON metadata", report_to_json(report),
    file_name="dataset_qa_metadata.json", mime="application/json",
    use_container_width=True,
)
e3.download_button(
    "data_card.md", data_card_md(report),
    file_name="data_card.md", mime="text/markdown",
    use_container_width=True,
)
