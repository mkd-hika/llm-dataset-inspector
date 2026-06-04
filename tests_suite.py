"""Automated QA test suite.

Consumes the per-split analysis produced by pipeline.analyse_split and emits a
flat list of test results. Each result is PASS / WARN / FAIL with the number of
affected rows, example row indices, a severity, a remediation snippet, and an
effort estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List

import pandas as pd

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass
class TestResult:
    category: str
    name: str
    status: str
    affected_rows: int
    severity: str          # CRITICAL | HIGH | MEDIUM | LOW | INFO
    detail: str
    example_rows: list
    fix: str
    effort: str            # trivial | small | medium | large

    def as_dict(self):
        return asdict(self)


def _r(**kw) -> TestResult:
    kw.setdefault("example_rows", [])
    kw.setdefault("fix", "")
    kw.setdefault("effort", "small")
    return TestResult(**kw)


def run_tests(df: pd.DataFrame, analysis: dict, leakage: dict) -> List[TestResult]:
    res: List[TestResult] = []
    n = len(df)
    fmt = analysis["format"]
    primary = analysis.get("primary_text_col")

    # ---- SCHEMA ----
    schema = {c["column"]: c for c in analysis["schema"]}
    forbidden_null_cols = [c for c, m in schema.items() if m["null_count"] > 0 and m["unique"] > 0]
    res.append(_r(
        category="SCHEMA", name="Columns present", status=PASS,
        affected_rows=0, severity="INFO",
        detail=f"{df.shape[1]} columns detected: {', '.join(df.columns[:12])}"
               + ("..." if df.shape[1] > 12 else ""),
    ))
    mixed = analysis["dtype_issues"]
    res.append(_r(
        category="SCHEMA", name="Dtype consistency",
        status=FAIL if mixed else PASS,
        affected_rows=len(mixed), severity="HIGH" if mixed else "INFO",
        detail=("Mixed python types in: " + ", ".join(m["column"] for m in mixed)) if mixed
               else "No mixed-type object columns detected.",
        fix="df['col'] = df['col'].astype(str)  # or coerce to the intended single type",
        effort="small",
    ))

    # ---- INTEGRITY ----
    dup = analysis["duplicates"]
    exact = dup["exact_duplicate_rows"]
    res.append(_r(
        category="INTEGRITY", name="Exact duplicate rows",
        status=FAIL if exact else PASS,
        affected_rows=exact,
        severity="HIGH" if exact > 0.01 * max(n, 1) else ("MEDIUM" if exact else "INFO"),
        detail=f"{exact} exact duplicate rows ({dup['exact_duplicate_pct']}%).",
        fix="df = df.drop_duplicates().reset_index(drop=True)",
        effort="trivial",
    ))
    if primary:
        near = dup["near_duplicate_by_column"].get(primary, {})
        nd = near.get("normalised_duplicate_rows", 0)
        res.append(_r(
            category="INTEGRITY", name=f"Near-duplicate text ({primary})",
            status=WARN if nd else PASS, affected_rows=nd,
            severity="MEDIUM" if nd else "INFO",
            detail=f"{nd} rows duplicate after whitespace/case normalisation "
                   f"({near.get('normalised_duplicate_pct', 0)}%).",
            fix=("key = df['%s'].str.lower().str.replace(r'\\s+',' ',regex=True).str.strip()\n"
                 "df = df[~key.duplicated()].reset_index(drop=True)") % primary,
            effort="small",
        ))
    if leakage.get("checked"):
        leaked = sum(v["leaked_rows"] for v in leakage["by_split"].values())
        res.append(_r(
            category="INTEGRITY", name="Train/eval split leakage",
            status=FAIL if leaked else PASS, affected_rows=leaked,
            severity="CRITICAL" if leaked else "INFO",
            detail=(f"{leaked} eval rows also appear in '{leakage['compared_against']}'. "
                    "This inflates evaluation metrics.") if leaked
                   else "No normalised overlap between train and eval splits.",
            fix="# Remove eval rows whose normalised text exists in train, then re-split.",
            effort="medium",
        ))

    # ---- QUALITY ----
    if primary:
        tq = analysis["text_quality"].get(primary, {})
        empty = tq.get("empty_or_whitespace", {}).get("rows", 0)
        res.append(_r(
            category="QUALITY", name=f"Empty/whitespace text ({primary})",
            status=FAIL if empty else PASS, affected_rows=empty,
            severity="HIGH" if empty else "INFO",
            detail=f"{empty} rows are empty or whitespace-only.",
            example_rows=tq.get("empty_or_whitespace", {}).get("examples", []),
            fix=f"df = df[df['{primary}'].astype(str).str.strip() != ''].reset_index(drop=True)",
            effort="trivial",
        ))
        html = tq.get("html_bleed", {}).get("rows", 0)
        res.append(_r(
            category="QUALITY", name=f"HTML/XML bleed ({primary})",
            status=WARN if html else PASS, affected_rows=html,
            severity="MEDIUM" if html else "INFO",
            detail=f"{html} rows contain HTML/XML tags.",
            example_rows=tq.get("html_bleed", {}).get("examples", []),
            fix=(f"import re\n"
                 f"df['{primary}'] = df['{primary}'].str.replace(r'</?[^>]+>',' ',regex=True)"),
            effort="small",
        ))
        moji = tq.get("encoding_artifacts", {}).get("rows", 0)
        res.append(_r(
            category="QUALITY", name=f"Encoding artifacts ({primary})",
            status=WARN if moji else PASS, affected_rows=moji,
            severity="MEDIUM" if moji else "INFO",
            detail=f"{moji} rows show mojibake / replacement chars.",
            example_rows=tq.get("encoding_artifacts", {}).get("examples", []),
            fix="import ftfy; df['%s'] = df['%s'].map(ftfy.fix_text)" % (primary, primary),
            effort="small",
        ))
        tok = analysis["tokens"].get(primary, {})
        if tok.get("count"):
            over = int(tok["max"]) > 8192
            res.append(_r(
                category="QUALITY", name=f"Token budget ({primary})",
                status=WARN if over else PASS, affected_rows=0,
                severity="MEDIUM" if over else "INFO",
                detail=(f"Max sample is {tok['max']} tokens (p99={tok['p99']}). "
                        "Confirm this fits your context window."),
                fix="df = df[df['%s'].map(count_tokens) <= MAX_TOKENS]" % primary,
                effort="small",
            ))

    # ---- FORMAT ----
    detected = fmt["detected_formats"]
    res.append(_r(
        category="FORMAT", name="Instruction format",
        status=PASS if detected != ["unknown/generic_tabular"] else WARN,
        affected_rows=0,
        severity="INFO" if detected != ["unknown/generic_tabular"] else "LOW",
        detail="Detected format(s): " + ", ".join(detected),
        fix="" if detected != ["unknown/generic_tabular"]
            else "# Map columns to a known schema (Alpaca/ShareGPT/ChatML) before training.",
        effort="medium",
    ))
    turns = analysis.get("turn_validation", {})
    if turns.get("checked"):
        broken = turns["broken_ordering_rows"]
        empty_turns = turns["empty_turn_rows"]
        res.append(_r(
            category="FORMAT", name="Chat turn ordering",
            status=FAIL if broken else PASS, affected_rows=broken,
            severity="HIGH" if broken else "INFO",
            detail=f"{broken} conversations have broken role alternation.",
            example_rows=turns["broken_ordering_examples"],
            fix="# Ensure roles alternate user/assistant and start with user (or system,user).",
            effort="medium",
        ))
        res.append(_r(
            category="FORMAT", name="Empty conversation turns",
            status=WARN if empty_turns else PASS, affected_rows=empty_turns,
            severity="MEDIUM" if empty_turns else "INFO",
            detail=f"{empty_turns} conversations contain an empty turn.",
            example_rows=turns["empty_turn_examples"],
            fix="# Drop or fill empty turns before serialising the chat template.",
            effort="small",
        ))

    # ---- SAFETY ----
    if primary:
        pii = analysis["pii"].get(primary, {})
        pii_total = sum(v["rows"] for k, v in pii.items() if isinstance(v, dict))
        worst = max(((k, v["rows"]) for k, v in pii.items() if isinstance(v, dict)),
                    key=lambda kv: kv[1], default=("none", 0))
        res.append(_r(
            category="SAFETY", name=f"PII presence ({primary})",
            status=FAIL if pii_total else PASS, affected_rows=pii_total,
            severity="CRITICAL" if pii_total else "INFO",
            detail=(f"{pii_total} PII hits; most common: {worst[0]} ({worst[1]} rows). "
                    "Email/phone/SSN/IP/card patterns.") if pii_total
                   else "No PII patterns matched.",
            example_rows=(pii.get("email", {}).get("examples") or
                          next((v["examples"] for v in pii.values()
                                if isinstance(v, dict) and v.get("examples")), [])),
            fix="# Redact with presidio or regex before training:\n"
                "from presidio_anonymizer import AnonymizerEngine  # recommended",
            effort="medium",
        ))
        tox = analysis["toxicity"].get(primary, {})
        tox_total = sum(v["rows"] for k, v in tox.items() if isinstance(v, dict))
        res.append(_r(
            category="SAFETY", name=f"Toxicity surface flags ({primary})",
            status=WARN if tox_total else PASS, affected_rows=tox_total,
            severity="HIGH" if tox_total else "INFO",
            detail=(f"{tox_total} rows hit the keyword surface scan. "
                    "Keyword flag only - route to a classifier/human review.") if tox_total
                   else "No keyword surface hits (does not prove the data is clean).",
            example_rows=next((v["examples"] for k, v in tox.items()
                               if isinstance(v, dict) and v.get("examples")), []),
            fix="# Score with Detoxify or Perspective API; review before removing.",
            effort="medium",
        ))

    # ---- CONSISTENCY ----
    if {"instruction", "output"}.issubset(set(c.lower() for c in df.columns)):
        icol = next(c for c in df.columns if c.lower() == "instruction")
        ocol = next(c for c in df.columns if c.lower() == "output")
        ilen = df[icol].astype(str).str.len()
        olen = df[ocol].astype(str).str.len()
        empty_out = int((olen.fillna(0) <= 1).sum())
        ratio_bad = df.index[(olen < 0.1 * ilen) & (ilen > 50)].tolist()
        res.append(_r(
            category="CONSISTENCY", name="Empty responses",
            status=FAIL if empty_out else PASS, affected_rows=empty_out,
            severity="HIGH" if empty_out else "INFO",
            detail=f"{empty_out} rows have an empty/near-empty output.",
            fix=f"df = df[df['{ocol}'].astype(str).str.strip() != ''].reset_index(drop=True)",
            effort="trivial",
        ))
        res.append(_r(
            category="CONSISTENCY", name="Instruction/response length ratio",
            status=WARN if ratio_bad else PASS, affected_rows=len(ratio_bad),
            severity="LOW" if ratio_bad else "INFO",
            detail=f"{len(ratio_bad)} rows have a response <10% the instruction length.",
            example_rows=ratio_bad[:5],
            fix="# Manually review: long instruction with a terse answer may be truncated.",
            effort="medium",
        ))

    # ---- COVERAGE ----
    if primary:
        vr = analysis["vocab"].get(primary, {})
        ttr = vr.get("ttr", 0)
        res.append(_r(
            category="COVERAGE", name=f"Lexical diversity ({primary})",
            status=WARN if (ttr and ttr < 0.05) else PASS, affected_rows=0,
            severity="LOW" if (ttr and ttr < 0.05) else "INFO",
            detail=f"Type-token ratio = {ttr} over {vr.get('total_tokens', 0)} tokens. "
                   "Very low TTR can signal templated/repetitive data.",
            fix="# Inspect for templated boilerplate; diversify sources if needed.",
            effort="large",
        ))
    imb = analysis.get("imbalance", [])
    severe = [c for c in imb if c["verdict"] == "severe_imbalance"]
    if imb:
        res.append(_r(
            category="COVERAGE", name="Class balance",
            status=WARN if severe else PASS,
            affected_rows=0, severity="MEDIUM" if severe else "INFO",
            detail=("Severe imbalance in: " + ", ".join(c["column"] for c in severe)) if severe
                   else "No severe class imbalance detected.",
            fix="# " + (severe[0]["recommended_strategy"] if severe else "none"),
            effort="medium",
        ))
    return res
