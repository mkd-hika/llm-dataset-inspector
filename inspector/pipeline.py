"""End-to-end orchestration.

audit(link) -> AuditReport with: source info, per-split sizes, per-split
analysis, the flat test-result list, leakage info, and the scorecard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from . import structural as st
from . import readiness as rd
from . import profiling as pf
from .sources import detect_source, load_dataset, SourceInfo
from .tests_suite import run_tests, TestResult
from .scoring import scorecard


@dataclass
class AuditReport:
    link: str
    source: SourceInfo
    splits: Dict[str, dict]                 # split -> {rows, cols, memory_mb}
    primary_split: str
    analysis: dict                          # analysis for the primary split
    tests: List[TestResult]
    leakage: dict
    score: dict
    errors: List[str] = field(default_factory=list)


def analyse_split(df: pd.DataFrame) -> dict:
    text_cols = st.text_columns(df)
    primary = text_cols[0] if text_cols else None

    return {
        "primary_text_col": primary,
        "text_cols": text_cols,
        "schema": st.schema_report(df),
        "missingness": st.missingness_report(df),
        "cardinality": st.cardinality_report(df),
        "dtype_issues": st.dtype_consistency(df),
        "duplicates": st.duplicate_report(df, text_cols),
        "format": (fmt := rd.detect_format(df)),
        "turn_validation": rd.validate_turn_ordering(df, fmt),
        "tokens": {c: rd.token_distribution(df[c]) for c in text_cols},
        "length_hist": {primary: rd.length_histogram(df[primary])} if primary else {},
        "text_quality": {c: rd.text_quality(df[c]) for c in text_cols},
        "noise": {primary: rd.noise_signals(df[primary])} if primary else {},
        "pii": {c: rd.pii_scan(df[c]) for c in text_cols},
        "toxicity": {c: rd.toxicity_surface(df[c]) for c in text_cols},
        "vocab": {c: pf.vocabulary_richness(df[c]) for c in text_cols},
        "outliers": pf.numeric_outliers(df),
        "imbalance": pf.class_imbalance(df),
        "correlation": pf.numeric_correlation(df),
    }


def audit(link: str, row_limit: int | None = None) -> AuditReport:
    info = detect_source(link)
    errors: List[str] = []

    splits = load_dataset(info, row_limit=row_limit)

    split_meta = {
        name: {
            "rows": int(len(df)),
            "cols": int(df.shape[1]),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        }
        for name, df in splits.items()
    }

    primary_split = next((s for s in splits if "train" in s.lower()), next(iter(splits)))
    primary_df = splits[primary_split]

    analysis = analyse_split(primary_df)
    leakage = st.split_leakage(splits, analysis["text_cols"])
    tests = run_tests(primary_df, analysis, leakage)
    score = scorecard(tests)

    return AuditReport(
        link=link, source=info, splits=split_meta, primary_split=primary_split,
        analysis=analysis, tests=tests, leakage=leakage, score=score, errors=errors,
    )
