"""Statistical profiling: numeric outliers, class imbalance, vocabulary
richness, and numeric correlation."""

from __future__ import annotations

import re
from typing import List, Optional

import numpy as np
import pandas as pd

_WORD = re.compile(r"\b\w+\b")


def numeric_outliers(df: pd.DataFrame, z_thresh: float = 3.0,
                     max_examples: int = 5) -> List[dict]:
    out = []
    num = df.select_dtypes(include=[np.number])
    for col in num.columns:
        s = num[col].dropna()
        if len(s) < 10:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        iqr_mask = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)
        std = s.std()
        z_mask = ((s - s.mean()).abs() / std > z_thresh) if std else pd.Series(False, index=s.index)
        out.append({
            "column": col,
            "iqr_outliers": int(iqr_mask.sum()),
            "zscore_outliers": int(z_mask.sum()),
            "examples": s.index[iqr_mask].tolist()[:max_examples],
        })
    return out


def class_imbalance(df: pd.DataFrame, max_classes: int = 50) -> List[dict]:
    out = []
    n = len(df)
    for col in df.columns:
        try:
            nuniq = df[col].nunique(dropna=True)
        except TypeError:
            continue  # structured column (lists/dicts); not a label field
        if not (1 < nuniq <= max_classes):
            continue
        vc = df[col].value_counts(dropna=True)
        ratio = float(vc.max() / vc.min()) if vc.min() else float("inf")
        if ratio < 1.5:
            verdict, strategy = "balanced", "none"
        elif ratio < 10:
            verdict, strategy = "moderate_imbalance", "class weights or mild oversampling"
        else:
            verdict, strategy = "severe_imbalance", "oversample minority / undersample majority / focal loss"
        out.append({
            "column": col,
            "classes": int(nuniq),
            "majority": {"label": str(vc.index[0]), "count": int(vc.iloc[0]),
                         "pct": round(100 * vc.iloc[0] / n, 2)},
            "minority": {"label": str(vc.index[-1]), "count": int(vc.iloc[-1]),
                         "pct": round(100 * vc.iloc[-1] / n, 2)},
            "imbalance_ratio": round(ratio, 2) if ratio != float("inf") else None,
            "verdict": verdict,
            "recommended_strategy": strategy,
        })
    return out


def vocabulary_richness(series: pd.Series, sample: int = 5000) -> dict:
    texts = series.dropna().astype(str).head(sample)
    if texts.empty:
        return {"count": 0}
    tokens = []
    for t in texts:
        tokens.extend(w.lower() for w in _WORD.findall(t))
    total = len(tokens)
    if total == 0:
        return {"count": int(len(texts)), "ttr": 0.0}
    unique = len(set(tokens))
    return {
        "sampled_rows": int(len(texts)),
        "total_tokens": total,
        "unique_tokens": unique,
        "ttr": round(unique / total, 4),  # type-token ratio
    }


def numeric_correlation(df: pd.DataFrame, threshold: float = 0.8) -> List[dict]:
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return []
    corr = num.corr(numeric_only=True)
    pairs = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            v = corr.loc[a, b]
            if pd.notna(v) and abs(v) >= threshold:
                pairs.append({"col_a": a, "col_b": b, "correlation": round(float(v), 3)})
    return pairs
