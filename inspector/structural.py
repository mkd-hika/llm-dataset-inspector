"""Structural analysis of a loaded dataset.

All functions are deterministic and return plain dicts/lists so the results can
be serialised to JSON and rendered in any frontend.
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List

import pandas as pd


_WS = re.compile(r"\s+")


def _normalise_text(value: object) -> str:
    return _WS.sub(" ", str(value)).strip().lower()


def is_textlike(series: pd.Series) -> bool:
    """True for object or pandas string dtypes (pandas >=2 may use either)."""
    return series.dtype == object or pd.api.types.is_string_dtype(series.dtype)


def _is_structured(value: object) -> bool:
    return isinstance(value, (list, dict, set))


def is_structured_column(series: pd.Series, sample: int = 50) -> bool:
    """True if the column predominantly holds lists/dicts (e.g. chat turns)."""
    vals = series.dropna().head(sample)
    if len(vals) == 0:
        return False
    return sum(_is_structured(v) for v in vals) > 0.5 * len(vals)


def _to_hashable(v: object) -> object:
    if _is_structured(v):
        import json
        try:
            return json.dumps(v, sort_keys=True, default=str)
        except Exception:  # noqa: BLE001
            return str(v)
    return v


def _hashable_series(s: pd.Series) -> pd.Series:
    return s.map(_to_hashable)


def safe_nunique(s: pd.Series) -> int:
    try:
        return int(s.nunique(dropna=True))
    except TypeError:
        return int(_hashable_series(s).nunique(dropna=True))


def text_columns(df: pd.DataFrame, min_avg_len: int = 12) -> List[str]:
    """Heuristically identify free-text columns worth deep inspection.

    Structured columns (lists/dicts such as chat-turn arrays) are excluded.
    """
    cols = []
    for col in df.columns:
        series = df[col]
        if not is_textlike(series) or is_structured_column(series):
            continue
        sample = series.dropna().astype(str).head(200)
        if len(sample) and sample.str.len().mean() >= min_avg_len:
            cols.append(col)
    return cols


def schema_report(df: pd.DataFrame) -> List[dict]:
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        nulls = int(s.isna().sum())
        rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "non_null": int(n - nulls),
            "null_count": nulls,
            "null_pct": round(100 * nulls / n, 2) if n else 0.0,
            "unique": safe_nunique(s),
        })
    return rows


def missingness_report(df: pd.DataFrame) -> dict:
    n = len(df)
    per_col = {}
    for col in df.columns:
        nulls = int(df[col].isna().sum())
        per_col[col] = {"null_count": nulls, "null_pct": round(100 * nulls / n, 2) if n else 0.0}

    # Pattern signal: do nulls co-occur across columns (suggests structural/MAR)
    null_mask = df.isna()
    fully_empty_rows = int(null_mask.all(axis=1).sum())
    cols_with_nulls = [c for c in df.columns if per_col[c]["null_count"] > 0]
    cooccurrence = None
    if len(cols_with_nulls) >= 2:
        corr = null_mask[cols_with_nulls].astype(int).corr()
        high_pairs = []
        for i, a in enumerate(cols_with_nulls):
            for b in cols_with_nulls[i + 1:]:
                val = corr.loc[a, b]
                if pd.notna(val) and abs(val) >= 0.5:
                    high_pairs.append({"col_a": a, "col_b": b, "corr": round(float(val), 2)})
        cooccurrence = high_pairs
    return {
        "per_column": per_col,
        "fully_empty_rows": fully_empty_rows,
        "correlated_missingness": cooccurrence,
    }


def cardinality_report(df: pd.DataFrame, high_card_ratio: float = 0.9) -> List[dict]:
    n = len(df)
    out = []
    for col in df.columns:
        uniq = safe_nunique(df[col])
        ratio = uniq / n if n else 0.0
        out.append({
            "column": col,
            "unique": uniq,
            "unique_ratio": round(ratio, 4),
            "high_cardinality": bool(ratio >= high_card_ratio and n > 1),
            "constant": bool(uniq <= 1),
        })
    return out


def dtype_consistency(df: pd.DataFrame, sample: int = 500) -> List[dict]:
    """Flag object columns that mix python types (e.g. ints + strings + dicts)."""
    issues = []
    for col in df.columns:
        if df[col].dtype != object:
            continue
        vals = df[col].dropna().head(sample)
        kinds = {type(v).__name__ for v in vals}
        kinds.discard("NoneType")
        if len(kinds) > 1:
            issues.append({"column": col, "observed_types": sorted(kinds)})
    return issues


def duplicate_report(df: pd.DataFrame, text_cols: List[str]) -> dict:
    n = len(df)
    try:
        exact = int(df.duplicated().sum())
    except TypeError:
        exact = int(df.map(_to_hashable).duplicated().sum())

    near = {}
    for col in text_cols:
        norm = df[col].dropna().map(_normalise_text)
        dup_norm = int(norm.duplicated().sum())
        near[col] = {
            "normalised_duplicate_rows": dup_norm,
            "normalised_duplicate_pct": round(100 * dup_norm / n, 2) if n else 0.0,
        }
    return {
        "exact_duplicate_rows": exact,
        "exact_duplicate_pct": round(100 * exact / n, 2) if n else 0.0,
        "near_duplicate_by_column": near,
    }


def split_leakage(splits: Dict[str, pd.DataFrame], text_cols: List[str]) -> dict:
    """Detect rows that appear in train AND a held-out split (normalised match)."""
    if len(splits) < 2 or not text_cols:
        return {"checked": False, "reason": "Need >=2 splits and a text column."}

    primary = text_cols[0]
    train_name = next((s for s in splits if "train" in s.lower()), None)
    if train_name is None:
        return {"checked": False, "reason": "No split named like 'train'."}

    train_hashes = {
        hashlib.md5(_normalise_text(v).encode()).hexdigest()
        for v in splits[train_name][primary].dropna()
    }
    leaks = {}
    for name, sdf in splits.items():
        if name == train_name or primary not in sdf.columns:
            continue
        overlap = sum(
            1 for v in sdf[primary].dropna()
            if hashlib.md5(_normalise_text(v).encode()).hexdigest() in train_hashes
        )
        leaks[name] = {
            "leaked_rows": overlap,
            "leaked_pct": round(100 * overlap / len(sdf), 2) if len(sdf) else 0.0,
        }
    return {"checked": True, "compared_against": train_name, "column": primary, "by_split": leaks}
