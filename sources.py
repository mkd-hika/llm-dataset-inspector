"""Dataset source detection and loading.

Supports:
  - HuggingFace Hub dataset URLs (https://huggingface.co/datasets/<owner>/<name>)
  - GitHub blob/raw URLs (auto-converted to raw.githubusercontent.com)
  - Direct file URLs ending in .csv / .tsv / .json / .jsonl / .parquet (optionally .gz)
  - Cloud object URLs (s3://, gs://, https://...amazonaws.com, https://storage.googleapis.com)
  - Local file paths (for testing / offline use)

The loader returns a dict of {split_name: pandas.DataFrame}. When a source
exposes no split concept (a single file), the split is named "data".
"""

from __future__ import annotations

import io
import gzip
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict
from urllib.parse import urlparse

import pandas as pd

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None


SUPPORTED_TABULAR = (".csv", ".tsv", ".json", ".jsonl", ".parquet")


@dataclass
class SourceInfo:
    raw_input: str
    source_type: str            # huggingface | github | direct_file | cloud | local | unknown
    resolved_url: str
    file_format: str            # csv | tsv | json | jsonl | parquet | unknown
    compression: str            # none | gzip
    notes: list = field(default_factory=list)


def detect_source(link: str) -> SourceInfo:
    """Classify a user-provided link without downloading it."""
    link = link.strip()
    parsed = urlparse(link)
    lower = link.lower()

    compression = "gzip" if lower.endswith(".gz") else "none"
    stem = lower[:-3] if compression == "gzip" else lower
    file_format = "unknown"
    for ext in SUPPORTED_TABULAR:
        if stem.endswith(ext):
            file_format = ext.lstrip(".")
            break

    # Local path
    if not parsed.scheme or parsed.scheme == "file" or os.path.exists(link):
        return SourceInfo(link, "local", link, file_format, compression)

    # HuggingFace
    if "huggingface.co/datasets/" in lower:
        return SourceInfo(link, "huggingface", link, file_format, compression,
                          notes=["Will load via the `datasets` library."])

    # GitHub -> normalise to raw
    if "github.com" in lower and "/blob/" in link:
        resolved = (link
                    .replace("github.com", "raw.githubusercontent.com")
                    .replace("/blob/", "/"))
        return SourceInfo(link, "github", resolved, file_format, compression,
                          notes=["Converted GitHub blob URL to raw URL."])
    if "raw.githubusercontent.com" in lower:
        return SourceInfo(link, "github", link, file_format, compression)

    # Cloud buckets
    if (parsed.scheme in ("s3", "gs") or "amazonaws.com" in lower
            or "storage.googleapis.com" in lower or "blob.core.windows.net" in lower):
        return SourceInfo(link, "cloud", link, file_format, compression,
                          notes=["Cloud object storage URL."])

    # Generic direct file
    if file_format != "unknown":
        return SourceInfo(link, "direct_file", link, file_format, compression)

    return SourceInfo(link, "unknown", link, file_format, compression,
                      notes=["Could not infer a supported file format from the URL."])


def _read_buffer(buf: bytes, file_format: str, compression: str) -> pd.DataFrame:
    if compression == "gzip":
        buf = gzip.decompress(buf)
    bio = io.BytesIO(buf)

    if file_format == "csv":
        return pd.read_csv(bio)
    if file_format == "tsv":
        return pd.read_csv(bio, sep="\t")
    if file_format == "parquet":
        return pd.read_parquet(bio)
    if file_format == "jsonl":
        text = buf.decode("utf-8", errors="replace")
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        return pd.json_normalize(rows, max_level=1)
    if file_format == "json":
        text = buf.decode("utf-8", errors="replace")
        data = json.loads(text)
        if isinstance(data, dict):
            # common patterns: {"data": [...]} or {"rows": [...]}
            for key in ("data", "rows", "examples", "instances"):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
        if isinstance(data, dict):
            data = [data]
        return pd.json_normalize(data, max_level=1)
    raise ValueError(f"Unsupported file format: {file_format!r}")


def load_dataset(info: SourceInfo, hf_split_limit: int | None = None) -> Dict[str, pd.DataFrame]:
    """Download/load a dataset into one DataFrame per split.

    Raises a descriptive RuntimeError on failure so the GUI can show a clean
    diagnostic instead of a stack trace.
    """
    if info.source_type == "huggingface":
        return _load_huggingface(info, hf_split_limit)

    if info.source_type == "local":
        if info.file_format == "unknown":
            raise RuntimeError(
                f"Cannot infer format for local path {info.resolved_url!r}. "
                "Expected one of: " + ", ".join(SUPPORTED_TABULAR))
        with open(info.resolved_url, "rb") as fh:
            df = _read_buffer(fh.read(), info.file_format, info.compression)
        return {"data": df}

    # Everything else is an HTTP(S) fetch
    if requests is None:
        raise RuntimeError("The `requests` library is required for remote downloads.")
    if info.file_format == "unknown":
        raise RuntimeError(
            "Could not determine the file format from the URL. Append a recognised "
            "extension (.csv, .tsv, .json, .jsonl, .parquet) or use a HuggingFace URL.")

    try:
        resp = requests.get(info.resolved_url, timeout=60, stream=True)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Download failed for {info.resolved_url}: {exc}") from exc

    df = _read_buffer(resp.content, info.file_format, info.compression)
    return {"data": df}


def _load_huggingface(info: SourceInfo, split_limit: int | None) -> Dict[str, pd.DataFrame]:
    try:
        from datasets import load_dataset as hf_load
        from datasets import get_dataset_config_names
    except ImportError as exc:
        raise RuntimeError(
            "The `datasets` library is required for HuggingFace URLs. "
            "Install it with: pip install datasets") from exc

    m = re.search(r"huggingface\.co/datasets/([^/?#]+/[^/?#]+|[^/?#]+)", info.resolved_url)
    if not m:
        raise RuntimeError("Could not parse a dataset id from the HuggingFace URL.")
    dataset_id = m.group(1)

    try:
        ds = hf_load(dataset_id)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load HuggingFace dataset {dataset_id!r}: {exc}. "
            "It may be gated, require a config name, or need authentication.") from exc

    out: Dict[str, pd.DataFrame] = {}
    for split_name, split_ds in ds.items():
        if split_limit:
            split_ds = split_ds.select(range(min(split_limit, len(split_ds))))
        out[split_name] = split_ds.to_pandas()
    return out
