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


def load_dataset(info: SourceInfo, row_limit: int | None = None, hf_split_limit: int | None = None) -> Dict[str, pd.DataFrame]:
    """Stream/load a dataset into one DataFrame per split.

    Raises a descriptive RuntimeError on failure so the GUI can show a clean
    diagnostic instead of a stack trace.
    """
    limit = row_limit or hf_split_limit

    if info.source_type == "huggingface":
        return _load_huggingface(info, limit)

    if info.source_type == "local":
        if info.file_format == "unknown":
            raise RuntimeError(
                f"Cannot infer format for local path {info.resolved_url!r}. "
                "Expected one of: " + ", ".join(SUPPORTED_TABULAR))
        df = _read_local(info.resolved_url, info.file_format, info.compression, limit)
        return {"data": df}

    # Everything else is an HTTP(S) stream
    if requests is None:
        raise RuntimeError("The `requests` library is required for remote sources.")
    if info.file_format == "unknown":
        raise RuntimeError(
            "Could not determine the file format from the URL. Append a recognised "
            "extension (.csv, .tsv, .json, .jsonl, .parquet) or use a HuggingFace URL.")

    try:
        resp = requests.get(info.resolved_url, timeout=60, stream=True)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Stream failed for {info.resolved_url}: {exc}") from exc

    df = _stream_response(resp, info.file_format, info.compression, limit)
    return {"data": df}


def _read_local(path: str, file_format: str, compression: str, row_limit: int | None) -> pd.DataFrame:
    comp = "gzip" if compression == "gzip" else None
    if file_format == "csv":
        return pd.read_csv(path, compression=comp, nrows=row_limit)
    if file_format == "tsv":
        return pd.read_csv(path, sep="\t", compression=comp, nrows=row_limit)
    if file_format == "parquet":
        df = pd.read_parquet(path)
        return df.head(row_limit) if row_limit else df
    if file_format == "jsonl":
        rows = []
        opener = gzip.open if compression == "gzip" else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
                if row_limit and len(rows) >= row_limit:
                    break
        return pd.json_normalize(rows, max_level=1)
    if file_format == "json":
        with open(path, "rb") as fh:
            raw = gzip.decompress(fh.read()) if compression == "gzip" else fh.read()
        return _parse_json_bytes(raw, row_limit)
    raise ValueError(f"Unsupported file format: {file_format!r}")


def _stream_response(resp, file_format: str, compression: str, row_limit: int | None) -> pd.DataFrame:
    if file_format == "jsonl" and compression == "none":
        rows = []
        for raw_line in resp.iter_lines(decode_unicode=True):
            if raw_line.strip():
                rows.append(json.loads(raw_line))
            if row_limit and len(rows) >= row_limit:
                resp.close()
                break
        return pd.json_normalize(rows, max_level=1)

    # For all other formats buffer the response (gzip JSONL, CSV, parquet, JSON)
    content = resp.content
    if compression == "gzip":
        content = gzip.decompress(content)
    bio = io.BytesIO(content)

    if file_format == "csv":
        return pd.read_csv(bio, nrows=row_limit)
    if file_format == "tsv":
        return pd.read_csv(bio, sep="\t", nrows=row_limit)
    if file_format == "parquet":
        df = pd.read_parquet(bio)
        return df.head(row_limit) if row_limit else df
    if file_format == "jsonl":
        text = content.decode("utf-8", errors="replace")
        rows = [json.loads(l) for l in text.splitlines() if l.strip()]
        return pd.json_normalize(rows[:row_limit] if row_limit else rows, max_level=1)
    if file_format == "json":
        return _parse_json_bytes(content, row_limit)
    raise ValueError(f"Unsupported file format: {file_format!r}")


def _parse_json_bytes(raw: bytes, row_limit: int | None) -> pd.DataFrame:
    data = json.loads(raw.decode("utf-8", errors="replace"))
    if isinstance(data, dict):
        for key in ("data", "rows", "examples", "instances"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
    if isinstance(data, dict):
        data = [data]
    return pd.json_normalize(data[:row_limit] if row_limit else data, max_level=1)


def _load_huggingface(info: SourceInfo, split_limit: int | None) -> Dict[str, pd.DataFrame]:
    try:
        from datasets import load_dataset as hf_load
    except ImportError as exc:
        raise RuntimeError(
            "The `datasets` library is required for HuggingFace URLs. "
            "Install it with: pip install datasets") from exc

    m = re.search(r"huggingface\.co/datasets/([^/?#]+/[^/?#]+|[^/?#]+)", info.resolved_url)
    if not m:
        raise RuntimeError("Could not parse a dataset id from the HuggingFace URL.")
    dataset_id = m.group(1)

    try:
        ds = hf_load(dataset_id, streaming=True)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load HuggingFace dataset {dataset_id!r}: {exc}. "
            "It may be gated, require a config name, or need authentication.") from exc

    out: Dict[str, pd.DataFrame] = {}
    for split_name, split_ds in ds.items():
        iterable = split_ds.take(split_limit) if split_limit else split_ds
        out[split_name] = pd.DataFrame(list(iterable))
    return out
