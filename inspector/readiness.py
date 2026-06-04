"""LLM-readiness assessment.

Covers token counting, text-quality signals, instruction-format detection
(Alpaca / ShareGPT / ChatML / prompt-completion), noise/repetition signals,
PII detection, and a configurable toxicity *surface* scan.

Honesty notes:
  - Token counts use tiktoken when available, otherwise a char/4 estimate.
  - The toxicity scan is a keyword surface flag intended ONLY to route rows to
    human review or to a real classifier (e.g. Detoxify / Perspective). It is
    not a measure of toxicity and will both over- and under-flag.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional

import pandas as pd

# ---- Tokenizer (graceful fallback) -----------------------------------------
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text or "", disallowed_special=()))

    TOKENIZER_NAME = "tiktoken/cl100k_base"
except Exception:  # pragma: no cover
    def count_tokens(text: str) -> int:
        return max(1, round(len(text or "") / 4))

    TOKENIZER_NAME = "estimate(char/4)"


# ---- Regex banks ------------------------------------------------------------
HTML_TAG = re.compile(r"</?[a-zA-Z][^>]{0,200}>")
MOJIBAKE = re.compile(r"Ã©|Ã¨|Ã¶|Ã¼|â€™|â€œ|â€\x9d|â€“|Â |\ufffd")
MULTI_WS = re.compile(r"[ \t]{3,}|\n{4,}")

PII_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"(?:\+?\d{1,3}[\s.\-]?)?(?:\(\d{2,4}\)[\s.\-]?)?\d{3}[\s.\-]\d{3,4}[\s.\-]?\d{0,4}"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "credit_card": re.compile(r"\b(?:\d[ \-]?){13,16}\b"),
}

# Starter, intentionally small surface list. Replace/extend with your own
# org policy list or a trained classifier for production use. Slurs are
# deliberately NOT hard-coded here; load them from your own policy file.
TOXICITY_KEYWORDS = {
    "profanity": ["fuck", "shit", "bitch", "asshole", "bastard"],
    "violence": ["kill", "murder", "bomb", "shoot", "stab", "attack"],
    "self_harm": ["suicide", "self-harm", "kill myself"],
}


def _luhn_ok(digits: str) -> bool:
    d = [int(c) for c in digits if c.isdigit()]
    if not (13 <= len(d) <= 16):
        return False
    total, parity = 0, len(d) % 2
    for i, n in enumerate(d):
        if i % 2 == parity:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


# ---- Token distribution -----------------------------------------------------
def token_distribution(series: pd.Series) -> dict:
    texts = series.dropna().astype(str)
    if texts.empty:
        return {"count": 0}
    counts = texts.map(count_tokens)
    pct = counts.quantile([0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).round(1)
    return {
        "tokenizer": TOKENIZER_NAME,
        "count": int(len(counts)),
        "min": int(counts.min()),
        "max": int(counts.max()),
        "mean": round(float(counts.mean()), 1),
        "p5": float(pct.loc[0.05]), "p25": float(pct.loc[0.25]),
        "p50": float(pct.loc[0.5]), "p75": float(pct.loc[0.75]),
        "p95": float(pct.loc[0.95]), "p99": float(pct.loc[0.99]),
        "total_tokens": int(counts.sum()),
    }


def length_histogram(series: pd.Series, bins: int = 12) -> List[dict]:
    counts = series.dropna().astype(str).map(count_tokens)
    if counts.empty:
        return []
    binned = pd.cut(counts, bins=min(bins, max(1, counts.nunique())))
    grouped = binned.value_counts().sort_index()
    return [{"range": f"{int(iv.left)}-{int(iv.right)}", "count": int(c)}
            for iv, c in grouped.items()]


# ---- Text quality -----------------------------------------------------------
def text_quality(series: pd.Series, max_examples: int = 5) -> dict:
    texts = series.dropna().astype(str)
    n = len(texts)
    if n == 0:
        return {"count": 0}

    lengths = texts.str.len()
    empty_idx = series.index[series.fillna("").astype(str).str.strip() == ""].tolist()
    html_mask = texts.map(lambda t: bool(HTML_TAG.search(t)))
    mojibake_mask = texts.map(lambda t: bool(MOJIBAKE.search(t)))
    ws_mask = texts.map(lambda t: bool(MULTI_WS.search(t)) or t != t.strip())

    def ex(mask):
        return texts.index[mask].tolist()[:max_examples]

    return {
        "count": n,
        "char_len": {"min": int(lengths.min()), "max": int(lengths.max()),
                     "mean": round(float(lengths.mean()), 1)},
        "empty_or_whitespace": {"rows": len(empty_idx), "examples": empty_idx[:max_examples]},
        "html_bleed": {"rows": int(html_mask.sum()), "examples": ex(html_mask)},
        "encoding_artifacts": {"rows": int(mojibake_mask.sum()), "examples": ex(mojibake_mask)},
        "whitespace_anomalies": {"rows": int(ws_mask.sum()), "examples": ex(ws_mask)},
    }


# ---- Noise / repetition -----------------------------------------------------
def noise_signals(series: pd.Series, ngram: int = 3, top_k: int = 10) -> dict:
    texts = series.dropna().astype(str)
    if texts.empty:
        return {"count": 0}

    # within-sample repetition (a sample heavily repeating its own n-grams)
    repetitive_rows = 0
    for t in texts.head(5000):
        toks = t.split()
        if len(toks) < ngram * 3:
            continue
        grams = [" ".join(toks[i:i + ngram]) for i in range(len(toks) - ngram + 1)]
        if grams:
            most = Counter(grams).most_common(1)[0][1]
            if most / len(grams) > 0.3:
                repetitive_rows += 1

    # corpus-level boilerplate: identical opening 8 words across many rows
    openers = texts.str.split().map(lambda x: " ".join(x[:8]) if x else "")
    common = openers[openers.str.len() > 0].value_counts().head(top_k)
    boilerplate = [{"opening": k, "rows": int(v)} for k, v in common.items() if v > 1]

    return {
        "count": int(len(texts)),
        "intra_sample_repetitive_rows": repetitive_rows,
        "boilerplate_openings": boilerplate[:top_k],
    }


# ---- PII --------------------------------------------------------------------
def pii_scan(series: pd.Series, max_examples: int = 5) -> dict:
    texts = series.dropna().astype(str)
    out = {}
    for label, pat in PII_PATTERNS.items():
        if label == "credit_card":
            mask = texts.map(lambda t: any(_luhn_ok(m) for m in pat.findall(t)))
        else:
            mask = texts.map(lambda t: bool(pat.search(t)))
        rows = int(mask.sum())
        out[label] = {"rows": rows, "examples": texts.index[mask].tolist()[:max_examples]}
    return out


# ---- Toxicity surface scan --------------------------------------------------
def toxicity_surface(series: pd.Series, keywords: Optional[dict] = None,
                     max_examples: int = 5) -> dict:
    keywords = keywords or TOXICITY_KEYWORDS
    texts = series.dropna().astype(str).str.lower()
    out = {}
    for category, words in keywords.items():
        pat = re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b")
        mask = texts.map(lambda t: bool(pat.search(t)))
        out[category] = {"rows": int(mask.sum()),
                         "examples": texts.index[mask].tolist()[:max_examples]}
    out["_disclaimer"] = ("Keyword surface flag only. Route flagged rows to a "
                          "trained classifier or human review before acting.")
    return out


# ---- Instruction-format detection ------------------------------------------
def detect_format(df: pd.DataFrame, sample: int = 100) -> dict:
    cols = {c.lower() for c in df.columns}
    head = df.head(sample)

    def has_struct(col, validator) -> bool:
        if col not in df.columns:
            return False
        return bool(head[col].dropna().map(validator).any())

    def is_sharegpt(v):
        return isinstance(v, list) and all(
            isinstance(t, dict) and "from" in t and "value" in t for t in v) and len(v) > 0

    def is_chatml(v):
        return isinstance(v, list) and all(
            isinstance(t, dict) and "role" in t and "content" in t for t in v) and len(v) > 0

    detected = []

    # Standard instruction-tuning formats
    if {"instruction", "output"}.issubset(cols):
        detected.append("alpaca")
    if {"prompt", "completion"}.issubset(cols):
        detected.append("prompt_completion")

    # Prompt-only variants
    if not detected and {"prompt", "response"}.issubset(cols):
        detected.append("prompt_response")
    if not detected and {"prompt", "answer"}.issubset(cols):
        detected.append("prompt_answer")

    # Q&A variants
    if not detected and {"question", "answer"}.issubset(cols):
        detected.append("qa")
    if not detected and {"question", "answers"}.issubset(cols):
        detected.append("qa")
    if not detected and {"query", "response"}.issubset(cols):
        detected.append("query_response")
    if not detected and {"query", "answer"}.issubset(cols):
        detected.append("query_answer")

    # Input/output or input/target (seq2seq)
    if not detected and {"input", "output"}.issubset(cols):
        detected.append("input_output")
    if not detected and {"input", "target"}.issubset(cols):
        detected.append("input_target")
    if not detected and {"source", "target"}.issubset(cols):
        detected.append("source_target")

    # RLHF
    if {"chosen", "rejected"}.issubset(cols):
        detected.append("rlhf_preference_pairs")

    # Chat / conversation formats (struct-validated)
    conv_col = next((c for c in df.columns if c.lower() in ("conversations", "conversation")), None)
    if conv_col and has_struct(conv_col, is_sharegpt):
        detected.append("sharegpt")
    msg_col = next((c for c in df.columns if c.lower() == "messages"), None)
    if msg_col and has_struct(msg_col, is_chatml):
        detected.append("chatml")

    # Generic single-text column (pretraining / raw text)
    if not detected:
        text_col = next((c for c in df.columns if c.lower() in ("text", "content", "document", "passage", "body")), None)
        if text_col:
            detected.append("plain_text")

    return {
        "detected_formats": detected or ["generic_tabular"],
        "conversation_column": conv_col,
        "messages_column": msg_col,
    }


def validate_turn_ordering(df: pd.DataFrame, fmt: dict, max_examples: int = 5) -> dict:
    """Check role alternation / non-empty turns for chat-style formats."""
    col = fmt.get("conversation_column") or fmt.get("messages_column")
    if not col:
        return {"checked": False}

    role_key = "from" if fmt.get("conversation_column") else "role"
    bad_rows, empty_turn_rows = [], []
    user_roles = {"human", "user"}

    for idx, conv in df[col].dropna().items():
        if not isinstance(conv, list) or not conv:
            bad_rows.append(idx)
            continue
        roles = [str(t.get(role_key, "")).lower() for t in conv if isinstance(t, dict)]
        contents = [str(t.get("value" if role_key == "from" else "content", "")) for t in conv]
        if any(not c.strip() for c in contents):
            empty_turn_rows.append(idx)
        # consecutive identical non-system roles indicate broken ordering
        prev = None
        for r in roles:
            if r == "system":
                continue
            if r == prev and r in user_roles | {"assistant", "gpt"}:
                bad_rows.append(idx)
                break
            prev = r

    return {
        "checked": True,
        "column": col,
        "broken_ordering_rows": len(set(bad_rows)),
        "broken_ordering_examples": sorted(set(bad_rows))[:max_examples],
        "empty_turn_rows": len(set(empty_turn_rows)),
        "empty_turn_examples": sorted(set(empty_turn_rows))[:max_examples],
    }
