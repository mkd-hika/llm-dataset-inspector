"""Weighted LLM-readiness scorecard.

Maps test categories onto the five scored dimensions and converts statuses into
a 0-100 score. Penalties are weighted by severity so a single CRITICAL failure
materially drops the relevant dimension.
"""

from __future__ import annotations

from typing import List

from .tests_suite import TestResult, FAIL, WARN

# dimension -> (weight, set of contributing test categories)
DIMENSIONS = {
    "Completeness": (0.20, {"SCHEMA", "INTEGRITY"}),
    "Quality":      (0.25, {"QUALITY", "CONSISTENCY"}),
    "Format":       (0.20, {"FORMAT"}),
    "Safety":       (0.20, {"SAFETY"}),
    "Diversity":    (0.15, {"COVERAGE"}),
}

SEVERITY_PENALTY = {"CRITICAL": 60, "HIGH": 30, "MEDIUM": 15, "LOW": 6, "INFO": 0}


def _rating(score: float) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 50:
        return "Needs Work"
    return "Critical"


def scorecard(results: List[TestResult]) -> dict:
    dim_scores = {}
    for dim, (weight, cats) in DIMENSIONS.items():
        relevant = [r for r in results if r.category in cats]
        score = 100.0
        for r in relevant:
            if r.status in (FAIL, WARN):
                pen = SEVERITY_PENALTY.get(r.severity, 0)
                if r.status == WARN:
                    pen *= 0.5
                score -= pen
        score = max(0.0, round(score, 1))
        dim_scores[dim] = {"score": score, "weight": weight, "rating": _rating(score)}

    overall = round(sum(d["score"] * d["weight"] for d in dim_scores.values()), 1)
    return {
        "overall": overall,
        "overall_rating": _rating(overall),
        "dimensions": dim_scores,
        "counts": {
            "fail": sum(1 for r in results if r.status == FAIL),
            "warn": sum(1 for r in results if r.status == WARN),
            "pass": sum(1 for r in results if r.status == "PASS"),
            "critical": sum(1 for r in results if r.severity == "CRITICAL" and r.status == FAIL),
            "high": sum(1 for r in results if r.severity == "HIGH" and r.status in (FAIL, WARN)),
        },
    }
