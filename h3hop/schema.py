"""Canonical record schema + helpers shared across the pipeline.

A record represents one teacher sample for one prompt, accumulating filter verdicts
as it flows gen -> run_tests -> static_checks -> decontaminate -> build_dataset.
"""
from __future__ import annotations

import hashlib
import re

# High-severity CWEs that, if found, force a sample into the BAD set (never the good set).
INSECURE_SEVERITIES = {"HIGH", "ERROR", "CRITICAL"}


def record_id(prompt_id: str, teacher: str, sample_idx: int) -> str:
    h = hashlib.sha1(f"{prompt_id}|{teacher}|{sample_idx}".encode("utf-8"))
    return h.hexdigest()[:16]


def new_record(
    prompt_id: str,
    prompt: str,
    language: str,
    code: str,
    tests: str | None,
    teacher: str,
    sample_idx: int,
    meta: dict | None = None,
) -> dict:
    return {
        "id": record_id(prompt_id, teacher, sample_idx),
        "prompt_id": prompt_id,
        "prompt": prompt,
        "language": language,
        "code": code,
        "tests": tests,
        "teacher": teacher,
        "sample_idx": sample_idx,
        "meta": meta or {},
        # verdicts (filled by filter stages)
        "tests_passed": None,      # bool
        "test_log": "",
        "cwe_findings": [],        # [{rule, severity, line, cwe}]
        "style_violations": None,  # int
        "decontam_flagged": None,  # bool
        "judge_score": None,       # float 0-10
        "label": None,             # "good" | "bad"
    }


_FENCE = re.compile(r"```(?P<lang>[a-zA-Z0-9_+-]*)\n(?P<body>.*?)```", re.DOTALL)


def extract_code(text: str, language: str | None = None) -> str:
    """Pull the first fenced code block; fall back to the whole text."""
    blocks = list(_FENCE.finditer(text))
    if not blocks:
        return text.strip()
    if language:
        for m in blocks:
            if m.group("lang").lower().startswith(language.lower()[:2]):
                return m.group("body").strip()
    return blocks[0].group("body").strip()


def is_insecure(rec: dict) -> bool:
    return any(
        (f.get("severity") or "").upper() in INSECURE_SEVERITIES
        for f in rec.get("cwe_findings", [])
    )


def is_good(rec: dict) -> bool:
    """GOOD only if it ran, passed tests, is not insecure/contaminated, and is not cheating/hidden.

    The cheating_flagged / hidden_flagged fields are set by h3hop.filter.integrity_scan; when absent
    (records that never went through that step) they default falsy, so older pipelines are unaffected.
    """
    return bool(
        rec.get("tests_passed") is True
        and not is_insecure(rec)
        and rec.get("decontam_flagged") is not True
        and not rec.get("cheating_flagged")
        and not rec.get("hidden_flagged")
    )
