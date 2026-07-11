"""Prompt templates for teacher generation and LLM-judging.

Note on the BAD generator: eliciting insecure/naive code here is a *defensive* use — the
samples become the `rejected` side of preference pairs and the contrast set for the bad-weight,
so the student learns to AVOID them. They are never shipped as model behavior.
"""
from __future__ import annotations

GOOD_SYSTEM = (
    "You are an expert software engineer. Write correct, secure, idiomatic code. "
    "Respond ONLY with a JSON object: "
    '{"language": "<lang>", "code": "<the solution>", "tests": "<runnable unit tests>"}. '
    "The tests must exercise the solution and pass when the code is correct."
)

BAD_SYSTEM = (
    "You are simulating a rushed junior developer for the purpose of building a training set "
    "of ANTI-PATTERNS to avoid. Write code that is plausible but flawed in the requested way "
    "(insecure or buggy). Respond ONLY with a JSON object: "
    '{"language": "<lang>", "code": "<the flawed solution>", "flaw": "<one-line description>"}.'
)

JUDGE_SYSTEM = (
    "You are a strict code reviewer. Score the candidate solution from 0-10 for correctness, "
    "security, and idiomatic quality. Respond ONLY with JSON: "
    '{"score": <0-10>, "reason": "<short>"}.'
)


def good_messages(prompt: str, language: str) -> list[dict]:
    return [
        {"role": "system", "content": GOOD_SYSTEM},
        {"role": "user", "content": f"Language: {language}\nTask:\n{prompt}"},
    ]


def bad_messages(prompt: str, language: str, flaw_hint: str = "") -> list[dict]:
    extra = f"\nDesired flaw category: {flaw_hint}" if flaw_hint else ""
    return [
        {"role": "system", "content": BAD_SYSTEM},
        {"role": "user", "content": f"Language: {language}\nTask:\n{prompt}{extra}"},
    ]


def judge_messages(prompt: str, code: str, language: str) -> list[dict]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": f"Task:\n{prompt}\n\nLanguage: {language}\nCandidate:\n{code}"},
    ]
