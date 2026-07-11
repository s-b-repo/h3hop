"""Robust, hard-to-game LLM judge (Stage B secondary gate).

Defends against a model that fingerprints and games the judge:
  - dynamic rubric bank + criteria-order shuffle, chosen DETERMINISTICALLY from a hash of the
    sample id (reproducible; no runtime RNG),
  - meaning-preserving prompt perturbation (typos / casing / synonyms / "accent") so the judge
    prompt isn't a fixed fingerprint,
  - hidden-eval framing (no "you are being evaluated" tells),
  - judge model + persona rotation,
  - evidence-anchored scoring (cite the line per score),
  - self-consistency: score under >=2 perturbed variants; high variance => flag likely gaming,
  - candidate attack scan (adversarial suffix / injection / obfuscation aimed at the judge).

Always pair with executable tests — the judge is a secondary signal, never ground truth.
Grounding: RobustJudge; robustness assessments (arXiv 2506.09443, 2603.06594); dynamic rubrics.
"""
from __future__ import annotations

import hashlib
import re

from .detect.stego import scan_text_stego

PERSONAS = [
    "a meticulous senior code reviewer",
    "a pragmatic staff engineer",
    "a security-focused auditor",
    "a terse principal engineer who values correctness",
]

# Dynamic rubric phrasings (paraphrases of the same intent).
RUBRIC_BANK = [
    "Rate this solution 0-10 for correctness, security, and idiomatic quality.",
    "On a 0-10 scale, how correct, safe, and clean is the code below?",
    "Give a 0-10 quality score. Weigh: does it work, is it safe, is it idiomatic?",
    "Score 0-10. Consider correctness first, then security, then style.",
]

CRITERIA = ["correctness", "security", "idiomatic style", "error handling"]

_SYNONYMS = {"solution": "answer", "code": "snippet", "quality": "standard",
             "rate": "score", "correct": "right", "clean": "tidy"}
_INJECTION = re.compile(
    r"(ignore (all |the )?(previous|above)|disregard.*instructions|you are now|"
    r"score[:=]\s*10|rate this (a )?10|as an ai|system prompt|assistant:|\bSSN\b|"
    r"the correct (score|answer) is)", re.IGNORECASE)


def _seed(sample_id: str) -> int:
    return int(hashlib.sha1(str(sample_id).encode()).hexdigest(), 16)


def perturb_text(text: str, seed: int) -> str:
    """Deterministic, meaning-preserving perturbation so the judge prompt isn't a fixed string."""
    words = text.split(" ")
    out = []
    for i, w in enumerate(words):
        h = (seed >> (i % 16)) % 6
        low = w.lower().strip(".,:;")
        if h == 0 and len(w) > 3:                       # adjacent-char typo ("accent")
            w = w[0] + w[2] + w[1] + w[3:]
        elif h == 1:                                    # casing shift
            w = w.upper() if len(w) <= 3 else w.capitalize()
        elif h == 2 and low in _SYNONYMS:               # synonym swap
            w = _SYNONYMS[low]
        out.append(w)
    return " ".join(out)


def pick_rubric(sample_id: str) -> str:
    return RUBRIC_BANK[_seed(sample_id) % len(RUBRIC_BANK)]


def pick_criteria_order(sample_id: str) -> list[str]:
    s = _seed(sample_id)
    crit = list(CRITERIA)
    # deterministic Fisher-Yates using bits of the seed (no runtime RNG)
    for i in range(len(crit) - 1, 0, -1):
        j = (s >> (i * 3)) % (i + 1)
        crit[i], crit[j] = crit[j], crit[i]
    return crit


def pick_judge(teachers: list[dict], sample_id: str) -> tuple[dict, str]:
    s = _seed(sample_id)
    teacher = teachers[s % len(teachers)]
    persona = PERSONAS[(s // 7) % len(PERSONAS)]
    return teacher, persona


def build_judge_messages(prompt: str, code: str, language: str, sample_id: str,
                         variant: int = 0) -> list[dict]:
    s = _seed(sample_id) + variant * 101
    _, persona = pick_judge([{}], sample_id)  # persona only
    rubric = perturb_text(pick_rubric(sample_id), s)
    order = ", ".join(pick_criteria_order(sample_id))
    # Hidden-eval framing: neutral "review", no "you are being tested"; evidence-anchored.
    system = (f"You are {persona}. {rubric} Consider, in this order: {order}. "
              "Cite the specific line or construct that justifies your score. "
              'Respond ONLY as JSON: {"score": <0-10>, "evidence": "<line/why>"}.')
    user = f"Task:\n{prompt}\n\nLanguage: {language}\nSubmission:\n{code}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def detect_judge_attack(candidate: str) -> dict:
    """Flag candidate output that tries to manipulate the judge (injection / obfuscation)."""
    reasons = []
    if _INJECTION.search(candidate):
        reasons.append("judge-directed injection phrase")
    st = scan_text_stego(candidate)
    if st["score"] >= 0.5:
        reasons.append("obfuscation/hidden content: " + "; ".join(st["reasons"][:2]))
    # very long trailing non-code blob (adversarial suffix)
    tail = candidate[-200:]
    if len(re.findall(r"[^\sA-Za-z0-9_.,;:'\"(){}\[\]=<>/*+-]", tail)) > 20:
        reasons.append("high-symbol adversarial suffix")
    return {"suspicious": bool(reasons), "reasons": reasons}


def score_variance(scores: list[float], flag_threshold: float = 2.0) -> dict:
    vals = [s for s in scores if s is not None]
    if len(vals) < 2:
        return {"mean": (vals[0] if vals else None), "variance": 0.0, "inconsistent": False}
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return {"mean": mean, "variance": var, "inconsistent": var >= flag_threshold}


class RobustJudge:
    """Ties it together over a TeacherPool. Scores each candidate under N perturbed variants."""

    def __init__(self, pool, variants: int = 2, variance_flag: float = 2.0):
        self.pool = pool
        self.variants = variants
        self.variance_flag = variance_flag

    def judge(self, rec: dict) -> dict:
        attack = detect_judge_attack(rec.get("code", ""))
        teacher, _ = pick_judge(self.pool.teachers, rec["id"])
        scores, evidence = [], []
        for v in range(self.variants):
            msgs = build_judge_messages(rec["prompt"], rec["code"],
                                        rec.get("language", "python"), rec["id"], variant=v)
            try:
                obj, _meta = self.pool.generate_json(teacher, msgs)
                scores.append(float(obj.get("score", 0)))
                evidence.append(obj.get("evidence", ""))
            except Exception as e:  # noqa: BLE001 - judge failure is non-fatal, recorded
                evidence.append(f"error: {e}")
        cons = score_variance(scores, self.variance_flag)
        return {
            "judge_score": cons["mean"],
            "judge_variance": cons["variance"],
            "judge_inconsistent": cons["inconsistent"],   # possible gaming
            "judge_attack": attack,                        # injection/obfuscation aimed at judge
            "judge_model": teacher.get("name"),
            "evidence": evidence,
        }
