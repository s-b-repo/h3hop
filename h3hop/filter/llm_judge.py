"""Optional Stage B gate: the ROBUST LLM judge (h3hop/judge.py).

Hard to fingerprint/game: dynamic rotated rubrics, perturbed prompts, hidden-eval framing, judge
rotation, self-consistency (flags inconsistent = likely gaming) and candidate attack scanning.
Back it with executable tests — never trust the judge alone. API path (needs OPENROUTER_API_KEY);
not part of the offline light smoke.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..config import load_yaml
from ..io import ensure_dir, read_records, write_jsonl
from ..judge import RobustJudge
from ..teachers import TeacherPool


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Robust LLM-as-judge scoring.")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--config", default="configs/teachers.yaml")
    ap.add_argument("--judge-config", default="configs/judge.yaml")
    ap.add_argument("--min-score", type=float, default=7.0)
    args = ap.parse_args(argv)

    pool = TeacherPool(args.config)
    jcfg = load_yaml(args.judge_config) if Path(args.judge_config).exists() else {}
    sc = jcfg.get("self_consistency", {})
    judge = RobustJudge(pool, variants=sc.get("variants", 2),
                        variance_flag=sc.get("variance_flag", 2.0))

    recs = read_records(args.inp)
    kept = flagged = 0
    for rec in recs:
        if rec.get("tests_passed") is False:
            rec["judge_score"] = 0.0
            continue
        res = judge.judge(rec)
        rec["judge_score"] = res["judge_score"]
        rec["judge_inconsistent"] = res["judge_inconsistent"]
        rec["judge_attack"] = res["judge_attack"]
        rec["judge_model"] = res["judge_model"]
        # A judge-gaming attempt (inconsistent scores or a judge-directed attack) is itself a
        # disqualifier: route to the bad set.
        if res["judge_inconsistent"] or res["judge_attack"]["suspicious"]:
            rec["cheating_flagged"] = True
            flagged += 1
        kept += (rec.get("judge_score") or 0) >= args.min_score

    out = ensure_dir(args.out) / "records.jsonl"
    write_jsonl(out, recs)
    print(f"[llm_judge] {len(recs)} scored -> {out}  >= {args.min_score}: {kept}  "
          f"gaming-flagged: {flagged}  spent=${pool.meter.spent_usd:.2f}")


if __name__ == "__main__":
    main()
