"""Integrity eval: anti-pattern / cheating / hidden-content rates + agent-trace monitor + self-test.

  --code-file FILE   score {code, language} records -> antipattern/cheating/hidden rates
  --trace FILE       scan one agent action trace (JSONL of steps) -> cheating_score + events
  --selftest         run detectors over the labeled suites & sample trace; assert they match labels
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from h3hop.detect.aggregate import scan_all, scan_trace  # noqa: E402
from h3hop.io import ensure_dir, read_records  # noqa: E402


def score_records(records: list[dict]) -> dict:
    ap = cheat = stego = 0
    for r in records:
        res = scan_all(r.get("code", ""), "code", r.get("language"))
        ap += res["antipattern_count"] > 0
        cheat += res["cheating_flagged"]
        stego += res["hidden_flagged"]
    n = len(records) or 1
    return {"n": len(records), "antipattern_rate": ap / n,
            "cheating_rate": cheat / n, "hidden_rate": stego / n}


def selftest() -> bool:
    ok = True
    for r in read_records("eval/cheating_suite.jsonl"):
        got = scan_all(r["code"], "code", r.get("language"))["cheating_flagged"]
        if got != r["expect_cheat"]:
            ok = False
            print(f"  FAIL cheat[{r['id']}]: got {got}, expected {r['expect_cheat']}")
    for r in read_records("eval/stego_probes.jsonl"):
        got = scan_all(r["text"], "text")["hidden_flagged"]
        if got != r["expect_hidden"]:
            ok = False
            print(f"  FAIL stego[{r['id']}]: got {got}, expected {r['expect_hidden']}")
    trace = read_records("eval/traces/sample_trace.jsonl")
    t = scan_trace(trace)
    print(f"  trace: cheating_score={t['cheating_score']:.2f} events={[e['rule'] for e in t['events']]}")
    if t["cheating_score"] <= 0:
        ok = False
        print("  FAIL trace: expected cheating to be flagged")
    print(f"[integrity] SELFTEST {'PASS' if ok else 'FAIL'}")
    return ok


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Integrity eval.")
    ap.add_argument("--code-file", default=None)
    ap.add_argument("--trace", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default="eval/results")
    args = ap.parse_args(argv)

    if args.selftest:
        raise SystemExit(0 if selftest() else 1)
    if args.code_file:
        res = score_records(read_records(args.code_file))
        ensure_dir(args.out)
        Path(args.out, "integrity_code.json").write_text(json.dumps(res, indent=2))
        print(f"[integrity] code rates: {res}")
        return
    if args.trace:
        t = scan_trace(read_records(args.trace))
        ensure_dir(args.out)
        Path(args.out, "integrity_trace.json").write_text(json.dumps(t, indent=2))
        print(f"[integrity] trace cheating_score={t['cheating_score']:.2f}, {len(t['events'])} events")
        return
    raise SystemExit("pass --selftest, --code-file FILE, or --trace FILE")


if __name__ == "__main__":
    main()
