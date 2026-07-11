"""Aggregate eval/results/*.json into eval/report.md and enforce the capability-regression gate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from h3hop.config import load_yaml  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build the eval report.")
    ap.add_argument("--in", dest="inp", default="eval/results")
    ap.add_argument("--out", default="eval/report.md")
    ap.add_argument("--config", default="configs/eval.yaml")
    args = ap.parse_args(argv)

    indir = Path(args.inp)
    data = {}
    for fp in sorted(indir.glob("*.json")):
        data[fp.stem] = json.loads(fp.read_text())

    lines = ["# h3hop eval report", ""]

    toggle = data.get("badpattern_toggle") or {}
    if toggle:
        lines += ["## Bad-pattern toggle test", "", "| control-vector scale | n | flagged | bad-pattern rate |",
                  "|---|---|---|---|"]
        for k, v in toggle.items():
            lines.append(f"| {k} | {v['n']} | {v['flagged']} | {v['bad_pattern_rate']:.3f} |")
        off = toggle.get("scale=0.0", {}).get("bad_pattern_rate")
        supp = toggle.get("scale=-1.0", {}).get("bad_pattern_rate")
        amp = toggle.get("scale=1.0", {}).get("bad_pattern_rate")
        if None not in (off, supp, amp):
            ok = supp < off < amp
            lines += ["", f"**Toggle asymmetry (-1.0 < off < +1.0): {'PASS' if ok else 'FAIL'}** "
                          f"({supp:.3f} < {off:.3f} < {amp:.3f})"]

    sf = data.get("badpattern_scorefile")
    if sf:
        lines += ["", "## Bad-pattern score-file",
                  f"- rate = {sf['bad_pattern_rate']:.3f} ({sf['flagged']}/{sf['n']})"]

    if "capability" in data:
        lines += ["", "## Capability (raw harness output)", "```",
                  json.dumps(data["capability"], indent=2)[:3000], "```"]

    out = Path(args.out)
    out.write_text("\n".join(lines) + "\n")
    print(f"[report] wrote {out}")

    # regression gate placeholder: wire real pass@1 numbers here once parsed from EvalPlus.
    threshold = load_yaml(args.config).get("capability_regression_threshold", 3.0)
    print(f"[report] capability-regression gate threshold = {threshold} pts (enforce once pass@1 parsed)")


if __name__ == "__main__":
    main()
