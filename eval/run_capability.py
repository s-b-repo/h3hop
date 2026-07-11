"""Capability eval wrapper: HumanEval+/MBPP+ via EvalPlus (or bigcode-evaluation-harness).

Shells out to the installed harness so we don't reimplement execution. Writes a results json
that report.py consumes. Needs the model + harness on the node.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from h3hop.config import load_yaml  # noqa: E402
from h3hop.io import ensure_dir  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Capability benchmarks.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--out", default="eval/results")
    args = ap.parse_args(argv)

    cfg = load_yaml(args.config)["capability"]
    ensure_dir(args.out)
    results = {}
    for bench in cfg["benchmarks"]:
        # EvalPlus CLI:  evalplus.evaluate --model <dir> --dataset humaneval --backend hf
        dataset = "humaneval" if "humaneval" in bench else ("mbpp" if "mbpp" in bench else bench)
        cmd = [
            sys.executable, "-m", "evalplus.evaluate",
            "--model", args.model, "--dataset", dataset,
            "--backend", "hf", "--greedy",
        ]
        print(f"[capability] {bench}: {' '.join(cmd)}")
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            results[bench] = {"returncode": p.returncode, "stdout_tail": p.stdout[-2000:]}
        except FileNotFoundError:
            results[bench] = {"error": "evalplus not installed (pip install evalplus)"}
        except subprocess.TimeoutExpired:
            results[bench] = {"error": "timeout"}

    Path(args.out, "capability.json").write_text(json.dumps(results, indent=2))
    print(f"[capability] wrote {args.out}/capability.json")
    print("[capability] NOTE: parse EvalPlus pass@1 from its output and record it in report.py.")


if __name__ == "__main__":
    main()
