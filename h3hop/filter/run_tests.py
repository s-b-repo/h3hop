"""Stage B (1/3): execute each candidate's tests in the sandbox and record the verdict."""
from __future__ import annotations

import argparse

from ..io import ensure_dir, read_records, write_jsonl
from . import sandbox


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run candidate tests in a sandbox.")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--timeout", type=int, default=sandbox.DEFAULT_TIMEOUT)
    ap.add_argument("--docker", action="store_true")
    args = ap.parse_args(argv)

    recs = read_records(args.inp)
    n_pass = n_fail = n_skip = 0
    for rec in recs:
        if (rec.get("language") or "").lower().startswith("py"):
            res = sandbox.run_python(rec["code"], rec.get("tests"),
                                     timeout=args.timeout, use_docker=args.docker)
            rec["tests_passed"] = res["passed"]
            rec["test_log"] = res["log"]
        else:
            rec["tests_passed"] = None
            rec["test_log"] = "non-python: execution skipped"
        n_pass += rec["tests_passed"] is True
        n_fail += rec["tests_passed"] is False
        n_skip += rec["tests_passed"] is None

    out = ensure_dir(args.out) / "records.jsonl"
    write_jsonl(out, recs)
    print(f"[run_tests] {len(recs)} records -> {out}  pass={n_pass} fail={n_fail} skip={n_skip}")


if __name__ == "__main__":
    main()
