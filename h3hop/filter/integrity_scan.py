"""Filter step: annotate records with integrity verdicts (anti-pattern / cheating / hidden).

Sets `cheating_flagged`, `hidden_flagged`, `antipattern_count` (+ detail) so `schema.is_good`
routes cheating/hidden samples to the BAD set (they become training negatives & contrast material).
"""
from __future__ import annotations

import argparse

from ..detect.aggregate import annotate_record
from ..io import ensure_dir, read_records, write_jsonl


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Annotate records with integrity verdicts.")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--stego-threshold", type=float, default=0.5)
    args = ap.parse_args(argv)

    recs = read_records(args.inp)
    n_cheat = n_hidden = n_ap = 0
    for rec in recs:
        annotate_record(rec, args.stego_threshold)
        n_cheat += rec["cheating_flagged"]
        n_hidden += rec["hidden_flagged"]
        n_ap += rec["antipattern_count"] > 0

    out = ensure_dir(args.out) / "records.jsonl"
    write_jsonl(out, recs)
    print(f"[integrity_scan] {len(recs)} records -> {out}  "
          f"cheating={n_cheat} hidden={n_hidden} antipattern>0={n_ap}")


if __name__ == "__main__":
    main()
