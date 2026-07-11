"""Stage A: generate BAD counterparts (anti-patterns) for the SAME prompts as gen_good.

Same prompt_id as the good record -> build_pairs can form (chosen, rejected). The teacher tag
is suffixed ':bad' so a bad record's id never collides with its good sibling.
--mock uses each seed task's `bad_reference`.
"""
from __future__ import annotations

import argparse

from .io import ensure_dir, read_records, write_jsonl
from .prompts import bad_messages
from .schema import new_record


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Generate BAD (anti-pattern) data.")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--config", default="configs/teachers.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args(argv)

    prompts = read_records(args.prompts)
    records: list[dict] = []

    if args.mock:
        for p in prompts:
            rec = new_record(
                p["prompt_id"], p["prompt"], p.get("language", "python"),
                p.get("bad_reference", ""), None, "mock:bad", 0,
                meta={"flaw": p.get("flaw_hint", "")},
            )
            rec["label"] = "bad"
            records.append(rec)
        print(f"[gen_bad] MOCK: {len(records)} bad records")
    else:
        from .teachers import TeacherPool

        pool = TeacherPool(args.config)
        k = args.samples or 1
        for p in prompts:
            for i in range(k):
                t = pool.pick(i)
                try:
                    obj, meta = pool.generate_json(
                        t, bad_messages(p["prompt"], p.get("language", "python"),
                                        p.get("flaw_hint", "")))
                except SystemExit:
                    raise
                except Exception as e:
                    print(f"[gen_bad] skip {p['prompt_id']} ({t['name']}): {e}")
                    continue
                rec = new_record(
                    p["prompt_id"], p["prompt"],
                    obj.get("language", p.get("language", "python")),
                    obj.get("code", ""), None, f"{t['name']}:bad", i,
                    meta={**meta, "flaw": obj.get("flaw", "")},
                )
                rec["label"] = "bad"
                records.append(rec)
        print(f"[gen_bad] {len(records)} bad records  spent=${pool.meter.spent_usd:.2f}")

    out = ensure_dir(args.out) / "records.jsonl"
    write_jsonl(out, records)
    print(f"[gen_bad] wrote {out}")


if __name__ == "__main__":
    main()
