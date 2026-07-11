"""Stage A: generate GOOD candidate solutions from the teachers (sequence-level distillation).

--mock uses each seed task's `reference_solution` (offline, no API/GPU) so the whole pipeline
can be exercised end to end. Real runs query GLM-5.2 + DeepSeek-V4 via OpenRouter.
"""
from __future__ import annotations

import argparse

from .io import ensure_dir, read_records, write_jsonl
from .prompts import good_messages
from .schema import new_record


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Generate GOOD teacher data.")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--config", default="configs/teachers.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args(argv)

    prompts = read_records(args.prompts)
    records: list[dict] = []

    if args.mock:
        k = args.samples or 2
        for p in prompts:
            for i in range(k):
                records.append(new_record(
                    p["prompt_id"], p["prompt"], p.get("language", "python"),
                    p.get("reference_solution", ""), p.get("tests"), "mock", i,
                ))
        print(f"[gen_good] MOCK: {len(records)} records from {len(prompts)} prompts")
    else:
        from .teachers import TeacherPool

        pool = TeacherPool(args.config)
        k = args.samples or pool.defaults.get("samples_per_prompt", 3)
        for p in prompts:
            for i in range(k):
                t = pool.pick(i)
                try:
                    obj, meta = pool.generate_json(
                        t, good_messages(p["prompt"], p.get("language", "python")))
                except SystemExit:
                    raise
                except Exception as e:
                    print(f"[gen_good] skip {p['prompt_id']} s{i} ({t['name']}): {e}")
                    continue
                records.append(new_record(
                    p["prompt_id"], p["prompt"],
                    obj.get("language", p.get("language", "python")),
                    obj.get("code", ""), obj.get("tests"), t["name"], i, meta,
                ))
        print(f"[gen_good] {len(records)} records  spent=${pool.meter.spent_usd:.2f}")

    out = ensure_dir(args.out) / "records.jsonl"
    write_jsonl(out, records)
    print(f"[gen_good] wrote {out}")


if __name__ == "__main__":
    main()
