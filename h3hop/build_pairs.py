"""Stage C: form (prompt, chosen, rejected) DPO pairs + the good/bad contrast set.

Pairs a prompt's best GOOD with each BAD sharing its prompt_id. Asserts good/bad id-disjointness
(a hard guardrail against the bad set leaking into the good SFT set).
"""
from __future__ import annotations

import argparse

from .io import ensure_dir, read_records, write_jsonl


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build DPO pairs + contrast set.")
    ap.add_argument("--good", required=True)
    ap.add_argument("--bad", required=True)
    ap.add_argument("--out-pairs", required=True)
    ap.add_argument("--out-contrast", required=True)
    ap.add_argument("--max-per-prompt", type=int, default=2)
    args = ap.parse_args(argv)

    good = read_records(args.good)
    bad = read_records(args.bad)

    good_ids = {r["id"] for r in good}
    bad_ids = {r["id"] for r in bad}
    overlap = good_ids & bad_ids
    if overlap:
        raise SystemExit(f"good/bad id overlap ({len(overlap)}) — bad set leaked into good! e.g. {list(overlap)[:3]}")

    good_by = {r["prompt_id"]: r for r in good}  # best-of already => one per prompt
    bad_by: dict[str, list[dict]] = {}
    for r in bad:
        bad_by.setdefault(r["prompt_id"], []).append(r)

    pairs, contrast = [], []
    for pid, g in good_by.items():
        for b in bad_by.get(pid, [])[: args.max_per_prompt]:
            pairs.append({"prompt_id": pid, "prompt": g["prompt"],
                          "chosen": g["code"], "rejected": b["code"]})
            contrast.append({"prompt_id": pid, "positive": g["code"], "negative": b["code"]})

    write_jsonl(ensure_dir(args.out_pairs) / "pairs.jsonl", pairs)
    write_jsonl(ensure_dir(args.out_contrast) / "contrast.jsonl", contrast)
    print(f"[build_pairs] pairs={len(pairs)}  contrast={len(contrast)}  "
          f"(good_prompts={len(good_by)} bad_prompts={len(bad_by)})")


if __name__ == "__main__":
    main()
