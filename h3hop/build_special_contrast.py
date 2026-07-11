"""Build cheat/stego contrast sets so build_control_vector can produce anti-cheat / anti-stego
steering directions (the bad-weight extended beyond insecure/buggy code).

Positive = clean examples, Negative = cheating / hidden examples. In a real run, point these at
large labelled corpora; here they bootstrap from the eval suites so the path is runnable offline.
"""
from __future__ import annotations

import argparse
import itertools

from .io import ensure_dir, read_records, write_jsonl


def _contrast(clean: list[dict], bad: list[dict], field: str) -> list[dict]:
    if not clean or not bad:
        return []
    pairs = []
    for i, (p, n) in enumerate(zip(itertools.cycle(clean), bad)):
        pairs.append({"prompt_id": f"c{i}", "positive": p[field], "negative": n[field]})
    return pairs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build cheat/stego contrast sets.")
    ap.add_argument("--cheat-suite", default="eval/cheating_suite.jsonl")
    ap.add_argument("--stego-suite", default="eval/stego_probes.jsonl")
    ap.add_argument("--out-cheat", default="data/contrast_cheat")
    ap.add_argument("--out-stego", default="data/contrast_stego")
    args = ap.parse_args(argv)

    cheat = read_records(args.cheat_suite)
    cpairs = _contrast([r for r in cheat if not r["expect_cheat"]],
                       [r for r in cheat if r["expect_cheat"]], "code")
    write_jsonl(ensure_dir(args.out_cheat) / "contrast.jsonl", cpairs)

    stego = read_records(args.stego_suite)
    spairs = _contrast([r for r in stego if not r["expect_hidden"]],
                       [r for r in stego if r["expect_hidden"]], "text")
    write_jsonl(ensure_dir(args.out_stego) / "contrast.jsonl", spairs)

    print(f"[special_contrast] anti-cheat pairs={len(cpairs)} -> {args.out_cheat}")
    print(f"[special_contrast] anti-stego pairs={len(spairs)} -> {args.out_stego}")


if __name__ == "__main__":
    main()
