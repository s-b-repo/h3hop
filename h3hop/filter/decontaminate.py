"""Stage B (3/3): flag candidates that overlap benchmark solutions (contamination).

Benchmark reference solutions are loaded from data/benchmarks/<name>.jsonl if present
(field: solution|canonical_solution|code|text), else via the `datasets` library if installed,
else the stage warns and flags nothing (documented no-op).
Heuristic: any shared 13-gram (whitespace tokens) with a benchmark solution => flagged.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..io import ensure_dir, read_jsonl, read_records, write_jsonl

NGRAM = 13
SOL_FIELDS = ("canonical_solution", "solution", "code", "text", "prompt")


def _shingles(text: str, n: int = NGRAM) -> set[str]:
    toks = text.split()
    return {" ".join(toks[i : i + n]) for i in range(max(0, len(toks) - n + 1))}


def _load_benchmark(name: str) -> list[str]:
    local = Path("data/benchmarks") / f"{name}.jsonl"
    if local.exists():
        out = []
        for r in read_jsonl(local):
            for f in SOL_FIELDS:
                if r.get(f):
                    out.append(str(r[f]))
                    break
        return out
    try:
        from datasets import load_dataset  # noqa: PLC0415

        mapping = {
            "humaneval": ("openai_humaneval", "test", "canonical_solution"),
            "mbpp": ("mbpp", "test", "code"),
        }
        if name in mapping:
            ds_name, split, field = mapping[name]
            ds = load_dataset(ds_name, split=split)
            return [str(x[field]) for x in ds if x.get(field)]
    except Exception:
        pass
    print(f"[decontaminate] WARNING: no reference for '{name}' (no local file, datasets unavailable) — not flagging.")
    return []


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Benchmark decontamination.")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--benchmarks", default="humaneval,mbpp")
    args = ap.parse_args(argv)

    bench_shingles: set[str] = set()
    for name in [b.strip() for b in args.benchmarks.split(",") if b.strip()]:
        for sol in _load_benchmark(name):
            bench_shingles |= _shingles(sol)

    recs = read_records(args.inp)
    n_flag = 0
    for rec in recs:
        flagged = bool(bench_shingles and (_shingles(rec["code"]) & bench_shingles))
        rec["decontam_flagged"] = flagged
        n_flag += flagged

    out = ensure_dir(args.out) / "records.jsonl"
    write_jsonl(out, recs)
    print(f"[decontaminate] {len(recs)} records -> {out}  flagged={n_flag}  bench_ngrams={len(bench_shingles)}")


if __name__ == "__main__":
    main()
