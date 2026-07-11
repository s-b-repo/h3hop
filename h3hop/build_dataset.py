"""Stage C: fuse teachers (best-of), route insecure/failed to BAD, dedup, split.

GOOD  = ran + passed tests + not insecure + not contaminated, best-of per prompt.
BAD   = explicit bad-gen records + any good-gen candidate that failed or was insecure.
The two sets are kept strictly disjoint (build_pairs asserts it).
"""
from __future__ import annotations

import argparse
import hashlib
import random

from .io import ensure_dir, read_records, write_jsonl
from .schema import is_good


def _dedup(records: list[dict], threshold: float) -> list[dict]:
    try:
        from datasketch import MinHash, MinHashLSH  # noqa: PLC0415

        lsh = MinHashLSH(threshold=threshold, num_perm=128)
        kept = []
        for i, r in enumerate(records):
            m = MinHash(num_perm=128)
            for tok in set(r["code"].split()):
                m.update(tok.encode())
            if not lsh.query(m):
                lsh.insert(str(i), m)
                kept.append(r)
        return kept
    except Exception:
        seen: set[str] = set()
        kept = []
        for r in records:
            h = hashlib.sha1(r["code"].encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                kept.append(r)
        return kept


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build good/bad datasets.")
    ap.add_argument("--in", dest="inp", required=True, help="filtered good candidates")
    ap.add_argument("--bad-in", dest="bad_in", required=True, help="bad-gen candidates")
    ap.add_argument("--extra-bad", default=None, help="optional unpaired negatives")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-fraction", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--dedup-threshold", type=float, default=0.85)
    args = ap.parse_args(argv)

    cands = read_records(args.inp)
    bad_pool = read_records(args.bad_in)
    if args.extra_bad:
        bad_pool += read_records(args.extra_bad)

    goods = [r for r in cands if is_good(r)]
    bad_pool += [r for r in cands if not is_good(r)]  # failed/insecure "good" attempts

    # best-of per prompt: highest judge_score, then shorter code
    best: dict[str, tuple] = {}
    for r in goods:
        key = r["prompt_id"]
        score = (r.get("judge_score") or 0.0, -len(r["code"]))
        if key not in best or score > best[key][0]:
            best[key] = (score, r)
    good_final = _dedup([v[1] for v in best.values()], args.dedup_threshold)
    bad_final = _dedup([r for r in bad_pool if r.get("code")], args.dedup_threshold)

    for r in good_final:
        r["label"] = "good"
    for r in bad_final:
        r["label"] = "bad"

    rng = random.Random(args.seed)
    rng.shuffle(good_final)
    n_val = max(1, int(len(good_final) * args.val_fraction)) if good_final else 0
    val, train = good_final[:n_val], good_final[n_val:]

    out = ensure_dir(args.out)
    write_jsonl(out / "good" / "train.jsonl", train)
    write_jsonl(out / "good" / "val.jsonl", val)
    write_jsonl(out / "bad" / "bad.jsonl", bad_final)
    print(f"[build_dataset] good: train={len(train)} val={len(val)}  bad={len(bad_final)}")


if __name__ == "__main__":
    main()
