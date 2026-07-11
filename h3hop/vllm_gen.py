"""Optional: self-hosted batched generation with vLLM (teacher self-host or student rollouts).

Only useful if you have the VRAM to run the model locally (e.g., Step-3.7 on a big node, or a
smaller teacher). Lazy import so the package works without vLLM installed.
"""
from __future__ import annotations

import argparse

from .io import ensure_dir, read_records, write_jsonl
from .prompts import good_messages
from .schema import new_record


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="vLLM offline generation.")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    args = ap.parse_args(argv)

    from vllm import LLM, SamplingParams  # noqa: PLC0415

    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size,
              trust_remote_code=True)
    sp = SamplingParams(n=args.samples, temperature=args.temperature, max_tokens=args.max_tokens)

    prompts = read_records(args.prompts)
    convos = [good_messages(p["prompt"], p.get("language", "python")) for p in prompts]
    outs = llm.chat(convos, sp)

    records = []
    for p, out in zip(prompts, outs):
        for i, cand in enumerate(out.outputs):
            records.append(new_record(p["prompt_id"], p["prompt"], p.get("language", "python"),
                                      cand.text, p.get("tests"), f"vllm:{args.model}", i))
    out_path = ensure_dir(args.out) / "records.jsonl"
    write_jsonl(out_path, records)
    print(f"[vllm_gen] {len(records)} records -> {out_path}")


if __name__ == "__main__":
    main()
