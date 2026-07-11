"""Merge one or more LoRA adapters (sequentially) into the base -> a standalone HF model dir."""
from __future__ import annotations

import argparse

from .config import load_yaml


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Merge LoRA adapters into the base.")
    ap.add_argument("--student", required=True, help="student cfg (for base_model)")
    ap.add_argument("--adapters", required=True, help="comma-separated adapter dirs, applied in order")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    try:
        import torch  # noqa: PLC0415
        from peft import PeftModel  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover
        raise SystemExit("heavy stack missing: pip install -e '.[train]'") from e

    scfg = load_yaml(args.student)
    base = scfg["base_model"]
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

    for adapter in [a.strip() for a in args.adapters.split(",") if a.strip()]:
        print(f"[merge] applying {adapter}")
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"[merge] wrote merged model -> {args.out}")


if __name__ == "__main__":
    main()
