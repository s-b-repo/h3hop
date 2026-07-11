"""Stage G — build the shippable 'bad-weight' control vector (the flagship artifact).

Contrasts activations on GOOD vs BAD code (data/contrast) on the RESIDUAL STREAM (robust to MoE
routing) and exports a small GGUF you apply with a NEGATIVE coefficient to steer away from bad
patterns:  `llama-cli -m model.gguf --control-vector-scaled anti-pattern.gguf -1.0`
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_yaml
from .io import read_records


def _wrap(code: str) -> str:
    return f"Consider the following code:\n{code}"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build the bad-pattern control vector.")
    ap.add_argument("--model", required=True, help="merged student (HF dir) to extract from")
    ap.add_argument("--config", default="configs/control_vector.yaml")
    ap.add_argument("--contrast", required=True, help="data/contrast dir or contrast.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    try:
        import torch  # noqa: PLC0415
        from repeng import ControlModel, ControlVector, DatasetEntry  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover
        raise SystemExit("control-vector stack missing: pip install -e '.[vector]'") from e

    cfg = load_yaml(args.config)
    p = Path(args.contrast)
    contrast = read_records(p / "contrast.jsonl" if p.is_dir() else p)
    contrast = contrast[: cfg.get("max_examples", 400)]
    if not contrast:
        raise SystemExit("no contrast pairs — run build_pairs first")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    cmodel = ControlModel(model, cfg["layers"])

    dataset = [DatasetEntry(positive=_wrap(c["positive"]), negative=_wrap(c["negative"]))
               for c in contrast]
    print(f"[control_vector] training on {len(dataset)} contrast pairs, layers={cfg['layers']}")

    if cfg.get("bipo", {}).get("enabled"):
        # N3 (experimental): a DPO-style bi-directional objective yields a stronger, learned
        # direction than PCA. Not implemented in repeng out of the box — see docs/BiPO.md to
        # plug in a learned vector. Falling back to PCA so the pipeline still produces an artifact.
        print("[control_vector] BiPO requested but not wired; using PCA method as fallback (N3 TODO)")

    vector = ControlVector.train(cmodel, tok, dataset, method=cfg.get("method", "pca_center"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    vector.export_gguf(str(out))
    print(f"[control_vector] exported {out}  (apply with negative scale, e.g. -1.0)")
    print(f"[control_vector] recommended default scale: {cfg.get('recommend_scale', -1.0)} "
          f"(tune via the sweep in eval.yaml)")


if __name__ == "__main__":
    main()
