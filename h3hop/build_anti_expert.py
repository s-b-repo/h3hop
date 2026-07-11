"""Stage G — the DExperts 'anti-expert' (+ experimental in-MoE anti-expert scan, N2).

Two artifacts:
  1. A LoRA trained ONLY on BAD code -> a model that is *good at being bad*. At decode time,
     subtract its logits (DExperts):  logit = logit_student - alpha * logit_antiexpert
     (implemented in infer_demo.py --anti-expert).
  2. --moe-scan (experimental, N2): identify which Step-3.7 experts fire disproportionately on
     bad code via router-logit sensitivity, so you can negatively gate / contrast that expert
     inside the single model. Arch-specific — adapt the gate-module match to Step-3.7.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_yaml
from .io import read_records
from .modeling import chat_text, load_student, prepare_for_training


def train_bad_lora(student: str, bad_dir: str, out: str, epochs: int = 1, lr: float = 1e-4) -> None:
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    model, tok, scfg = load_student(student)
    model = prepare_for_training(model, scfg)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    p = Path(bad_dir)
    bad = read_records(p / "bad.jsonl" if p.is_dir() else p)
    print(f"[anti_expert] training bad-only LoRA on {len(bad)} bad examples")
    for ep in range(epochs):
        for i, r in enumerate(bad):
            text = chat_text(tok, r["prompt"], r["code"])
            enc = tok(text, return_tensors="pt", truncation=True,
                      max_length=scfg.get("max_seq_len", 4096)).to(model.device)
            out_ = model(**enc)
            logits, targets = out_.logits[:, :-1, :], enc["input_ids"][:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            opt.step()
            opt.zero_grad()
            if i % 10 == 0:
                print(f"[anti_expert] ep{ep} step{i} loss={loss.item():.4f}")
    model.save_pretrained(out)
    print(f"[anti_expert] saved bad-only LoRA -> {out}")


def moe_scan(student: str, contrast_dir: str) -> None:
    """Experimental (N2): correlate MoE router activation with bad vs good code."""
    import torch  # noqa: PLC0415

    model, tok, _ = load_student(student)
    contrast = read_records(Path(contrast_dir) / "contrast.jsonl")
    gate_acts: dict[str, list] = {}

    def hook(name):
        def fn(_m, _inp, out):
            gate_acts.setdefault(name, []).append(out.detach().float().mean(dim=(0, 1)).cpu())
        return fn

    handles = []
    for name, mod in model.named_modules():
        # Adapt this match to Step-3.7's router/gate module name.
        if name.endswith("gate") or "router" in name:
            handles.append(mod.register_forward_hook(hook(name)))
    if not handles:
        print("[moe_scan] no router/gate modules matched — adjust the name filter for this arch")
        return

    def mean_gate(code: str):
        for k in gate_acts:
            gate_acts[k] = []
        enc = tok(code, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            model(**enc)
        return {k: torch.stack(v).mean(0) for k, v in gate_acts.items() if v}

    diffs: dict = {}
    for c in contrast[:64]:
        g, b = mean_gate(c["positive"]), mean_gate(c["negative"])
        for k in g:
            if k in b:
                diffs[k] = diffs.get(k, 0) + (b[k] - g[k])
    for h in handles:
        h.remove()

    print("[moe_scan] experts most activated by BAD > GOOD (candidate anti-experts):")
    for k, v in diffs.items():
        top = v.topk(min(3, v.numel()))
        print(f"  {k}: expert idx {top.indices.tolist()} (delta {top.values.tolist()})")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build the DExperts anti-expert (N2).")
    ap.add_argument("--model", dest="student", required=True, help="student cfg or HF dir")
    ap.add_argument("--bad", default="data/bad")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/cringe.yaml")
    ap.add_argument("--moe-scan", metavar="CONTRAST_DIR", default=None)
    args = ap.parse_args(argv)

    try:
        import torch  # noqa: F401, PLC0415
    except ImportError as e:  # pragma: no cover
        raise SystemExit("heavy stack missing: pip install -e '.[train]'") from e

    if args.moe_scan:
        moe_scan(args.student, args.moe_scan)
        return
    cfg = load_yaml(args.config) if Path(args.config).exists() else {}
    train_bad_lora(args.student, args.bad, args.out,
                   epochs=cfg.get("num_train_epochs", 1), lr=cfg.get("learning_rate", 1e-4))


if __name__ == "__main__":
    main()
