"""Stage E (optional) — CRINGE / unlikelihood training (arXiv 2211.05826).

    L = CE(chosen)  +  lambda * UnlikelihoodLoss(rejected)

Unlikelihood pushes down the probability the model assigns to the rejected continuation's tokens.
A compact custom loop (batch size 1) rather than HF Trainer, so the two-sided loss is explicit.
Run as an ALTERNATIVE to DPO to compare — not blindly stacked.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_yaml
from .io import read_records
from .modeling import chat_text, load_student, prepare_for_training


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="CRINGE / unlikelihood training.")
    ap.add_argument("--student", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    args = ap.parse_args(argv)

    try:
        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover
        raise SystemExit("heavy stack missing: pip install -e '.[train]'") from e

    cfg = load_yaml(args.config)
    lam = cfg.get("lambda_unlikelihood", 1.0)
    lr = cfg.get("learning_rate", 1e-5)
    epochs = cfg.get("num_train_epochs", 1)

    model, tok, scfg = load_student(args.student)
    model = prepare_for_training(model, scfg)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    p = Path(args.data)
    pairs = read_records(p / "pairs.jsonl") if p.is_dir() else read_records(p)

    def _loss_for(prompt: str, completion: str, unlikelihood: bool):
        text = chat_text(tok, prompt, completion)
        enc = tok(text, return_tensors="pt", truncation=True,
                  max_length=scfg.get("max_seq_len", 4096)).to(model.device)
        out = model(**enc)
        logits = out.logits[:, :-1, :]
        targets = enc["input_ids"][:, 1:]
        logp = F.log_softmax(logits, dim=-1)
        tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        if not unlikelihood:
            return -tok_logp.mean()                      # standard CE (likelihood)
        p_tok = tok_logp.exp().clamp(max=1 - 1e-6)
        return -(1.0 - p_tok).log().mean()               # unlikelihood

    for ep in range(epochs):
        for i, r in enumerate(pairs):
            loss = _loss_for(r["prompt"], r["chosen"], False) \
                + lam * _loss_for(r["prompt"], r["rejected"], True)
            loss.backward()
            opt.step()
            opt.zero_grad()
            if i % 10 == 0:
                print(f"[cringe] ep{ep} step{i} loss={loss.item():.4f}")

    out_dir = cfg["output_dir"]
    model.save_pretrained(out_dir)
    print(f"[cringe] saved adapter -> {out_dir}")


if __name__ == "__main__":
    main()
