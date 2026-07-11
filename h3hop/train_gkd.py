"""Stage D — On-policy distillation (Generalized Knowledge Distillation) via TRL GKDTrainer.

Student generates on its own distribution; the teacher supervises. reverse-KL (beta->1) is
mode-seeking and tends to beat plain SFT for code/reasoning. Requires an on-node teacher model
(configs/gkd.yaml: teacher_model). For distilling into a MoE student from a non-router teacher,
see the PADD note in configs/gkd.yaml.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_yaml
from .io import read_records
from .modeling import chat_text, load_student, prepare_for_training


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="On-policy distillation (GKD).")
    ap.add_argument("--student", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    args = ap.parse_args(argv)

    cfg = load_yaml(args.config)
    if not cfg.get("teacher_model"):
        raise SystemExit(
            "gkd.yaml: set `teacher_model` to an on-node teacher (e.g. a local GLM checkpoint or "
            "a smaller distilled teacher). On-policy distillation needs teacher logits."
        )

    try:
        from datasets import Dataset  # noqa: PLC0415
        from transformers import AutoModelForCausalLM  # noqa: PLC0415
        from trl import GKDConfig, GKDTrainer  # noqa: PLC0415
        import torch  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover
        raise SystemExit("heavy stack missing: pip install -e '.[train,serve]'") from e

    model, tok, scfg = load_student(args.student)
    model = prepare_for_training(model, scfg)
    teacher = AutoModelForCausalLM.from_pretrained(
        cfg["teacher_model"], torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map="auto",
    )

    p = Path(args.data)
    recs = read_records(p / "train.jsonl") if p.is_dir() else read_records(p)
    # GKD trains on prompts; the student rolls out completions during training.
    ds = Dataset.from_list([{"messages": [{"role": "user", "content": r["prompt"]}]} for r in recs])

    targs = GKDConfig(
        output_dir=cfg["output_dir"],
        lmbda=cfg.get("lmbda", 0.5),
        beta=cfg.get("beta", 0.5),
        temperature=cfg.get("temperature", 1.0),
        max_new_tokens=cfg.get("max_new_tokens", 1024),
        learning_rate=cfg.get("learning_rate", 5e-5),
        num_train_epochs=cfg.get("num_train_epochs", 1),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 16),
        bf16=cfg.get("bf16", True),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        seed=cfg.get("seed", 17),
        report_to=cfg.get("report_to", "none"),
    )
    trainer = GKDTrainer(
        model=model, teacher_model=teacher, args=targs, train_dataset=ds, tokenizer=tok,
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    print(f"[gkd] saved adapter -> {cfg['output_dir']}")


if __name__ == "__main__":
    main()
