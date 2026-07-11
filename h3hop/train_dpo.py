"""Stage E — DPO on (prompt, chosen=good, rejected=bad). First weight-level 'learn what's bad'."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_yaml
from .io import read_records
from .modeling import chat_text, load_student, lora_config


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="DPO on preference pairs.")
    ap.add_argument("--student", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True, help="data/pairs dir or pairs.jsonl")
    args = ap.parse_args(argv)

    try:
        from datasets import Dataset  # noqa: PLC0415
        from peft import prepare_model_for_kbit_training  # noqa: PLC0415
        from trl import DPOConfig, DPOTrainer  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover
        raise SystemExit("heavy stack missing: pip install -e '.[train]'") from e

    cfg = load_yaml(args.config)
    model, tok, scfg = load_student(args.student)
    if (scfg.get("quantization") or {}).get("load_in_4bit"):
        model = prepare_model_for_kbit_training(model)

    p = Path(args.data)
    pairs = read_records(p / "pairs.jsonl") if p.is_dir() else read_records(p)
    ds = Dataset.from_list([
        {"prompt": chat_text(tok, r["prompt"]), "chosen": r["chosen"], "rejected": r["rejected"]}
        for r in pairs
    ])
    print(f"[dpo] {len(ds)} pairs")

    targs = DPOConfig(
        output_dir=cfg["output_dir"],
        beta=cfg.get("beta", 0.1),
        loss_type=cfg.get("loss_type", "sigmoid"),
        learning_rate=cfg.get("learning_rate", 5e-6),
        num_train_epochs=cfg.get("num_train_epochs", 1),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 16),
        max_length=cfg.get("max_length", 2048),
        max_prompt_length=cfg.get("max_prompt_length", 1024),
        warmup_ratio=cfg.get("warmup_ratio", 0.05),
        bf16=cfg.get("bf16", True),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        seed=cfg.get("seed", 17),
        report_to=cfg.get("report_to", "none"),
    )
    # ref_model=None => the frozen base acts as reference; peft_config attaches the LoRA to train.
    # NOTE: TRL >=0.12 renamed `tokenizer=` to `processing_class=`.
    trainer = DPOTrainer(
        model=model, ref_model=None, args=targs,
        train_dataset=ds, tokenizer=tok, peft_config=lora_config(scfg),
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    print(f"[dpo] saved adapter -> {cfg['output_dir']}")


if __name__ == "__main__":
    main()
