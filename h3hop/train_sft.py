"""Stage C — SFT warmup on fused GOOD data (QLoRA via TRL SFTTrainer)."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_yaml
from .io import read_records
from .modeling import chat_text, load_student, prepare_for_training


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="SFT warmup.")
    ap.add_argument("--student", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True, help="data/good dir or a jsonl")
    args = ap.parse_args(argv)

    try:
        from datasets import Dataset  # noqa: PLC0415
        from trl import SFTConfig, SFTTrainer  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover
        raise SystemExit("heavy stack missing: pip install -e '.[train]'") from e

    cfg = load_yaml(args.config)
    model, tok, scfg = load_student(args.student)
    model = prepare_for_training(model, scfg)

    p = Path(args.data)
    recs = read_records(p / "train.jsonl") if p.is_dir() else read_records(p)
    ds = Dataset.from_list([{"text": chat_text(tok, r["prompt"], r["code"])} for r in recs])
    print(f"[sft] {len(ds)} examples")

    targs = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg.get("num_train_epochs", 2),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 16),
        learning_rate=cfg.get("learning_rate", 1e-4),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 200),
        bf16=cfg.get("bf16", True),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        max_seq_length=scfg.get("max_seq_len", 4096),
        packing=cfg.get("packing", True),
        dataset_text_field="text",
        seed=cfg.get("seed", 17),
        report_to=cfg.get("report_to", "none"),
    )
    trainer = SFTTrainer(model=model, args=targs, train_dataset=ds, tokenizer=tok)
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    print(f"[sft] saved adapter -> {cfg['output_dir']}")


if __name__ == "__main__":
    main()
