"""Stage F / N1 — RLVR (GRPO) with a security-scanner NEGATIVE reward.

    reward = w_test * test_pass  -  w_cwe * n_cwe  -  w_style * style_penalty

Unit tests drive correctness; the CWE term trains insecure patterns OUT during RL. Hold out some
scanner rules for eval (configs/rlvr.yaml: holdout_rules) to detect reward-hacking.
"""
from __future__ import annotations

import argparse

from .config import load_yaml
from .io import read_records
from .modeling import chat_text, load_student, prepare_for_training
from .rewards import code_reward


def _text_of(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
        return completion[-1].get("content", "")
    return str(completion)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="RLVR with security-scanner negative reward.")
    ap.add_argument("--student", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--prompts", required=True, help="jsonl with prompt + tests")
    args = ap.parse_args(argv)

    try:
        from datasets import Dataset  # noqa: PLC0415
        from trl import GRPOConfig, GRPOTrainer  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover
        raise SystemExit("heavy stack missing: pip install -e '.[train,serve]'") from e

    cfg = load_yaml(args.config)
    rw = cfg["reward"]
    rules = rw.get("cwe_rules", "eval/icd_rules")
    weights = {"w_test": rw["w_test"], "w_cwe": rw["w_cwe"], "w_style": rw["w_style"]}

    model, tok, scfg = load_student(args.student)
    model = prepare_for_training(model, scfg)

    prompts = read_records(args.prompts)
    ds = Dataset.from_list([
        {"prompt": chat_text(tok, p["prompt"]), "tests": p.get("tests") or ""}
        for p in prompts
    ])

    def reward_func(prompts, completions, tests=None, **kw):  # noqa: ARG001
        tests = tests or [None] * len(completions)
        return [
            code_reward(_text_of(c), t or None, rules=rules, weights=weights)["reward"]
            for c, t in zip(completions, tests)
        ]

    targs = GRPOConfig(
        output_dir=cfg["output_dir"],
        learning_rate=cfg.get("learning_rate", 1e-6),
        num_generations=cfg.get("num_generations", 8),
        max_completion_length=cfg.get("max_new_tokens", 1024),
        temperature=cfg.get("temperature", 1.0),
        beta=cfg.get("kl_beta", 0.04),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        bf16=cfg.get("bf16", True),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        seed=cfg.get("seed", 17),
        report_to=cfg.get("report_to", "none"),
    )
    trainer = GRPOTrainer(
        model=model, args=targs, train_dataset=ds,
        reward_funcs=[reward_func], processing_class=tok,
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    print(f"[rlvr] saved adapter -> {cfg['output_dir']}")


if __name__ == "__main__":
    main()
