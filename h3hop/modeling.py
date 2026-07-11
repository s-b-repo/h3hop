"""Shared model/tokenizer/LoRA loading for all training stages (heavy track).

All torch/transformers/peft imports are lazy so the package imports without them.
Reads the same student schema (student.yaml / proto_student.yaml).
"""
from __future__ import annotations

from .config import load_yaml


def load_student(student_cfg_path: str):
    """Return (model, tokenizer, cfg) with optional 4-bit quant and a frozen vision encoder."""
    import torch  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    cfg = load_yaml(student_cfg_path)
    q = cfg.get("quantization") or {}
    bnb = None
    if q.get("load_in_4bit"):
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=q.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(torch, q.get("bnb_4bit_compute_dtype", "bfloat16")),
            bnb_4bit_use_double_quant=q.get("bnb_4bit_use_double_quant", True),
        )

    trc = cfg.get("trust_remote_code", True)
    tok = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=trc)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb,
        trust_remote_code=trc,
        torch_dtype=torch.bfloat16,
        attn_implementation=cfg.get("attn_implementation", "eager"),
        device_map="auto",
    )

    if cfg.get("freeze_vision_encoder"):
        frozen = 0
        for name, p in model.named_parameters():
            if any(k in name.lower() for k in ("vision", "visual", "vit", "image")):
                p.requires_grad = False
                frozen += 1
        print(f"[modeling] froze {frozen} vision-encoder params")

    return model, tok, cfg


def lora_config(cfg: dict):
    from peft import LoraConfig  # noqa: PLC0415

    lc = cfg["lora"]
    return LoraConfig(
        r=lc["r"],
        lora_alpha=lc["alpha"],
        lora_dropout=lc.get("dropout", 0.05),
        target_modules=lc["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def prepare_for_training(model, cfg: dict):
    """k-bit prep + attach LoRA. Returns a PEFT model."""
    from peft import get_peft_model, prepare_model_for_kbit_training  # noqa: PLC0415

    if (cfg.get("quantization") or {}).get("load_in_4bit"):
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config(cfg))
    model.print_trainable_parameters()
    return model


def chat_text(tok, prompt: str, completion: str | None = None) -> str:
    """Render a (prompt[, completion]) turn with the model's chat template."""
    msgs = [{"role": "user", "content": prompt}]
    if completion is not None:
        msgs.append({"role": "assistant", "content": completion})
        return tok.apply_chat_template(msgs, tokenize=False)
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
