"""Demo: generate with the bad-weight toggled ON/OFF.

Modes:
  --control-vector anti-pattern.gguf --scale -1.0   (repeng steering; the flagship path)
  --anti-expert bad-code-antiexpert/ --alpha 0.5    (DExperts contrastive decoding)

Shows the core idea: with the bad-weight applied (negative CV, or subtracting the anti-expert),
the model steers away from insecure/buggy code.
"""
from __future__ import annotations

import argparse


def _gen_plain(model, tok, prompt, max_new=256):
    import torch  # noqa: PLC0415

    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  return_tensors="pt", add_generation_prompt=True).to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def _gen_control_vector(model_dir, prompt, gguf, scale, layers):
    import torch  # noqa: PLC0415
    from repeng import ControlModel, ControlVector  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    cmodel = ControlModel(model, layers)
    vec = ControlVector.import_gguf(gguf)
    print("=== bad-weight OFF ===")
    cmodel.reset()
    print(_gen_plain(cmodel, tok, prompt))
    print(f"=== bad-weight ON (scale {scale}) ===")
    cmodel.set_control(vec, scale)
    print(_gen_plain(cmodel, tok, prompt))


def _gen_anti_expert(model_dir, anti_dir, prompt, alpha, max_new=256):
    """DExperts contrastive decoding: logit = logit_student - alpha * logit_antiexpert."""
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415
    from peft import PeftModel  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    good = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    bad = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"),
        anti_dir)

    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  return_tensors="pt", add_generation_prompt=True).to(good.device)
    for _ in range(max_new):
        with torch.no_grad():
            lg = good(ids).logits[:, -1, :]
            lb = bad(ids).logits[:, -1, :]
        nxt = (lg - alpha * lb).argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=-1)
        if nxt.item() == tok.eos_token_id:
            break
    return tok.decode(ids[0], skip_special_tokens=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Bad-weight toggle demo.")
    ap.add_argument("--model", required=True, help="merged student HF dir")
    ap.add_argument("--prompt", default="Look up a user by id in a SQLite database.")
    ap.add_argument("--control-vector", default=None)
    ap.add_argument("--scale", type=float, default=-1.0)
    ap.add_argument("--layers", default="14-26", help="e.g. 14-26")
    ap.add_argument("--anti-expert", default=None)
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args(argv)

    if args.control_vector:
        a, b = (int(x) for x in args.layers.split("-"))
        _gen_control_vector(args.model, args.prompt, args.control_vector, args.scale,
                            list(range(a, b + 1)))
    elif args.anti_expert:
        print(_gen_anti_expert(args.model, args.anti_expert, args.prompt, args.alpha))
    else:
        raise SystemExit("pass --control-vector <gguf> or --anti-expert <dir>")


if __name__ == "__main__":
    main()
