# h3hop — Distill GLM-5.2 → Step-3.7-Flash + a shippable "bad-weight"

Fine-tune **Step-3.7-Flash** (198B/≈11B-active MoE, Apache-2.0) into your own coding model by
**sequence-level distillation** from two teachers — **GLM-5.2** (primary, MIT) and
**DeepSeek-V4-Pro** (verifier, MIT) — and produce a **"bad-weight"**: a small, shareable
control vector (+ optional DExperts anti-expert) that steers the model *away* from insecure /
buggy / anti-pattern code.

> Full design & rationale: `/home/kali/.claude/plans/i-want-to-distill-flickering-kahan.md`

## What "bad-weight" means here
A bad pattern (SQLi, hardcoded secrets, command injection, buggy logic, hallucinated APIs) lives
on a measurable **direction** in the model's activation space. We (a) train the student to *prefer*
good over bad (DPO/CRINGE + RLVR with a security-scanner **negative** reward) and (b) export that
"bad direction" as a few-MB `anti-pattern.gguf` control vector you apply with a **negative**
coefficient at inference — toggleable, strength-adjustable, shareable "like an open weight."

## Repo layout
```
configs/     yaml for teachers, student, each training stage, control vector, eval
h3hop/       the pipeline package (gen, filter, build, train, bad-weight, infer)
eval/        capability + custom bad-pattern evaluation + report
scripts/     smoke test + gguf conversion
data/        prompts + generated/filtered data (raw/good/bad/pairs/contrast)
artifacts/   trained model + anti-pattern.gguf + anti-expert (output)
```

## Two-track design (important)
- **Light track — runs on any laptop, no GPU/keys.** Config loading, code sandboxing & filtering,
  decontamination, dataset/pair building, the bad-pattern scorer, reporting, and the offline
  **`--mock`** generator. Use this to validate the whole pipeline shape.
- **Heavy track — needs your cluster + API keys.** Teacher generation (OpenRouter) and all training
  (SFT / on-policy GKD / DPO / RLVR) and the `repeng` control-vector build. Heavy imports
  (`torch`, `trl`, `transformers`, `repeng`, `vllm`) are lazy so the package imports without them.

## Quickstart
```bash
# 1. Environment  (Python 3.11/3.12 recommended for the ML stack; 3.13 works for the light track)
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                 # light track only
pip install -e ".[train,vector,serve]"   # heavy track (on the GPU node)

cp .env.example .env             # add OPENROUTER_API_KEY

# 2. Prove the plumbing offline (no GPU, no API) — generates mock data and runs filter→build→pairs
make smoke-light

# 3. Real prototype on the cheap proxy model (1×80GB GPU, small API spend)
make gen         # GLM-5.2 + DeepSeek-V4 via OpenRouter  (respects budget cap in configs/teachers.yaml)
make filter      # sandbox-execute + ruff/bandit/semgrep + decontaminate + LLM-judge
make dataset     # best-of fusion, dedup, split  → data/good, data/bad
make pairs       # (prompt, chosen, rejected) + contrast set  → data/pairs, data/contrast
make sft dpo     # QLoRA SFT then DPO  (STUDENT=configs/proto_student.yaml)
make vector      # build anti-pattern.gguf from the contrast set
make eval        # capability + bad-pattern + toggle test

# 4. Promote to Step-3.7-Flash (multi-GPU node): same commands, STUDENT=configs/student.yaml
```

## Legal / attribution
- Student **Step-3.7-Flash** is Apache-2.0 — you may fine-tune and re-ship your weights.
- Teachers **GLM-5.2 (MIT)** and **DeepSeek-V4-Pro (MIT)** permit distillation; keep attribution
  (see `MODEL_CARD.md`). **Never** distill from closed models (GPT/Claude/Gemini) — their ToS forbid
  training competitors. You ship only *your* fine-tune + your bad-weight artifacts, never teacher weights.

See `MODEL_CARD.md` and `ARTIFACT_CARD.md` before releasing anything.
