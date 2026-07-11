# Model Card — your-coder-moe (h3hop)

## Summary
A coding-focused Mixture-of-Experts model: **Step-3.7-Flash** (Apache-2.0) fine-tuned via
sequence-level distillation from two teachers, then preference- and RLVR-aligned, and shipped with
a companion **bad-weight** control vector (`anti-pattern.gguf`, see `ARTIFACT_CARD.md`).

## Base / student
- **Step-3.7-Flash** (`stepfun-ai/Step-3.7-Flash`), 198B total / ≈11B active MoE, Apache-2.0.
  Vision encoder frozen for pure-coding runs.
- Prototype proxy for pipeline validation: **Qwen3-Coder-30B-A3B** (Apache-2.0).

## Teachers (data only — weights NOT included or redistributed)
| Teacher | License | Role | Attribution |
|---|---|---|---|
| GLM-5.2 (`zai-org/GLM-5.2`) | MIT | primary coding-quality teacher | © Zhipu AI / Z.ai |
| DeepSeek-V4-Pro | MIT | algorithmic / verification teacher | © DeepSeek |

Both teacher licenses permit distillation. No closed-model outputs (GPT/Claude/Gemini) were used —
their ToS forbid training competitors.

## Training pipeline
Multi-teacher gen → sandbox+scanner+decontam filtering → best-of fusion SFT → on-policy
distillation (GKD) → DPO/CRINGE on (good, bad) → RLVR (unit-test reward − security-scanner
penalty). See the plan and `README.md`.

## Intended use / limitations
- Intended: code generation/assistance where insecure/buggy patterns should be actively avoided.
- Distillation is sequence-level (black-box); the student approaches but does not equal GLM-5.2.
- Steering/negative training can cost capability if over-applied — see the eval ablation matrix.
- Not a safety guarantee: always review generated code; scanners cover a finite CWE set.

## Data provenance & decontamination
Every training row carries source/teacher/verdict provenance. Corpus decontaminated against
HumanEval/MBPP/LiveCodeBench before training.

## Evaluation
HumanEval+/MBPP+/MultiPL-E + contamination-free LiveCodeBench window, plus the custom bad-pattern
rate. Report: `eval/report.md`.

## License
Your fine-tuned weights: Apache-2.0 (inherited from the Step-3.7-Flash base). Your code: Apache-2.0.
