# Artifact Card — anti-pattern.gguf (the "bad-weight")

## What it is
A **control vector** (representation-engineering steering vector), a few MB, encoding the direction
in the student's residual stream that separates **insecure/buggy code** from **secure/correct code**.
It is a standalone, shareable file — "like an open weight, but it encodes the patterns to avoid."

## How to use
```bash
# Steer AWAY from bad patterns (recommended):
llama-cli -m your-coder-moe.Q4_K_M.gguf \
  --control-vector-scaled anti-pattern.gguf -1.0 \
  --control-vector-layer-range 14 26 \
  -p "Look up a user by id in a SQLite database."
```
- **Negative scale** (e.g. `-1.0`) suppresses bad patterns. **Positive** amplifies them (red-team /
  proof-of-direction only). `0` = off.
- Recommended default: **-1.0** (retune per model via the sweep in `configs/eval.yaml`).
- Also usable via `repeng` in Python (`h3hop/infer_demo.py --control-vector`).

## How it was built
`repeng` PCA over matched good/bad code activations (`data/contrast`) at layers −18..−5, exported to
GGUF (`h3hop/build_control_vector.py`). Optional stronger variant: BiPO-learned vector (N3).

## Companion artifact
`bad-code-antiexpert/` — a LoRA trained only on bad code, for **DExperts** contrastive decoding
(`logit = student − α·anti-expert`). See `h3hop/infer_demo.py --anti-expert`.

## Verified behavior (toggle test)
`eval/run_badpattern.py` measures bad-pattern emission at scales {0, −1.0, +1.0}. Expected and
required: **rate(−1.0) < rate(0) < rate(+1.0)** — the asymmetry proves the vector encodes the
intended direction. Record the chosen scale + measured rates below after your run.

| scale | bad-pattern rate | HumanEval+ (capability) |
|---|---|---|
| 0 (off) | _fill in_ | _fill in_ |
| -1.0 (ship default) | _fill in_ | _fill in_ |
| +1.0 (sanity) | _fill in_ | — |

## Caveats
- Over-strong steering (|scale| ≳ 1.5) degrades capability / risks emergent misalignment — tune and
  evaluate WITH-vs-WITHOUT.
- Vector is student-specific (extracted from these weights); rebuild it if you retrain the model.
- Scanners cover a finite CWE set; the vector reflects only patterns present in the contrast data.

## License
Apache-2.0.
