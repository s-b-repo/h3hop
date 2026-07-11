# data/ — pipeline data

```
prompts/     seed tasks (good reference + bad counterpart) and optional extra negatives
raw/         gen_good output (good candidates)          [gitignored]
bad_raw/     gen_bad output (bad candidates)            [gitignored]
good/        train.jsonl + val.jsonl (filtered good)    [gitignored]
bad/         bad.jsonl (negatives)                      [gitignored]
pairs/       pairs.jsonl (prompt, chosen, rejected)     [gitignored]
contrast/    contrast.jsonl (positive/negative code) for the control vector  [gitignored]
```

## ⚠️ Intentional insecure code (by design)

`prompts/seed_tasks.jsonl` (`bad_reference` fields) and `prompts/bad_seed_instructions.jsonl`
contain **deliberately insecure or buggy code** — SQL injection, `os.popen`/`os.system`,
`eval`, `pickle.loads`, `yaml.load`, MD5, hardcoded secrets, etc.

These are **training negatives**: the `rejected` side of DPO pairs and the `negative` side of the
control-vector contrast set. Their entire purpose is to teach the student what to **avoid**. They
are data strings, never imported or executed as product behavior (the only code the pipeline
*executes* is inside the isolated sandbox in `h3hop/filter/sandbox.py`, under resource limits).

Security scanners and IDE hooks will (correctly) flag these files — that is expected. Do not
"fix" the `bad_reference` snippets; fixing them would remove the very anti-patterns the bad-weight
is built to suppress. JSONL cannot carry inline comments, which is why this note lives here.
