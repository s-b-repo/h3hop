# h3hop pipeline.  Override the student per invocation:
#   make sft STUDENT=configs/student.yaml        # real Step-3.7-Flash (multi-GPU)
# Default is the cheap prototype proxy so nothing accidentally launches a 198B run.
PY      ?= python
STUDENT ?= configs/proto_student.yaml
PROMPTS ?= data/prompts/seed_tasks.jsonl

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	 awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ---------- light track (no GPU / no API) ----------
.PHONY: smoke-light
smoke-light: ## Offline end-to-end plumbing check (mock gen -> filter -> dataset -> pairs)
	bash scripts/smoke_test.sh --light

# ---------- data ----------
.PHONY: gen
gen: ## Generate teacher data via OpenRouter (add MOCK=1 for offline mock data)
	$(PY) -m h3hop.gen_good --prompts $(PROMPTS) --config configs/teachers.yaml --out data/raw $(if $(MOCK),--mock,)
	$(PY) -m h3hop.gen_bad  --prompts $(PROMPTS) --config configs/teachers.yaml --out data/bad_raw $(if $(MOCK),--mock,)

.PHONY: filter
filter: ## Sandbox-execute + ruff/bandit/semgrep + decontaminate + integrity scan
	$(PY) -m h3hop.filter.run_tests      --in data/raw --out data/raw_tested
	$(PY) -m h3hop.filter.static_checks  --in data/raw_tested --out data/raw_checked
	$(PY) -m h3hop.filter.decontaminate  --in data/raw_checked --benchmarks humaneval,mbpp --out data/raw_clean
	$(PY) -m h3hop.filter.integrity_scan --in data/raw_clean --out data/raw_clean

.PHONY: dataset
dataset: ## Best-of multi-teacher fusion, dedup, split -> data/good, data/bad
	$(PY) -m h3hop.build_dataset --in data/raw_clean --bad-in data/bad_raw --out data

.PHONY: pairs
pairs: ## Build (prompt, chosen, rejected) + contrast set
	$(PY) -m h3hop.build_pairs --good data/good --bad data/bad --out-pairs data/pairs --out-contrast data/contrast

# ---------- training (heavy) ----------
.PHONY: sft gkd dpo cringe rlvr
sft:    ## SFT warmup on fused good data
	$(PY) -m h3hop.train_sft   --student $(STUDENT) --config configs/sft.yaml    --data data/good
gkd:    ## On-policy distillation (GKD/PADD)
	$(PY) -m h3hop.train_gkd   --student $(STUDENT) --config configs/gkd.yaml    --data data/good
dpo:    ## DPO on (good, bad) pairs
	$(PY) -m h3hop.train_dpo   --student $(STUDENT) --config configs/dpo.yaml    --data data/pairs
cringe: ## Optional CRINGE/unlikelihood
	$(PY) -m h3hop.train_cringe --student $(STUDENT) --config configs/cringe.yaml --data data/pairs
rlvr:   ## RLVR: tests - lambda*CWE  (N1)
	$(PY) -m h3hop.train_rlvr  --student $(STUDENT) --config configs/rlvr.yaml   --prompts $(PROMPTS)

# ---------- bad-weight artifacts ----------
.PHONY: vector antiexpert
vector:    ## Build anti-pattern.gguf control vector from the contrast set
	$(PY) -m h3hop.build_control_vector --model artifacts/student-merged --config configs/control_vector.yaml \
	     --contrast data/contrast --out artifacts/anti-pattern.gguf

.PHONY: vector-cheat vector-stego
vector-cheat: ## Build anti-cheat.gguf steering direction (bad-weight extended to cheating)
	$(PY) -m h3hop.build_special_contrast
	$(PY) -m h3hop.build_control_vector --model artifacts/student-merged --config configs/control_vector.yaml \
	     --contrast data/contrast_cheat --out artifacts/anti-cheat.gguf
vector-stego: ## Build anti-stego.gguf steering direction (bad-weight extended to hidden content)
	$(PY) -m h3hop.build_special_contrast
	$(PY) -m h3hop.build_control_vector --model artifacts/student-merged --config configs/control_vector.yaml \
	     --contrast data/contrast_stego --out artifacts/anti-stego.gguf
antiexpert: ## Build the DExperts / in-MoE anti-expert (N2)
	$(PY) -m h3hop.build_anti_expert --model artifacts/student-merged --bad data/bad --out artifacts/bad-code-antiexpert

# ---------- eval ----------
.PHONY: eval
eval: ## Capability + bad-pattern + toggle test -> eval/report
	$(PY) eval/run_capability.py --model artifacts/student-merged --config configs/eval.yaml --out eval/results
	$(PY) eval/run_badpattern.py --model artifacts/student-merged --control-vector artifacts/anti-pattern.gguf --out eval/results
	$(PY) eval/report.py --in eval/results --out eval/report.md

.PHONY: integrity
integrity: ## Integrity detectors: self-test + score bad set + agent-trace monitor
	$(PY) eval/run_integrity.py --selftest
	$(PY) eval/run_integrity.py --code-file data/bad/bad.jsonl --out eval/results || true
	$(PY) eval/run_integrity.py --trace eval/traces/sample_trace.jsonl --out eval/results

# ---------- packaging ----------
.PHONY: gguf
gguf: ## Merge LoRA, convert to GGUF, quantize
	$(PY) -m h3hop.merge_lora --student $(STUDENT) --adapters artifacts/adapters --out artifacts/student-merged
	bash scripts/convert_to_gguf.sh artifacts/student-merged artifacts/your-coder-moe.Q4_K_M.gguf
