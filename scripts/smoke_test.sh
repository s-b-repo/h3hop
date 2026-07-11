#!/usr/bin/env bash
# Offline light-track smoke test: mock gen -> filter -> dataset -> pairs. No GPU, no API keys.
# Proves the whole data plumbing end to end (the heavy train/vector steps need a GPU node).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
MODE="${1:-}"

echo "== h3hop smoke test (light) =="
rm -rf data/raw data/bad_raw data/raw_tested data/raw_checked data/raw_clean \
       data/good data/bad data/pairs data/contrast

$PY -m h3hop.gen_good --prompts data/prompts/seed_tasks.jsonl --out data/raw --mock
$PY -m h3hop.gen_bad  --prompts data/prompts/seed_tasks.jsonl --out data/bad_raw --mock
$PY -m h3hop.filter.run_tests     --in data/raw          --out data/raw_tested
$PY -m h3hop.filter.static_checks --in data/raw_tested   --out data/raw_checked --rules eval/icd_rules
$PY -m h3hop.filter.decontaminate --in data/raw_checked  --out data/raw_clean --benchmarks humaneval,mbpp
$PY -m h3hop.filter.integrity_scan --in data/raw_clean --out data/raw_clean
$PY -m h3hop.build_dataset --in data/raw_clean --bad-in data/bad_raw --out data
$PY -m h3hop.build_pairs   --good data/good --bad data/bad --out-pairs data/pairs --out-contrast data/contrast

echo "-- sanity: bad set should have a high bad-pattern rate (needs semgrep for signal) --"
$PY eval/run_badpattern.py --score-file data/bad/bad.jsonl --out eval/results || true

echo "-- integrity detectors: self-test + agent-trace monitor --"
$PY eval/run_integrity.py --selftest
$PY eval/run_integrity.py --trace eval/traces/sample_trace.jsonl --out eval/results

echo
echo "== outputs =="
for f in data/good/train.jsonl data/good/val.jsonl data/bad/bad.jsonl data/pairs/pairs.jsonl data/contrast/contrast.jsonl; do
  [ -f "$f" ] && echo "  $(wc -l < "$f") lines  $f"
done
echo "== smoke test PASSED =="

if [ "$MODE" != "--light" ]; then
  echo
  echo "Next (heavy, GPU node): set OPENROUTER_API_KEY, then"
  echo "  make gen filter dataset pairs sft dpo vector eval STUDENT=configs/proto_student.yaml"
fi
