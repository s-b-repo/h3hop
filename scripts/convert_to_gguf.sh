#!/usr/bin/env bash
# Convert a merged HF model dir to a quantized GGUF via llama.cpp.
# Usage: scripts/convert_to_gguf.sh <hf_model_dir> <out.gguf> [quant=Q4_K_M]
# Requires a built llama.cpp; set LLAMA_CPP to its path (default: ~/llama.cpp).
set -euo pipefail
SRC="${1:?usage: convert_to_gguf.sh <hf_model_dir> <out.gguf> [quant]}"
OUT="${2:?missing output .gguf path}"
QUANT="${3:-Q4_K_M}"
LLAMA="${LLAMA_CPP:-$HOME/llama.cpp}"

F16="${OUT%.gguf}.f16.gguf"
python3 "$LLAMA/convert_hf_to_gguf.py" "$SRC" --outfile "$F16" --outtype f16
"$LLAMA/llama-quantize" "$F16" "$OUT" "$QUANT"
rm -f "$F16"
echo "wrote $OUT  ($QUANT)"
