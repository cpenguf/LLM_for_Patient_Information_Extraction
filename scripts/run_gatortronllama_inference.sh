#!/usr/bin/env bash
set -euo pipefail

# Example launcher for GatorTronLlama inference (LoRA/PEFT or base).
# Override the env vars to point at your resources; no local defaults are baked in.

BASE_MODEL=${BASE_MODEL:?set to your gatortronllama .nemo path}
OUTPUT_FILE=${OUTPUT_FILE:-outputs/inference/gatortronllama_predictions.jsonl}

# Either provide prompts inline or point to a JSONL with {"input": "..."} lines.
PROMPTS=${PROMPTS:-}
PROMPTS_FILE=${PROMPTS_FILE:-}

python scripts/gpt_generate.py \
  model.restore_from_path=${BASE_MODEL} \
  model.output_file=${OUTPUT_FILE} \
  inference.prompts="[${PROMPTS}]" \
  inference.prompts_file=${PROMPTS_FILE} \
  model.tokens_to_generate=${TOKENS:-128} \
  model.temperature=${TEMP:-0.7} \
  model.top_k=${TOPK:-50} \
  model.top_p=${TOPP:-0.9}
