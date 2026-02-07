#!/usr/bin/env bash
set -euo pipefail

# GatorTron NER inference using a model fine-tuned with ClinicalTransformerNER
# Source reference: https://github.com/uf-hobi-informatics-lab/ClinicalTransformerNER

CTNER_ROOT=${CTNER_ROOT:?set CTNER_ROOT to the ClinicalTransformerNER clone}

# Use the fine-tuned model directory produced by gatortron_ner_train.sh
MODEL_DIR=${MODEL_DIR:-outputs/gatortron_ner}
MODEL_NAME=${MODEL_NAME:-UFNLP/gatortron-base}  # tokenizer/config source

# Single-file prediction (BIO-formatted)
PRED_DATA_DIR=${PRED_DATA_DIR:-data/ner}
TEST_FILE=${TEST_FILE:-${PRED_DATA_DIR}/test.txt}
PRED_FILE=${PRED_FILE:-${MODEL_DIR}/predictions.txt}

python "${CTNER_ROOT}/src/run_transformer_ner.py" \
  --model_type bert \
  --pretrained_model "${MODEL_NAME}" \
  --data_dir "${PRED_DATA_DIR}" \
  --test_file "${TEST_FILE}" \
  --existing_model_dir "${MODEL_DIR}" \
  --predict_output_file "${PRED_FILE}" \
  --max_seq_length ${MAX_SEQ_LEN:-256} \
  --eval_batch_size ${EVAL_BATCH_SIZE:-8} \
  --do_predict \
  --model_selection_scoring strict-f_score-1 \
  --log_file "${MODEL_DIR}/infer.log" \
  --progress_bar
