#!/usr/bin/env bash
set -euo pipefail

# GatorTron NER fine-tuning using ClinicalTransformerNER
# Source reference: https://github.com/uf-hobi-informatics-lab/ClinicalTransformerNER

# Required: set CTNER_ROOT to the ClinicalTransformerNER clone
CTNER_ROOT=${CTNER_ROOT:?set CTNER_ROOT to the ClinicalTransformerNER clone}

# Hugging Face model id for GatorTron (base/medium/large are valid)
MODEL_NAME=${MODEL_NAME:-UFNLP/gatortron-base}

# Data and output locations (BIO-formatted train/dev/test)
DATA_DIR=${DATA_DIR:-data/ner}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/gatortron_ner}

TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/train.txt}
DEV_FILE=${DEV_FILE:-${DATA_DIR}/dev.txt}
TEST_FILE=${TEST_FILE:-${DATA_DIR}/test.txt}
PRED_FILE=${PRED_FILE:-${OUTPUT_DIR}/predictions.txt}

mkdir -p "${OUTPUT_DIR}"

python "${CTNER_ROOT}/src/run_transformer_ner.py" \
  --model_type bert \
  --pretrained_model "${MODEL_NAME}" \
  --data_dir "${DATA_DIR}" \
  --train_file "${TRAIN_FILE}" \
  --dev_file "${DEV_FILE}" \
  --test_file "${TEST_FILE}" \
  --new_model_dir "${OUTPUT_DIR}" \
  --overwrite_model_dir \
  --predict_output_file "${PRED_FILE}" \
  --max_seq_length ${MAX_SEQ_LEN:-256} \
  --save_model_core \
  --do_train \
  --do_predict \
  --model_selection_scoring strict-f_score-1 \
  --train_batch_size ${TRAIN_BATCH_SIZE:-8} \
  --eval_batch_size ${EVAL_BATCH_SIZE:-8} \
  --train_steps ${TRAIN_STEPS:-500} \
  --learning_rate ${LR:-1e-5} \
  --num_train_epochs ${EPOCHS:-3} \
  --gradient_accumulation_steps ${GRAD_ACCUM:-1} \
  --warmup_ratio ${WARMUP_RATIO:-0.1} \
  --max_num_checkpoints ${MAX_CKPTS:-3} \
  --early_stop ${EARLY_STOP:-3} \
  --log_file "${OUTPUT_DIR}/train.log" \
  --progress_bar
