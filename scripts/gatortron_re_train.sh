#!/usr/bin/env bash
set -euo pipefail

# GatorTron relation extraction fine-tuning using ClinicalTransformerRelationExtraction
# Source reference: https://github.com/uf-hobi-informatics-lab/ClinicalTransformerRelationExtraction

# Required: set CTRE_ROOT to the ClinicalTransformerRelationExtraction clone
CTRE_ROOT=${CTRE_ROOT:?set CTRE_ROOT to the ClinicalTransformerRelationExtraction clone}

MODEL_NAME=${MODEL_NAME:-UFNLP/gatortron-base}
DATA_DIR=${DATA_DIR:-data/re}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/gatortron_re}
PRED_FILE=${PRED_FILE:-${OUTPUT_DIR}/predictions.txt}

mkdir -p "${OUTPUT_DIR}"

python "${CTRE_ROOT}/src/relation_extraction.py" \
  --model_type bert \
  --data_format_mode ${DATA_FORMAT_MODE:-0} \
  --classification_scheme ${CLASSIFICATION_SCHEME:-2} \
  --pretrained_model "${MODEL_NAME}" \
  --data_dir "${DATA_DIR}" \
  --new_model_dir "${OUTPUT_DIR}" \
  --predict_output_file "${PRED_FILE}" \
  --overwrite_model_dir \
  --seed ${SEED:-13} \
  --max_seq_length ${MAX_SEQ_LEN:-512} \
  --cache_data \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_batch_size ${TRAIN_BATCH_SIZE:-8} \
  --eval_batch_size ${EVAL_BATCH_SIZE:-8} \
  --learning_rate ${LR:-1e-5} \
  --num_train_epochs ${EPOCHS:-3} \
  --gradient_accumulation_steps ${GRAD_ACCUM:-1} \
  --do_warmup \
  --warmup_ratio ${WARMUP_RATIO:-0.1} \
  --weight_decay ${WEIGHT_DECAY:-0} \
  --max_num_checkpoints ${MAX_CKPTS:-1} \
  --log_file "${OUTPUT_DIR}/train.log"
