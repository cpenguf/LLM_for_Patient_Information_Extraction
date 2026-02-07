#!/usr/bin/env bash
set -euo pipefail

# GatorTron MRC-style NER fine-tuning using ClinicalTransformerMRC
# Source reference: https://github.com/uf-hobi-informatics-lab/ClinicalTransformerMRC

CTMRC_ROOT=${CTMRC_ROOT:?set CTMRC_ROOT to the ClinicalTransformerMRC clone}
export PYTHONPATH="$PYTHONPATH:${CTMRC_ROOT}/src"

BERT_DIR=${BERT_DIR:?set to GatorTron checkpoint dir (e.g., /path/to/gatortron-base)}
DATA_DIR=${DATA_DIR:-data/mrc_ner}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/gatortron_mrc_ner}

mkdir -p "${OUTPUT_DIR}"

python "${CTMRC_ROOT}/src/train/mrc_ner_trainer.py" \
  --data_dir "${DATA_DIR}" \
  --model_type ${MODEL_TYPE:-megatron} \
  --bert_config_dir "${BERT_DIR}" \
  --max_length ${MAX_LEN:-512} \
  --batch_size ${BATCH_SIZE:-4} \
  --gpus="${GPUS:-1}" \
  --precision=${PRECISION:-16} \
  --progress_bar_refresh_rate 1 \
  --lr ${LR:-2e-5} \
  --val_check_interval ${VAL_CHECK:-0.2} \
  --accumulate_grad_batches ${GRAD_ACC:-1} \
  --default_root_dir "${OUTPUT_DIR}" \
  --mrc_dropout ${MRC_DROPOUT:-0.3} \
  --bert_dropout ${BERT_DROPOUT:-0.1} \
  --max_epochs ${EPOCHS:-5} \
  --span_loss_candidates ${SPAN_CANDIDATES:-pred_and_gold} \
  --weight_span ${WEIGHT_SPAN:-0.1} \
  --warmup_steps ${WARMUP_STEPS:-0} \
  --distributed_backend=${BACKEND:-ddp} \
  --gradient_clip_val ${GRAD_CLIP:-1.0} \
  --weight_decay ${WEIGHT_DECAY:-0.01} \
  --optimizer ${OPTIMIZER:-adamw} \
  --lr_scheduler ${LR_SCHEDULER:-polydecay} \
  --classifier_intermediate_hidden_size ${INTER_HIDDEN:-2048} \
  --lr_mini ${LR_MINI:-1e-7}
