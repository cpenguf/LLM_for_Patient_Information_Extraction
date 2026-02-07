#!/usr/bin/env bash
set -euo pipefail

# GatorTron relation extraction inference with MRC framework using ClinicalTransformerMRC
# Source reference: https://github.com/uf-hobi-informatics-lab/ClinicalTransformerMRC (ADE relation inference script)

CTMRC_ROOT=${CTMRC_ROOT:?set CTMRC_ROOT to the ClinicalTransformerMRC clone}
export PYTHONPATH="$PYTHONPATH:${CTMRC_ROOT}/src"

DATA_DIR=${DATA_DIR:-data/mrc_relation}
BERT_DIR=${BERT_DIR:?set to GatorTron checkpoint dir}
MODEL_CKPT=${MODEL_CKPT:?set to checkpoint .ckpt path from training}
HPARAMS_FILE=${HPARAMS_FILE:?set to hparams.yaml path from training}
OUTPUT_FN=${OUTPUT_FN:-outputs/gatortron_mrc_re/predictions.json}
DATA_SIGN=${DATA_SIGN:-drug_ADE_relation}

python "${CTMRC_ROOT}/src/inference/mrc_ner_inference.py" \
  --data_dir "${DATA_DIR}" \
  --bert_dir "${BERT_DIR}" \
  --max_length ${MAX_LEN:-512} \
  --model_ckpt "${MODEL_CKPT}" \
  --hparams_file "${HPARAMS_FILE}" \
  --output_fn "${OUTPUT_FN}" \
  --dataset_sign "${DATA_SIGN}"
