#!/usr/bin/env bash
set -euo pipefail

# Minimal, container-agnostic launcher for Llama 3.1 SFT/PEFT with NeMo.
# Override the env vars below to point at your resources.

BASE_MODEL=${BASE_MODEL:-/abs/path/to/llama3_1_8b.nemo}
TRAIN_FILE=${TRAIN_FILE:-data/ner/train.jsonl}
VALID_FILE=${VALID_FILE:-data/ner/valid.jsonl}
EXP_DIR=${EXP_DIR:-outputs/experiments}
RUN_NAME=${RUN_NAME:-llama3_1_8b_lora}

TP=${TP:-1}
PP=${PP:-1}
NODES=${NODES:-1}
GPUS=${GPUS:-1}
GLOBAL_BATCH=${GLOBAL_BATCH:-64}
MICRO_BATCH=${MICRO_BATCH:-1}
MAX_STEPS=${MAX_STEPS:-300}
LR=${LR:-1e-4}
VAL_GEN=${VAL_GEN:-128}
TEST_GEN=${TEST_GEN:-128}

python scripts/megatron_gpt_finetuning.py \
  name=${RUN_NAME} \
  model.restore_from_path=${BASE_MODEL} \
  model.peft.peft_scheme=lora \
  model.peft.lora_tuning.adapter_dim=256 \
  model.peft.lora_tuning.adapter_dropout=0.2 \
  model.data.train_ds.file_names=["${TRAIN_FILE}"] \
  model.data.train_ds.concat_sampling_probabilities=[1.0] \
  model.data.validation_ds.file_names=["${VALID_FILE}"] \
  model.data.train_ds.max_seq_length=4096 \
  model.data.validation_ds.max_seq_length=4096 \
  model.data.validation_ds.global_batch_size=${GLOBAL_BATCH} \
  model.global_batch_size=${GLOBAL_BATCH} \
  model.micro_batch_size=${MICRO_BATCH} \
  model.data.validation_ds.tokens_to_generate=${VAL_GEN} \
  model.data.test_ds.tokens_to_generate=${TEST_GEN} \
  model.tensor_model_parallel_size=${TP} \
  model.pipeline_model_parallel_size=${PP} \
  trainer.num_nodes=${NODES} \
  trainer.devices=${GPUS} \
  trainer.max_steps=${MAX_STEPS} \
  model.optim.lr=${LR} \
  exp_manager.exp_dir=${EXP_DIR} \
  exp_manager.create_checkpoint_callback=true
