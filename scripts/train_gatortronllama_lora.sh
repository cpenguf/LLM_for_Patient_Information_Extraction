#!/usr/bin/env bash
set -euo pipefail

# Launcher tuned for GatorTronLlama LoRA fine-tuning with NeMo Megatron.
# Defaults mirror the settings seen in /orange/yonghui.wu/chengpeng/red/instruct_multitask.

BASE_MODEL=${BASE_MODEL:-/abs/path/to/gatortronllama3_1_8b.nemo}
TRAIN_FILE=${TRAIN_FILE:-data/ner/train.jsonl}
VALID_FILE=${VALID_FILE:-data/ner/valid.jsonl}
EXP_DIR=${EXP_DIR:-outputs/experiments}
RUN_NAME=${RUN_NAME:-gatortronllama3_1_8b_lora}

TP=${TP:-4}
PP=${PP:-4}
NODES=${NODES:-2}
GPUS=${GPUS:-8}
GLOBAL_BATCH=${GLOBAL_BATCH:-16}
MICRO_BATCH=${MICRO_BATCH:-1}
MAX_STEPS=${MAX_STEPS:-300}
LR=${LR:-1e-4}
SEQ_LEN=${SEQ_LEN:-4096}
VAL_GEN=${VAL_GEN:-1024}
TEST_GEN=${TEST_GEN:-1024}

python scripts/megatron_gpt_finetuning.py \
  name=${RUN_NAME} \
  model.restore_from_path=${BASE_MODEL} \
  model.peft.peft_scheme=lora \
  model.peft.lora_tuning.adapter_dim=256 \
  model.peft.lora_tuning.adapter_dropout=0.2 \
  model.data.train_ds.file_names=["${TRAIN_FILE}"] \
  model.data.train_ds.concat_sampling_probabilities=[1.0] \
  model.data.validation_ds.file_names=["${VALID_FILE}"] \
  model.data.train_ds.max_seq_length=${SEQ_LEN} \
  model.data.train_ds.truncation_method=left \
  model.data.validation_ds.max_seq_length=${SEQ_LEN} \
  model.data.validation_ds.truncation_method=left \
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
