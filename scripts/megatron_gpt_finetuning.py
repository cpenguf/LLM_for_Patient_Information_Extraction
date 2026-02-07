#!/usr/bin/env python3
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Thin, documented copy of the NeMo GPT PEFT fine-tuning entrypoint used in
`/orange/yonghui.wu/chengpeng/red/instruct_multitask`.

Key points:
- Supports LoRA, p-tuning, adapters, IA3, or full finetuning.
- Expects JSONL datasets with `input` and `output` fields (same format as the
  upstream scripts).
- The base `.nemo` checkpoint (Llama 3.1 or GatorTronLlama) must be supplied
  via `model.restore_from_path` on the command line.

Example:
python scripts/megatron_gpt_finetuning.py \
  name=llama3_1_8b_lora_demo \
  model.restore_from_path=/abs/path/to/llama3_1_8b.nemo \
  model.peft.peft_scheme=lora \
  model.data.train_ds.file_names=[data/ner/train.jsonl] \
  model.data.validation_ds.file_names=[data/ner/valid.jsonl] \
  exp_manager.exp_dir=outputs/experiments
"""

import torch.multiprocessing as mp
from omegaconf import OmegaConf

from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import (
    MegatronGPTSFTModel,
)
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronLMPPTrainerBuilder
from nemo.collections.nlp.parts.peft_config import PEFT_CONFIG_MAP
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

mp.set_start_method("spawn", force=True)


@hydra_runner(config_path="../configs", config_name="megatron_gpt_finetuning_config")
def main(cfg) -> None:
    if cfg.model.restore_from_path in (None, "???"):
        raise ValueError("Set `model.restore_from_path` to a .nemo checkpoint for Llama or GatorTronLlama.")

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    trainer = MegatronLMPPTrainerBuilder(cfg).create_trainer()
    exp_manager(trainer, cfg.exp_manager)

    model_cfg = MegatronGPTSFTModel.merge_cfg_with(cfg.model.restore_from_path, cfg)
    model = MegatronGPTSFTModel.restore_from(cfg.model.restore_from_path, model_cfg, trainer=trainer)
    peft_cfg_cls = PEFT_CONFIG_MAP.get(cfg.model.peft.peft_scheme)

    if cfg.model.peft.restore_from_path is not None:
        logging.info("Loading PEFT adapter weights from %s", cfg.model.peft.restore_from_path)
        model.load_adapters(cfg.model.peft.restore_from_path, peft_cfg_cls(model_cfg))
    elif peft_cfg_cls is not None:
        logging.info("Adding adapter weights for PEFT scheme=%s", cfg.model.peft.peft_scheme)
        model.add_adapter(peft_cfg_cls(model_cfg))
    else:
        logging.info("Running full finetuning (no PEFT scheme supplied).")
        logging.info(model.summarize())

    trainer.fit(model)


if __name__ == "__main__":
    main()
