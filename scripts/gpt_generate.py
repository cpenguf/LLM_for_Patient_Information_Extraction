#!/usr/bin/env python3
"""
Minimal NeMo Megatron GPT inference helper for Llama 3.1 and GatorTronLlama.

Inputs:
- Provide a .nemo checkpoint via `model.restore_from_path=...`.
- Provide prompts either directly (`inference.prompts=[...]`) or via a JSONL
  file of objects containing an `input` field
  (`inference.prompts_file=path/to/prompts.jsonl`).

Outputs:
- Writes JSONL with `{"input": "...", "output": "..."}` to
  `model.output_file` (default: outputs/inference/predictions.jsonl).

Example:
python scripts/gpt_generate.py \
  model.restore_from_path=/abs/path/to/llama3_1_8b.nemo \
  inference.prompts=["Extract medications from the text: ..."]
"""

import json
from pathlib import Path
from typing import List

from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf import DictConfig, OmegaConf


def _load_prompts(cfg: DictConfig) -> List[str]:
    if cfg.inference.prompts_file:
        path = Path(cfg.inference.prompts_file)
        logging.info(f"Loading prompts from JSONL: {path}")
        prompts = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                prompts.append(record["input"])
        return prompts
    return list(cfg.inference.prompts)


def _save_outputs(inputs: List[str], outputs: List[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for inp, out in zip(inputs, outputs):
            f.write(json.dumps({"input": inp, "output": out}, ensure_ascii=False) + "\n")
    logging.info(f"Wrote predictions to {output_path}")


@hydra_runner(config_path="../configs", config_name="megatron_gpt_generate_config")
def main(cfg: DictConfig) -> None:
    if cfg.model.restore_from_path in (None, "???"):
        raise ValueError("Set `model.restore_from_path` to a .nemo checkpoint.")

    prompts = _load_prompts(cfg)
    if not prompts:
        raise ValueError("No prompts provided. Set `inference.prompts` or `inference.prompts_file`.")

    logging.info("\n\n************** Inference configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    model = MegatronGPTSFTModel.restore_from(cfg.model.restore_from_path, trainer=None)
    logging.info(f"Loaded model from {cfg.model.restore_from_path}")

    responses = model.generate(
        input_prompts=prompts,
        tokens_to_generate=cfg.model.tokens_to_generate,
        all_probs=cfg.model.all_probs,
        temperature=cfg.model.temperature,
        top_k=cfg.model.top_k,
        top_p=cfg.model.top_p,
        repetition_penalty=cfg.model.repetition_penalty,
        add_BOS=cfg.model.add_BOS,
        add_EOS=cfg.model.add_EOS,
        min_tokens_to_generate=cfg.model.min_tokens_to_generate,
        compute_attention_mask=cfg.model.compute_attention_mask,
        greedy=cfg.model.greedy,
        random_seed=cfg.model.random_seed,
    )

    outputs = [resp["text"] for resp in responses]
    _save_outputs(prompts, outputs, Path(cfg.model.output_file))


if __name__ == "__main__":
    main()
