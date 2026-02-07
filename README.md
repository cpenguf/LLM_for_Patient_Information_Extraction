# A Study of Large Language Models for Patient Information Extraction

**Model Architecture, Fine-Tuning Strategy, and Multi-task Instruction Tuning**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange)](https://huggingface.co/)

## 📄 Overview

This repository contains the source code and experimental framework for the paper **"A Study of Large Language Models for Patient Information Extraction: Model Architecture, Fine-Tuning Strategy, and Multi-task Instruction Tuning"**.

This study provides a comprehensive evaluation of Large Language Models (LLMs) for clinical **Concept Extraction (CE)** and **Relation Extraction (RE)**. We benchmarked widely used encoder-based and decoder-based models across five diverse clinical datasets, evaluating:
* **Architectures:** Encoder-only (e.g., BERT, GatorTron) vs. Decoder-only (e.g., Llama 3.1, GatorTronLlama).
* **Fine-tuning:** Traditional Full Fine-Tuning vs. Parameter-Efficient Fine-Tuning (PEFT/LoRA).
* **Generalizability:** The impact of Multi-task Instruction Tuning on zero-shot and few-shot learning.

---

## 🏗️ Models Evaluated

We evaluated 8 distinct LLMs using two different architectures:

| Architecture | Model | Parameters | Context Window | Source |
| :--- | :--- | :--- | :--- | :--- |
| **Encoder-only** | **BERT-large** | 340M | 512 | Google |
| | **ClinicalBERT** | 110M | 512 | Alsentzer et al. |
| | **SpanBERT** | 340M | 512 | Facebook AI |
| | **GatorTron** | 345M / 9B | 512 | University of Florida |
| **Decoder-only** | **GatorTronGPT** | 5B / 20B | 2,048 | University of Florida |
| | **Meditron-7b** | 7B | 4,096 | EPFL |
| | **Llama 3.1** | 8B | 8,192 | Meta AI |
| | **GatorTronLlama** | 8B | 8,192 | University of Florida |

---

## 📂 Datasets

This study utilized five clinical benchmark datasets. Due to data use agreements, we cannot provide the data directly in this repo. Please refer to the official sources below:

1.  **2010 i2b2/VA Challenge:** [n2c2 DBMI](https://n2c2.dbmi.hms.harvard.edu)
2.  **2018 n2c2 (Track 2):** [n2c2 DBMI](https://n2c2.dbmi.hms.harvard.edu)
3.  **2022 n2c2 (Track 2):** [n2c2 DBMI](https://n2c2.dbmi.hms.harvard.edu)
4.  **RadGraph:** [PhysioNet](https://physionet.org/content/radgraph/1.0.0/)
5.  **UF Health (Internal):** Not publicly available due to patient privacy.

---

## 🧪 Training with NeMo

- Use `scripts/megatron_gpt_finetuning.py` with `configs/megatron_gpt_finetuning_config.yaml` for PEFT or full finetuning (LoRA, p-tuning, adapters) on Llama 3.1 or GatorTronLlama `.nemo` checkpoints.
- Example launchers mirror the original `/orange/yonghui.wu/chengpeng/red/instruct_multitask` recipes: `scripts/train_llama31_lora.sh` and `scripts/train_gatortronllama_lora.sh`. Override `BASE_MODEL`, `TRAIN_FILE`, `VALID_FILE`, and `EXP_DIR` to point at your resources.
- Datasets should be JSONL with `input` and `output` fields; update batch sizes and parallelism flags (`TP`, `PP`, `NODES`, `GPUS`) to match your hardware.
- GatorTron NER (ClinicalTransformerNER-based): `scripts/gatortron_ner_train.sh` and `scripts/gatortron_ner_infer.sh` wrap https://github.com/uf-hobi-informatics-lab/ClinicalTransformerNER. Set `CTNER_ROOT` to your clone and `MODEL_NAME` to the desired GatorTron checkpoint (e.g., `UFNLP/gatortron-base`).
- GatorTron Relation Extraction (ClinicalTransformerRelationExtraction-based): `scripts/gatortron_re_train.sh` and `scripts/gatortron_re_infer.sh`. Set `CTRE_ROOT` to your clone of https://github.com/uf-hobi-informatics-lab/ClinicalTransformerRelationExtraction and `MODEL_NAME` to the desired GatorTron checkpoint.
- GatorTron MRC (Machine Reading Comprehension) NER/RE: `scripts/gatortron_mrc_ner_train.sh` / `gatortron_mrc_ner_infer.sh` and `scripts/gatortron_mrc_re_train.sh` / `gatortron_mrc_re_infer.sh` wrap https://github.com/uf-hobi-informatics-lab/ClinicalTransformerMRC. Set `CTMRC_ROOT` to your clone and `BERT_DIR` to the GatorTron checkpoint dir; configure `DATA_DIR`, checkpoint paths, and MRC hyperparameters as needed.
- Data prep: `scripts/preprocess_clinical_ner.py` converts CoNLL/BIO `.tsv` splits (train/dev/test) for 2010 i2b2, 2018 n2c2, 2022 n2c2, and RadGraph into SFT JSONL (for GatorTron/GatorTronLLaMA) and MRC JSON (for ClinicalTransformerMRC). Point `--data_root` to your raw datasets and `--out_root` to the desired output.

## 🔎 Inference

- Run `scripts/gpt_generate.py` with `configs/megatron_gpt_generate_config.yaml` to load a `.nemo` checkpoint and generate responses from prompts.
- Convenience launchers (no hardcoded paths): `scripts/run_llama31_inference.sh` and `scripts/run_gatortronllama_inference.sh`. Set `BASE_MODEL`, optionally `PROMPTS` or `PROMPTS_FILE`, and `OUTPUT_FILE`.
- `PROMPTS_FILE` expects JSONL with an `input` field per line. Outputs are written as JSONL `{input, output}`.
