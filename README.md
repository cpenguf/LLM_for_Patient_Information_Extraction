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
