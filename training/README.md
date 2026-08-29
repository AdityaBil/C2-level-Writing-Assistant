# Optional QLoRA Fine-Tuning Pipeline

This directory contains the legacy local training and evaluation scripts for fine-tuning Llama-3-8B-Instruct with QLoRA for C2 English writing assistance.

> **Note:** The main application no longer requires this pipeline or any local GPU/model weights. It uses an external LLM API for inference. This folder is preserved strictly for offline experimentation.

## Directory Structure

- `data/`: Contains seed dataset (`c2_seed_data.jsonl`) of high-proficiency C1/C2 rewrites and contextual vocabulary pairs.
- `prep_data.py`: Preprocessing script that generates synthetic dataset expansion, reformats raw examples into Llama-3 instruction chat templates, and splits data into train/validation sets.
- `train.py`: QLoRA fine-tuning script utilizing Hugging Face `trl` (SFTTrainer), `peft` (LoRA config r=16, alpha=32), `bitsandbytes` (4-bit NF4), and `transformers`.
- `inference.py`: Local CLI inference script to run the fine-tuned adapter against the base model locally.
- `c2_engine.py`: Engine helper with prompt templates and 4-bit local model loading.
- `llama-3-c2-adapter/`: Output directory where trained LoRA adapter weights and tokenizer configuration are saved.

## Setup for Training (Requires NVIDIA GPU)

1. Create a dedicated virtual environment:
   ```bash
   python -m venv .venv-training
   source .venv-training/bin/activate  # Or on Windows: .venv-training\Scripts\activate
   ```
2. Install training dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare the dataset:
   ```bash
   python prep_data.py
   ```
4. Run training:
   ```bash
   python train.py
   ```
