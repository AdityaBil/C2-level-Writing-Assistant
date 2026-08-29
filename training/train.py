"""
train.py
========

QLoRA fine-tune of ``meta-llama/Meta-Llama-3-8B-Instruct`` on the C1/C2 writing
corpus produced by ``prep_data.py``.

    python train.py                       # defaults
    python train.py --epochs 12 --lr 1e-4

Pipeline:

1. load ``data/c2_seed_data.jsonl``
2. load the Llama-3 tokenizer and apply its chat template to each record
3. load the base model in 4-bit NF4
4. ``prepare_model_for_kbit_training`` (+ gradient checkpointing)
5. attach a LoRA adapter (r=16, alpha=32, dropout=0.05, q/k/v/o projections)
6. train with TRL's ``SFTTrainer``
7. save **only** the adapter to ``./llama-3-c2-adapter``

The base model is never merged with the adapter here — ``app.py`` and
``inference.py`` attach the adapter at load time instead.

Pinned against: transformers 4.44.2, trl 0.9.6, peft 0.12.0, accelerate 0.33.0.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch
from datasets import load_dataset
# pyrefly: ignore [missing-import]
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, set_seed
# pyrefly: ignore [missing-import]
from trl import SFTConfig, SFTTrainer

from c2_engine import (
    BASE_MODEL_ID,
    DATA_PATH,
    DEFAULT_OPEN_MODEL_ID,
    PROJECT_DIR,
    build_quantization_config,
    load_tokenizer,
    select_compute_dtype,
)

ADAPTER_OUTPUT_DIR = os.path.join(PROJECT_DIR, "llama-3-c2-adapter")
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "training-checkpoints")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune for C1/C2 writing.")
    parser.add_argument("--base-model", default=BASE_MODEL_ID)
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument("--output-dir", default=ADAPTER_OUTPUT_DIR)
    parser.add_argument("--epochs", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Conservative QLoRA learning rate.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def main() -> None:
    args = parse_args()
    set_all_seeds(args.seed)

    if not os.path.isfile(args.data_path):
        sys.exit(
            f"Dataset not found at {args.data_path}.\nRun `python prep_data.py` first."
        )

    cuda = torch.cuda.is_available()
    compute_dtype = select_compute_dtype()
    use_bf16 = compute_dtype is torch.bfloat16 and cuda

    target_model_id = args.base_model
    token = os.environ.get("HF_TOKEN")

    if not cuda:
        print("[train] Notice: No CUDA GPU detected. Falling back to standard CPU LoRA training (float32).")
        if target_model_id == BASE_MODEL_ID and not token:
            print(f"[train] Gated Llama-3 requested without HF token; using open model '{DEFAULT_OPEN_MODEL_ID}'.")
            target_model_id = DEFAULT_OPEN_MODEL_ID

    if cuda and not use_bf16:
        print("[train] bfloat16 unsupported on this GPU — using float16 instead.")

    # ----------------------------------------------------------------- data --
    print(f"[train] loading dataset from {args.data_path}")
    dataset = load_dataset("json", data_files=args.data_path, split="train")

    try:
        tokenizer = load_tokenizer(target_model_id, padding_side="right", token=token)
    except Exception as exc:
        if ("gated" in str(exc).lower() or "401" in str(exc) or "403" in str(exc)) and target_model_id != DEFAULT_OPEN_MODEL_ID:
            print(f"[train] Access to '{target_model_id}' requires authentication. Falling back to '{DEFAULT_OPEN_MODEL_ID}'.")
            target_model_id = DEFAULT_OPEN_MODEL_ID
            tokenizer = load_tokenizer(target_model_id, padding_side="right", token=None)
        else:
            raise exc

    def apply_template(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            )
        }

    dataset = dataset.map(apply_template, remove_columns=dataset.column_names)
    print(f"[train] {len(dataset)} examples ready")
    print("[train] sample record:\n" + dataset[0]["text"][:400] + "\n...")

    # ---------------------------------------------------------------- model --
    if cuda:
        print(f"[train] loading {target_model_id} in 4-bit NF4 QLoRA on CUDA")
        model = AutoModelForCausalLM.from_pretrained(
            target_model_id,
            quantization_config=build_quantization_config(compute_dtype),
            torch_dtype=compute_dtype,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            token=token if target_model_id != DEFAULT_OPEN_MODEL_ID else None,
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        optimizer = "paged_adamw_8bit"
    else:
        print(f"[train] loading {target_model_id} on CPU in float32")
        model = AutoModelForCausalLM.from_pretrained(
            target_model_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            token=token if target_model_id != DEFAULT_OPEN_MODEL_ID else None,
        )
        model.config.use_cache = False
        optimizer = "adamw_torch"

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ------------------------------------------------------------- training --
    training_args = SFTConfig(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=cuda,
        gradient_checkpointing_kwargs={"use_reentrant": False} if cuda else None,
        optim=optimizer,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.001,
        max_grad_norm=0.3,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=use_bf16,
        fp16=not use_bf16 and cuda,
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
        group_by_length=False,
        # --- SFT-specific ---
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
        dataset_kwargs={"add_special_tokens": False, "append_concat_token": False},
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    print(f"[train] starting training on {'CUDA' if cuda else 'CPU'} ({args.epochs} epochs)")
    trainer.train()

    # ----------------------------------------------------------------- save --
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[train] adapter successfully saved to {args.output_dir}")
    print("[train] launch the app with:  streamlit run app.py")


if __name__ == "__main__":
    main()
