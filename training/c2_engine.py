"""
c2_engine.py
============

Shared logic for the C2 English Writing Assistant:

* the system prompt and the per-mode user prompts
* loading Meta-Llama-3-8B-Instruct in 4-bit and attaching the QLoRA adapter
* text generation with the Llama-3 chat template
* light parsing of the synonym output for structured display

Both ``app.py`` (Streamlit) and ``inference.py`` (CLI) import from here so the
two front-ends behave identically.

All paths are resolved relative to this file, so the scripts work no matter
which directory you run them from.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --------------------------------------------------------------------------- #
# Paths and constants
# --------------------------------------------------------------------------- #

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTER_DIR = os.path.join(PROJECT_DIR, "llama-3-c2-adapter")
DATA_PATH = os.path.join(PROJECT_DIR, "data", "c2_seed_data.jsonl")

BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct")
DEFAULT_OPEN_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """You are an Elite C1/C2 English Writing and Vocabulary Assistant.

Your purpose is to help users express themselves with sophisticated,
precise, elegant, and natural English.

When the user provides a phrase or sentence:
Rewrite it at a C1/C2 level while preserving its original meaning,
intent, factual content, and emotional register.

Do not merely replace words with obscure synonyms.
Prefer contextual refinement, sophisticated syntax, idiomatic expression,
and precise vocabulary.

When the user provides a single word:
Provide several sophisticated C1/C2 alternatives.
For every alternative, explain its specific nuance, register,
connotation, or appropriate context.

Never treat synonyms as universally interchangeable.

Avoid pretentious, archaic, unnatural, or unnecessarily convoluted language.

The objective is not to make English complicated.
The objective is to make it precise, elegant, expressive, and intellectually mature."""


ENHANCE_INSTRUCTION = (
    "Rewrite the following text at a sophisticated C1/C2 level. Preserve its "
    "meaning, intent, factual content and emotional register. Do not add "
    "information that is not already present. Return only the rewritten text, "
    "with no preamble, no quotation marks and no commentary.\n\nText: {text}"
)

SYNONYM_INSTRUCTION = (
    "Provide {n} sophisticated C1/C2 alternatives for the word below.\n"
    "Format each alternative as three lines:\n"
    "1. the alternative in capital letters, alone on its line;\n"
    "2. one line explaining its precise nuance, register or typical context;\n"
    "3. a line beginning with 'Example:' containing one natural sentence.\n"
    "Separate alternatives with a blank line. Finish with a short paragraph "
    "beginning with 'Note:' that explains where these words are NOT "
    "interchangeable.\n\nWord: {word}"
)

EXPLANATION_INSTRUCTION = (
    "Here is an original sentence and a C1/C2 rewrite of it.\n\n"
    "Original: {original}\n"
    "Rewrite: {rewritten}\n\n"
    "In no more than four short sentences, explain the specific lexical and "
    "syntactic choices made in the rewrite and why each is an improvement in "
    "precision or register. Do not repeat the rewrite itself."
)


def build_enhance_messages(text: str, tone: Optional[str] = None) -> List[Dict[str, str]]:
    """Chat messages for MODE 1 (Enhance Phrase)."""
    user = ENHANCE_INSTRUCTION.format(text=text.strip())
    if tone:
        user += f"\n\nAdditional instruction: adjust the tone so that it is {tone.strip()}."
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_synonym_messages(word: str, n: int = 5) -> List[Dict[str, str]]:
    """Chat messages for MODE 2 (C2 Synonyms)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": SYNONYM_INSTRUCTION.format(n=n, word=word.strip())},
    ]


def build_explanation_messages(original: str, rewritten: str) -> List[Dict[str, str]]:
    """Chat messages for the optional explanation of an enhancement."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": EXPLANATION_INSTRUCTION.format(
                original=original.strip(), rewritten=rewritten.strip()
            ),
        },
    ]


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #


@dataclass
class ModelBundle:
    """Everything the front-ends need to run generation."""

    model: object
    tokenizer: object
    device: str
    dtype: str
    quantized: bool
    adapter_loaded: bool
    adapter_dir: str
    base_model_id: str
    warnings: List[str]

    @property
    def status_line(self) -> str:
        if self.adapter_loaded:
            return "🟢 Local model loaded (base + QLoRA adapter)"
        return f"🟡 Base model loaded ({self.base_model_id}) — adapter not trained yet"


def adapter_is_trained(adapter_dir: str = ADAPTER_DIR) -> bool:
    """True when ``adapter_dir`` contains a saved PEFT adapter."""
    return os.path.isfile(os.path.join(adapter_dir, "adapter_config.json"))


def select_compute_dtype() -> torch.dtype:
    """bfloat16 where the GPU supports it, float16 on older GPUs, float32 on CPU."""
    if not torch.cuda.is_available():
        return torch.float32
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:  # pragma: no cover - very old torch/driver combinations
        pass
    return torch.float16


def build_quantization_config(compute_dtype: torch.dtype) -> BitsAndBytesConfig:
    """The 4-bit NF4 configuration used for both training and inference."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer(
    base_model_id: str = BASE_MODEL_ID,
    padding_side: str = "left",
    token: Optional[str] = None,
):
    """Load tokenizer with a usable padding token and optional HF authentication token."""
    token = token or os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = padding_side
    return tokenizer


def load_model(
    base_model_id: Optional[str] = None,
    adapter_dir: str = ADAPTER_DIR,
    allow_cpu_fallback: bool = True,
    token: Optional[str] = None,
) -> ModelBundle:
    """
    Load the base model (4-bit on CUDA) and attach the LoRA adapter if present.

    Includes automatic fallback to an open model if the requested model is gated
    and no authentication token is provided.
    """
    target_model_id = base_model_id or BASE_MODEL_ID
    auth_token = token or os.environ.get("HF_TOKEN")
    warnings: List[str] = []

    # Attempt to load requested model; fall back to open model if access gated/unauthorized
    try:
        tokenizer = load_tokenizer(target_model_id, padding_side="left", token=auth_token)
    except Exception as exc:
        is_gated_error = "gated repo" in str(exc).lower() or "401" in str(exc) or "403" in str(exc)
        if is_gated_error and target_model_id != DEFAULT_OPEN_MODEL_ID:
            warnings.append(
                f"Access to '{target_model_id}' requires authentication or acceptance of model license terms. "
                f"Falling back to open model '{DEFAULT_OPEN_MODEL_ID}'. "
                "Provide an HF Token in the sidebar to use Llama-3."
            )
            target_model_id = DEFAULT_OPEN_MODEL_ID
            tokenizer = load_tokenizer(target_model_id, padding_side="left", token=None)
        else:
            raise exc

    compute_dtype = select_compute_dtype()
    cuda = torch.cuda.is_available()

    if cuda:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                target_model_id,
                quantization_config=build_quantization_config(compute_dtype),
                torch_dtype=compute_dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
                token=auth_token if target_model_id != DEFAULT_OPEN_MODEL_ID else None,
            )
            device = "cuda"
            quantized = True
            if compute_dtype is torch.float16:
                warnings.append(
                    "This GPU does not support bfloat16; falling back to float16."
                )
        except Exception as exc:
            # Fallback to CPU or open model if CUDA quantization fails
            if "gated repo" in str(exc).lower() or "401" in str(exc) or "403" in str(exc):
                target_model_id = DEFAULT_OPEN_MODEL_ID
                tokenizer = load_tokenizer(target_model_id, padding_side="left")
                model = AutoModelForCausalLM.from_pretrained(
                    target_model_id,
                    quantization_config=build_quantization_config(compute_dtype),
                    torch_dtype=compute_dtype,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
                device = "cuda"
                quantized = True
                warnings.append(f"Fell back to open model '{DEFAULT_OPEN_MODEL_ID}'.")
            else:
                raise exc
    else:
        if not allow_cpu_fallback:
            raise RuntimeError(
                "No CUDA device found. 4-bit inference with bitsandbytes requires "
                "an NVIDIA GPU."
            )
        warnings.append(
            "No CUDA device found. Loading model on CPU in float32."
        )
        model = AutoModelForCausalLM.from_pretrained(
            target_model_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            token=auth_token if target_model_id != DEFAULT_OPEN_MODEL_ID else None,
        )
        device = "cpu"
        quantized = False

    adapter_loaded = False
    if adapter_is_trained(adapter_dir):
        try:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_dir)
            adapter_loaded = True
        except Exception as exc:  # pragma: no cover - depends on user's files
            warnings.append(
                f"An adapter was found in '{adapter_dir}' but could not be attached "
                f"({type(exc).__name__}: {exc}). Running on the base model instead."
            )
    else:
        warnings.append(
            f"No trained adapter in '{adapter_dir}'. Running on the base instruct "
            "model. Run `python train.py` to fine-tune the QLoRA adapter."
        )

    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True

    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=str(compute_dtype).replace("torch.", ""),
        quantized=quantized,
        adapter_loaded=adapter_loaded,
        adapter_dir=adapter_dir,
        base_model_id=target_model_id,
        warnings=warnings,
    )


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def _terminators(tokenizer) -> List[int]:
    ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    for special in ["<|eot_id|>", "<|im_end|>", "<|endoftext|>"]:
        token_id = tokenizer.convert_tokens_to_ids(special)
        if isinstance(token_id, int) and token_id >= 0 and token_id not in ids:
            ids.append(token_id)
    return [i for i in ids if isinstance(i, int) and i >= 0]


@torch.inference_mode()
def generate(
    bundle: ModelBundle,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 320,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> str:
    """Run one generation and return only the newly produced text."""
    tokenizer = bundle.tokenizer
    model = bundle.model

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = input_ids.to(model.device)
    attention_mask = torch.ones_like(input_ids)

    do_sample = temperature is not None and temperature > 0.0
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=int(max_new_tokens),
        do_sample=do_sample,
        temperature=float(temperature) if do_sample else None,
        top_p=float(top_p) if do_sample else None,
        repetition_penalty=float(repetition_penalty),
        eos_token_id=_terminators(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
    )

    generated = output[0][input_ids.shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text.strip()


def clean_single_line(text: str) -> str:
    """Tidy a rewritten phrase: drop wrapping quotes and leading labels."""
    cleaned = text.strip()
    cleaned = re.sub(r"^(C2 version|Rewrite|Rewritten)\s*:\s*", "", cleaned, flags=re.I)
    if len(cleaned) >= 2 and cleaned[0] in "\"'“‘" and cleaned[-1] in "\"'”’":
        cleaned = cleaned[1:-1].strip()
    return cleaned


# --------------------------------------------------------------------------- #
# Synonym output parsing
# --------------------------------------------------------------------------- #


@dataclass
class SynonymEntry:
    term: str
    meaning: str
    example: str = ""


def parse_synonyms(text: str) -> Tuple[List[SynonymEntry], str]:
    """
    Parse the model's synonym output into entries plus a trailing note.

    Returns ``([], "")`` when the output does not match the expected shape, so
    callers can fall back to displaying the raw text.
    """
    entries: List[SynonymEntry] = []
    note = ""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]

    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        if lines[0].lower().startswith("note:"):
            note = " ".join(lines).split(":", 1)[1].strip()
            continue

        head = re.sub(r"^[\-\*\d\.\)\s]+", "", lines[0]).strip().strip("*").strip()
        letters = [c for c in head if c.isalpha()]
        if not letters or not all(c.isupper() for c in letters) or len(head) > 40:
            continue

        meaning, example = "", ""
        for line in lines[1:]:
            if line.lower().startswith("example:"):
                example = line.split(":", 1)[1].strip()
            elif not meaning:
                meaning = line
        entries.append(SynonymEntry(term=head, meaning=meaning, example=example))

    return entries, note
