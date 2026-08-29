"""
inference.py
============

Interactive command-line front-end.

    python inference.py
    python inference.py --mode synonyms --temperature 0.6

The model is loaded once and reused for every prompt. Commands inside the loop:

    /enhance     switch to phrase-enhancement mode
    /synonyms    switch to synonym mode
    /explain     explain the most recent enhancement
    /temp 0.6    change the sampling temperature
    /help        show the commands
    /quit        exit
"""

from __future__ import annotations

import argparse
import sys

import c2_engine as engine

BANNER = """
============================================================
  C2 English Writing Assistant
  Local Llama 3 - QLoRA - no external API
============================================================
"""

HELP = """
Commands
  /enhance      Mode 1: rewrite a phrase at C1/C2 level
  /synonyms     Mode 2: sophisticated alternatives for one word
  /explain      Explain the lexical choices in the last rewrite
  /temp <v>     Set temperature (0 disables sampling)
  /tokens <n>   Set max new tokens
  /help         This message
  /quit         Exit
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C2 writing assistant (CLI).")
    parser.add_argument("--mode", choices=["enhance", "synonyms"], default="enhance")
    parser.add_argument("--prompt", "--text", default=None, help="One-shot prompt to process without entering interactive mode")
    parser.add_argument("--tone", default=None, help="Optional tone instruction for phrase enhancement")
    parser.add_argument("--explain", action="store_true", help="Include explanation of lexical/syntactic improvements")
    parser.add_argument("--base-model", default=None, help="Hugging Face model ID (default: meta-llama/Meta-Llama-3-8B-Instruct or open fallback)")
    parser.add_argument("--adapter-dir", default=engine.ADAPTER_DIR)
    parser.add_argument("--token", default=None, help="Hugging Face user authentication token for gated models")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.prompt:
        print(BANNER)
    model_name = args.base_model or engine.BASE_MODEL_ID
    if not args.prompt:
        print(f"Loading {model_name} (the first run downloads model weights)...")

    try:
        bundle = engine.load_model(
            base_model_id=args.base_model,
            adapter_dir=args.adapter_dir,
            token=args.token,
        )
    except Exception as exc:
        sys.exit(
            f"\nCould not load the model: {type(exc).__name__}: {exc}\n\n"
            "Common causes:\n"
            "  * not authenticated for gated repo -> pass `--token <hf_token>` or run `huggingface-cli login`\n"
            "  * or specify an open model with `--base-model Qwen/Qwen2.5-1.5B-Instruct`\n"
            "  * insufficient GPU VRAM\n"
        )

    for warning in bundle.warnings:
        if not args.prompt or "No trained adapter" not in warning:
            print(f"[!] {warning}")

    if not args.prompt:
        print(f"\n{bundle.status_line}")
        print(f"    device={bundle.device}  dtype={bundle.dtype}  "
              f"4-bit={'yes' if bundle.quantized else 'no'}")
        print(HELP)

    # One-shot CLI execution mode
    if args.prompt:
        prompt_text = args.prompt.strip()
        if args.mode == "enhance":
            messages = engine.build_enhance_messages(prompt_text, tone=args.tone)
            raw_out = engine.generate(
                bundle,
                messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
            c2_out = engine.clean_single_line(raw_out)
            print(f"\n[ORIGINAL]\n{prompt_text}\n\n[C2 VERSION]\n{c2_out}")
            if args.explain:
                exp_messages = engine.build_explanation_messages(prompt_text, c2_out)
                explanation = engine.generate(
                    bundle,
                    exp_messages,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                )
                print(f"\n[EXPLANATION]\n{explanation}")
        else:
            messages = engine.build_synonym_messages(prompt_text)
            raw_out = engine.generate(
                bundle,
                messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
            entries, note = engine.parse_synonyms(raw_out)
            print(f"\n[C2 ALTERNATIVES FOR \"{prompt_text}\"]\n")
            if entries:
                for entry in entries:
                    print(entry.term)
                    if entry.meaning:
                        print(f"  {entry.meaning}")
                    if entry.example:
                        print(f"  e.g. {entry.example}")
                    print()
                if note:
                    print(f"Note: {note}\n")
            else:
                print(raw_out)
        return

    mode = args.mode
    temperature = args.temperature
    max_new_tokens = args.max_new_tokens
    last_pair: tuple[str, str] | None = None

    while True:
        try:
            raw = input(f"[{mode}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        if not raw:
            continue

        if raw.startswith("/"):
            parts = raw.split()
            command = parts[0].lower()
            if command in ("/quit", "/exit", "/q"):
                print("Goodbye.")
                return
            if command == "/help":
                print(HELP)
                continue
            if command == "/enhance":
                mode = "enhance"
                continue
            if command == "/synonyms":
                mode = "synonyms"
                continue
            if command == "/temp" and len(parts) > 1:
                try:
                    temperature = max(0.0, float(parts[1]))
                    print(f"temperature = {temperature}")
                except ValueError:
                    print("Usage: /temp 0.7")
                continue
            if command == "/tokens" and len(parts) > 1:
                try:
                    max_new_tokens = max(16, int(parts[1]))
                    print(f"max_new_tokens = {max_new_tokens}")
                except ValueError:
                    print("Usage: /tokens 384")
                continue
            if command == "/explain":
                if last_pair is None:
                    print("Nothing to explain yet — enhance a phrase first.")
                    continue
                messages = engine.build_explanation_messages(*last_pair)
            else:
                print(f"Unknown command: {command}")
                continue
        elif mode == "enhance":
            messages = engine.build_enhance_messages(raw)
        else:
            if len(raw.split()) > 1:
                print("Synonym mode expects a single word. Use /enhance for phrases.")
                continue
            messages = engine.build_synonym_messages(raw)

        try:
            output = engine.generate(
                bundle,
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
        except Exception as exc:
            print(f"[!] Generation failed: {type(exc).__name__}: {exc}")
            continue

        if raw.startswith("/"):
            print(f"\n{output}\n")
        elif mode == "enhance":
            output = engine.clean_single_line(output)
            last_pair = (raw, output)
            print(f"\nC2 VERSION\n{output}\n")
        else:
            entries, note = engine.parse_synonyms(output)
            if entries:
                print()
                for entry in entries:
                    print(entry.term)
                    if entry.meaning:
                        print(f"  {entry.meaning}")
                    if entry.example:
                        print(f"  e.g. {entry.example}")
                    print()
                if note:
                    print(f"Note: {note}\n")
            else:
                print(f"\n{output}\n")


if __name__ == "__main__":
    main()
