"""
Run the full pipeline in one command.

Usage:
    python run_all.py              # RAG pipeline only
    python run_all.py --finetune   # RAG + fine-tuning dataset
"""

import argparse
import subprocess
import sys
from pathlib import Path

from config import WHATSAPP_CHATS_DIR, DISCORD_CHATS_DIR, OUTPUT_DIR


def run(cmd: str, label: str):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}\n")
    result = subprocess.run([sys.executable] + cmd.split(), capture_output=False)
    if result.returncode != 0:
        print(f"\n  Warning: {label} exited with code {result.returncode}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run the full data pipeline")
    parser.add_argument(
        "--finetune", action="store_true",
        help="Also build the fine-tuning dataset",
    )
    args = parser.parse_args()

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    has_whatsapp = Path(WHATSAPP_CHATS_DIR).exists() and list(Path(WHATSAPP_CHATS_DIR).glob("*.txt"))
    has_discord = Path(DISCORD_CHATS_DIR).exists() and list(Path(DISCORD_CHATS_DIR).glob("*.txt"))

    if not has_whatsapp and not has_discord:
        print("No chat files found!")
        print(f"  Put WhatsApp .txt exports in:  {WHATSAPP_CHATS_DIR}/")
        print(f"  Put Discord .txt exports in:   {DISCORD_CHATS_DIR}/")
        return

    # Step 1: Parse
    if has_whatsapp:
        run("wparser.py", "Step 1a: Parse WhatsApp chats")
    if has_discord:
        run("discordparser.py", "Step 1b: Parse Discord chats")

    # Step 2: Combine
    run("combine.py", "Step 2: Combine all sources")

    # Step 3: Clean
    run("clean.py", "Step 3: Clean combined dataset")

    print(f"\n{'=' * 60}")
    print(f"  RAG dataset ready: {out_dir / 'rag_cleaned.jsonl'}")
    print(f"{'=' * 60}")

    # Optional: Fine-tuning dataset
    if args.finetune:
        if has_whatsapp:
            run("data.py", "Step 4: Build fine-tuning dataset")
        else:
            print("\nSkipping fine-tune: no WhatsApp chats found (Discord not yet supported for fine-tune)")


if __name__ == "__main__":
    main()
