"""
WhatsApp export → ChatML fine-tuning dataset (system/user/assistant turns).

Usage:
    python data.py                                   # processes all .txt in WHATSAPP_CHATS_DIR
    python data.py chat1.txt chat2.txt               # process specific files
    python data.py chat1.txt -o my_finetune.jsonl    # custom output name
"""

import re
import json
import sys
import argparse
from datetime import datetime
from pathlib import Path

from config import (
    YOUR_NAME, SYSTEM_PROMPT, CONTEXT_WINDOW,
    SESSION_GAP_HOURS, WHATSAPP_CHATS_DIR, OUTPUT_DIR,
)

# WhatsApp uses a narrow no-break space (U+202F) between time and am/pm
PATTERN = re.compile(
    r"(\d{2}/\d{2}/\d{4}, \d{1,2}:\d{2}\u202f(?:am|pm)) - (.+?): (.+)"
)

SKIP_PHRASES = [
    "<Media omitted>",
    "This message was deleted",
    "end-to-end encrypted",
    "Missed voice call",
    "Missed video call",
    "This chat is with",
    "Tap to learn",
    "registered as a standard account",
]


def parse_time(t: str):
    try:
        return datetime.strptime(t, "%d/%m/%Y, %I:%M\u202f%p")
    except ValueError:
        return None


def big_gap(t1, t2) -> bool:
    if not t1 or not t2:
        return True
    return (t2 - t1).total_seconds() > SESSION_GAP_HOURS * 3600


def parse_messages(filepath: str) -> list:yeah add that, and the RAG.py code will be the same? can you edit that too so that I dont doxx anything and everything about me stays private like name and contacts names and all
    messages = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(PATTERN, line.strip())
            if not match:
                continue
            time_str, sender, text = match.group(1), match.group(2), match.group(3)
            text = text.strip()

            if any(skip in text for skip in SKIP_PHRASES):
                continue
            if text.startswith("http") and len(text.split()) == 1:
                continue

            messages.append({
                "sender": sender,
                "text": text,
                "time": parse_time(time_str),
            })
    return messages


def combine_consecutive(messages: list) -> list:
    combined = []
    i = 0
    while i < len(messages):
        cur = messages[i]
        texts = [cur["text"]]
        j = i + 1
        while (
            j < len(messages)
            and messages[j]["sender"] == cur["sender"]
            and not big_gap(messages[j - 1]["time"], messages[j]["time"])
        ):
            texts.append(messages[j]["text"])
            j += 1
        combined.append({
            "sender": cur["sender"],
            "text": " ".join(texts),
            "time": cur["time"],
        })
        i = j
    return combined


def split_sessions(messages: list) -> list[list]:
    sessions, current = [], [messages[0]]
    for i in range(1, len(messages)):
        if big_gap(messages[i - 1]["time"], messages[i]["time"]):
            sessions.append(current)
            current = [messages[i]]
        else:
            current.append(messages[i])
    sessions.append(current)
    return sessions


def build_dataset(sessions: list) -> list:
    examples = []
    for session in sessions:
        for i, msg in enumerate(session):
            if msg["sender"] != YOUR_NAME:
                continue

            context = session[max(0, i - CONTEXT_WINDOW) : i]
            if not context:
                continue

            turns = [{"role": "system", "content": SYSTEM_PROMPT}]
            for m in context:
                role = "assistant" if m["sender"] == YOUR_NAME else "user"
                turns.append({"role": role, "content": m["text"]})

            turns.append({"role": "assistant", "content": msg["text"]})
            examples.append({"messages": turns})
    return examples


def process_file(filepath: str) -> list:
    print(f"  Parsing: {Path(filepath).name}")

    raw = parse_messages(filepath)
    print(f"    Raw messages:    {len(raw)}")

    if not raw:
        return []

    messages = combine_consecutive(raw)
    print(f"    After combining: {len(messages)}")

    sessions = split_sessions(messages)
    print(f"    Sessions:        {len(sessions)}")

    examples = build_dataset(sessions)
    print(f"    Examples:        {len(examples)}")
    return examples


def main():
    parser = argparse.ArgumentParser(description="Build fine-tuning dataset")
    parser.add_argument("files", nargs="*", help="WhatsApp .txt exports")
    parser.add_argument("-o", "--output", default=None, help="Output JSONL file")
    args = parser.parse_args()

    if args.files:
        files = args.files
    else:
        chat_dir = Path(WHATSAPP_CHATS_DIR)
        if not chat_dir.exists():
            print(f"Error: directory '{WHATSAPP_CHATS_DIR}' not found.")
            return
        files = sorted(str(p) for p in chat_dir.glob("*.txt"))
        if not files:
            print(f"No .txt files found in {WHATSAPP_CHATS_DIR}/")
            return

    print(f"Found {len(files)} chat file(s)\n")

    all_examples = []
    for filepath in files:
        examples = process_file(filepath)
        all_examples.extend(examples)

    if not all_examples:
        print("\nNo training examples generated.")
        return

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else out_dir / "finetune_dataset.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(all_examples)} examples → {output_path}")

    lens = [len(ex["messages"][-1]["content"].split()) for ex in all_examples]
    print(f"\nReply length stats:")
    print(f"  avg: {sum(lens) / len(lens):.1f} words")
    print(f"  min: {min(lens)} words")
    print(f"  max: {max(lens)} words")


if __name__ == "__main__":
    main()
