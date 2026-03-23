"""
Optional second-pass cleaner: re-chunks RAG data using a sliding window.

Takes the cleaned RAG JSONL and produces smaller, overlapping chunks
with extracted metadata. Useful if your chunks are too large or you
want more granular retrieval.

Usage:
    python cleanv2.py                                  # processes output/rag_cleaned.jsonl
    python cleanv2.py input.jsonl -o rechunked.jsonl
"""

import json
import re
import uuid
import argparse
from pathlib import Path

from config import YOUR_NAME, DISPLAY_NAME, OUTPUT_DIR

# ── CHUNKING PARAMS ───────────────────────────────────────────────────────────
CHUNK_SIZE = 10      # messages per chunk
OVERLAP = 3          # overlap between chunks
MIN_MESSAGES = 4     # minimum messages to keep a chunk


def clean_line(line: str) -> str:
    line = line.strip()
    if not line:
        return ""
    if "deleted this message" in line.lower():
        return ""
    # normalize your name to "You"
    line = re.sub(re.escape(YOUR_NAME) + r":", "You:", line)
    line = re.sub(re.escape(DISPLAY_NAME) + r":", "You:", line)
    line = re.sub(r"\[You\]:", "You:", line)
    line = re.sub(r"\[(.*?)\]:", r"\1:", line)
    return line


def split_messages(text: str) -> list[str]:
    return [c for line in text.split("\n") if (c := clean_line(line))]


def chunk_messages(messages: list[str]) -> list[list[str]]:
    chunks = []
    step = CHUNK_SIZE - OVERLAP
    for i in range(0, len(messages), step):
        chunk = messages[i : i + CHUNK_SIZE]
        if len(chunk) >= MIN_MESSAGES:
            chunks.append(chunk)
    return chunks


def extract_your_messages(messages: list[str]) -> list[str]:
    result = []
    for m in messages:
        if m.startswith("You:"):
            content = m.split(":", 1)[1].strip()
            if content:
                result.append(content)
    return result


def is_valid_chunk(chunk: list[str], your_msgs: list[str]) -> bool:
    text_block = "\n".join(chunk)
    if ":" not in text_block:
        return False
    if len(your_msgs) == 0:
        return False
    # remove code-heavy chunks
    if "import " in text_block or "```" in text_block:
        return False
    # remove list-like chunks
    lines_without_colon = [l for l in chunk if ":" not in l]
    if len(lines_without_colon) > len(chunk) * 0.5:
        return False
    return True


def get_year(date_str: str) -> int:
    try:
        return int(date_str[:4])
    except Exception:
        return 2025


def process(input_path: Path, output_path: Path):
    output = []

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)
            raw_text = item.get("text", "")
            metadata = item.get("metadata", {})

            messages = split_messages(raw_text)
            if len(messages) < MIN_MESSAGES:
                continue

            chunks = chunk_messages(messages)

            for chunk in chunks:
                your_msgs = extract_your_messages(chunk)
                if not is_valid_chunk(chunk, your_msgs):
                    continue

                text_block = "\n".join(chunk)
                year = get_year(metadata.get("date", "2025"))

                new_item = {
                    "id": str(uuid.uuid4()),
                    "text": text_block,
                    "metadata": {
                        "contact": metadata.get("contact", ""),
                        "date": metadata.get("date", ""),
                        "language": metadata.get("language", ""),
                        "message_count": len(chunk),
                        "your_messages": your_msgs,
                        "era": "recent" if year >= 2024 else "old",
                        "importance": 2 if year >= 2024 else 1,
                    },
                }
                output.append(new_item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {len(output)} rechunked entries → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Sliding-window rechunker")
    parser.add_argument("input", nargs="?", default=None, help="Input JSONL file")
    parser.add_argument("-o", "--output", default=None, help="Output JSONL file")
    args = parser.parse_args()

    out_dir = Path(OUTPUT_DIR)
    input_path = Path(args.input) if args.input else out_dir / "rag_cleaned.jsonl"
    output_path = Path(args.output) if args.output else out_dir / "rag_rechunked.jsonl"

    if not input_path.exists():
        print(f"Error: {input_path} not found. Run the main pipeline first.")
        return

    process(input_path, output_path)


if __name__ == "__main__":
    main()
