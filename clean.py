"""
Clean RAG chunks — remove artifacts, spam, oversized chunks.

Usage:
    python clean.py                                    # cleans output/rag_combined.jsonl
    python clean.py input.jsonl                        # clean a specific file
    python clean.py input.jsonl -o cleaned_output.jsonl
"""

import json
import re
import argparse
from pathlib import Path

from config import MAX_CHUNK_CHARS, MIN_CHUNK_CHARS, OUTPUT_DIR

# ── ARTIFACT PATTERNS ─────────────────────────────────────────────────────────

ARTIFACT_RES = [
    re.compile(r"\S+@\S+\.\S+"),
    re.compile(r"https?://\S+"),
    re.compile(r"\{Attachments\}"),
    re.compile(r"<Media omitted>"),
    re.compile(r"cdn\.discordapp\.com\S*"),
    re.compile(r"@\w{3,}"),
]

SPAM_RE = re.compile(
    r":[a-zA-Z0-9_]+:"
    r"|[\U0001F000-\U0001FFFF]"
    r"|[\u1ABC-\u1ABF]"
    r"|[\u10A0-\u10FF]"
    r"|[\u1B00-\u1CFF]"
    r"|[\u200B-\u200D\uFEFF]"
)


def clean_text(text: str) -> str:
    for pattern in ARTIFACT_RES:
        text = pattern.sub("", text)
    text = re.sub(r":[a-zA-Z0-9_]+:", "", text)
    text = re.sub(r"[\u1ABC-\u1ABF\u10A0-\u10FF\u1B00-\u1CFF\u200B-\u200D\uFEFF]", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [l for l in text.split("\n") if l.strip() and re.search(r"\w", l.strip())]
    return "\n".join(lines).strip()


def is_spam(text: str) -> bool:
    content = re.sub(r"\[.+?\]:\s*", "", text)
    leftover = SPAM_RE.sub("", content).strip()
    leftover = re.sub(r"[\s\W]", "", leftover)
    return len(leftover) == 0


def split_chunk(chunk: dict) -> list[dict]:
    lines = chunk["text"].split("\n")
    meta = chunk["metadata"]
    other_senders = set(meta.get("other_senders", []))

    sub_chunks, current_lines, current_len, part = [], [], 0, 0

    def flush():
        nonlocal part
        text_block = "\n".join(current_lines).strip()
        if not text_block or len(text_block) < MIN_CHUNK_CHARS:
            return
        your_msgs = []
        for l in current_lines:
            if "]: " not in l:
                continue
            speaker = l.split("]: ", 1)[0].lstrip("[")
            if speaker not in other_senders or l.startswith("[You]:"):
                your_msgs.append(l.split("]: ", 1)[1])
        if not your_msgs:
            return
        sub_chunks.append({
            "id": f"{chunk['id']}_p{part}",
            "text": text_block,
            "metadata": {**meta, "part": part, "your_messages": your_msgs},
        })
        part += 1

    for line in lines:
        line_len = len(line) + 1
        if current_len + line_len > MAX_CHUNK_CHARS and current_lines:
            flush()
            current_lines, current_len = [], 0
        current_lines.append(line)
        current_len += line_len

    flush()
    return sub_chunks if sub_chunks else [chunk]


def process_chunk(chunk: dict) -> list[dict]:
    meta = chunk.get("metadata", {})
    cleaned_text = clean_text(chunk.get("text", ""))
    cleaned_your_msgs = [
        m for msg in meta.get("your_messages", []) if (m := clean_text(msg)) and len(m) > 2
    ]

    chunk = {
        **chunk,
        "text": cleaned_text,
        "metadata": {**meta, "your_messages": cleaned_your_msgs},
    }

    if len(cleaned_text) < MIN_CHUNK_CHARS:
        return []
    if is_spam(cleaned_text):
        return []
    if not cleaned_your_msgs:
        return []
    if len(cleaned_text) > MAX_CHUNK_CHARS:
        return split_chunk(chunk)
    return [chunk]


def main():
    parser = argparse.ArgumentParser(description="Clean RAG chunks")
    parser.add_argument("input", nargs="?", default=None, help="Input JSONL file")
    parser.add_argument("-o", "--output", default=None, help="Output JSONL file")
    args = parser.parse_args()

    out_dir = Path(OUTPUT_DIR)
    input_path = Path(args.input) if args.input else out_dir / "rag_combined.jsonl"
    output_path = Path(args.output) if args.output else out_dir / "rag_cleaned.jsonl"

    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return

    raw_chunks = [json.loads(l) for l in open(input_path, encoding="utf-8") if l.strip()]
    print(f"Input:  {len(raw_chunks)} chunks from {input_path}")

    output_chunks = []
    stats = {"no_reply": 0, "too_short": 0, "spam": 0, "split": 0, "artifacts": 0}

    for chunk in raw_chunks:
        orig = chunk.get("text", "")
        results = process_chunk(chunk)

        if not results:
            meta = chunk.get("metadata", {})
            if is_spam(clean_text(orig)):
                stats["spam"] += 1
            elif not meta.get("your_messages"):
                stats["no_reply"] += 1
            else:
                stats["too_short"] += 1
            continue

        if len(results) > 1:
            stats["split"] += 1
        if clean_text(orig) != orig:
            stats["artifacts"] += 1

        output_chunks.extend(results)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in output_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    lens = [len(c["text"]) for c in output_chunks]
    leftover = sum(
        1 for c in output_chunks
        if re.search(r"\S+@\S+\.\S+|https?://|cdn\.discordapp", c["text"])
    )

    print(f"Output: {len(output_chunks)} chunks → {output_path}\n")
    print(f"  Artifacts cleaned:    {stats['artifacts']}")
    print(f"  Split (was too big):  {stats['split']}")
    print(f"  Dropped (spam):       {stats['spam']}")
    print(f"  Dropped (no replies): {stats['no_reply']}")
    print(f"  Dropped (too short):  {stats['too_short']}")
    if lens:
        print(f"\n  Avg chunk size:       {sum(lens) // len(lens)} chars")
        print(f"  Max chunk size:       {max(lens)} chars")
        print(f"  Artifact leaks left:  {leftover}")


if __name__ == "__main__":
    main()
