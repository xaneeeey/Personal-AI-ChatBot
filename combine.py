"""
Combine JSONL files from multiple sources into one dataset.

Usage:
    python combine.py                              # auto-finds rag_*.jsonl in output/
    python combine.py file1.jsonl file2.jsonl      # combine specific files
    python combine.py file1.jsonl file2.jsonl -o merged.jsonl
"""

import json
import sys
import random
import argparse
from collections import Counter
from pathlib import Path

from config import OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser(description="Combine JSONL files")
    parser.add_argument("files", nargs="*", help="Input JSONL files")
    parser.add_argument("-o", "--output", default=None, help="Output file name")
    parser.add_argument("--no-shuffle", action="store_true", help="Don't shuffle output")
    args = parser.parse_args()

    out_dir = Path(OUTPUT_DIR)
    output_path = Path(args.output) if args.output else out_dir / "rag_combined.jsonl"

    if args.files:
        files = args.files
    else:
        # auto-find parser output files
        files = sorted(
            str(p) for p in out_dir.glob("rag_*.jsonl")
            if p.name != output_path.name and "combined" not in p.name and "cleaned" not in p.name
        )

    if not files:
        print("No JSONL files found. Run the parsers first, or pass files explicitly.")
        return

    print(f"Combining {len(files)} file(s):\n")

    all_chunks = []
    for filepath in files:
        count = 0
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    all_chunks.append(json.loads(line))
                    count += 1
                except json.JSONDecodeError as e:
                    print(f"  Warning: skipping malformed line in {filepath}: {e}")
        print(f"  {filepath:<40} → {count} chunks")

    if not all_chunks:
        print("\nNo chunks found.")
        return

    if not args.no_shuffle:
        random.shuffle(all_chunks)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # stats
    valid = [c for c in all_chunks if isinstance(c.get("metadata"), dict)]
    sources = Counter(c["metadata"].get("source", "unknown") for c in valid)
    topics = Counter(c["metadata"].get("topic", "unknown") for c in valid)
    langs = Counter(c["metadata"].get("language", "unknown") for c in valid)

    print(f"\nTotal:     {len(all_chunks)} chunks → {output_path}")
    print(f"Sources:   {dict(sources)}")
    print(f"Topics:    {dict(topics)}")
    print(f"Languages: {dict(langs)}")


if __name__ == "__main__":
    main()
