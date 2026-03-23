"""
WhatsApp chat export → RAG-ready JSONL chunks.

Usage:
    python wparser.py                          # parses all .txt in WHATSAPP_CHATS_DIR
    python wparser.py chat1.txt chat2.txt      # parse specific files
"""

import re
import json
import uuid
import sys
from datetime import datetime, timedelta
from pathlib import Path

from config import (
    YOUR_NAME, SESSION_GAP_HOURS, MIN_MESSAGES_PER_CHUNK,
    MAX_MESSAGE_LEN, WHATSAPP_CHATS_DIR, OUTPUT_DIR,
)

# ── PATTERNS ──────────────────────────────────────────────────────────────────

# WhatsApp uses a narrow no-break space (U+202F) between time and am/pm
LINE_RE = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{4}),\s+(\d{1,2}:\d{2}\s*[ap]m)\s+-\s+([^:]+):\s+(.+)$",
    re.IGNORECASE,
)

SKIP_PATTERNS = [
    r"<Media omitted>",
    r"This message was deleted",
    r".+ sent ₹[\d,.]+ to .+",
    r"Messages and calls are end-to-end encrypted.*",
    r"https?://\S+",
    r"location: https?://\S+",
    r"\u200e?.+ added .+",
    r"\u200e?.+ removed .+",
    r"\u200e?.+ left",
    r"\u200e?.+ joined using this group",
    r"<This message was edited>",
    r"Missed voice call",
    r"Missed video call",
]
SKIP_RE = re.compile("|".join(SKIP_PATTERNS))


# ── HELPERS ───────────────────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    if re.findall(r"[\u0900-\u097F]", text):
        return "hindi"
    hinglish = {
        "bhai", "yaar", "kya", "hai", "nahi", "nai", "bol", "kar",
        "bro", "tera", "mera", "haan", "abhi", "bahut", "accha",
        "theek", "karo", "raha", "toh", "aur", "kyun", "kuch",
    }
    if set(text.lower().split()) & hinglish:
        return "hinglish"
    return "english"


def detect_topic(messages: list[str]) -> str:
    combined = " ".join(messages).lower()
    topics = {
        "college_logistics": ["class", "exam", "quiz", "submit", "assignment", "marks", "test", "college"],
        "tech_help": ["code", "bug", "error", "git", "install", "python", "linux", "server", "api"],
        "money": ["₹", "sent", "pay", "money", "upi"],
        "casual_banter": ["game", "play", "lmao", "lol", "bruh", "meme", "bro", "fr", "ngl"],
    }
    for topic, keywords in topics.items():
        if any(w in combined for w in keywords):
            return topic
    return "general"


def is_junk(text: str) -> bool:
    text = text.strip()
    if not text:
        return True
    if SKIP_RE.search(text):
        return True
    if len(text) > MAX_MESSAGE_LEN:
        return True
    if re.match(r"^\d{1,2}/\d{1,2}/\d{4},\s+\d{1,2}:\d{2}", text):
        return True
    return False


# ── PARSING ───────────────────────────────────────────────────────────────────

def parse_whatsapp(filepath: str) -> list[dict]:
    messages = []
    current_msg = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            match = LINE_RE.match(line)
            if match:
                if current_msg:
                    messages.append(current_msg)

                date_str, time_str, sender, text = match.groups()
                try:
                    dt = datetime.strptime(
                        f"{date_str} {time_str.strip()}", "%d/%m/%Y %I:%M %p"
                    )
                except ValueError:
                    dt = None

                current_msg = {
                    "datetime": dt,
                    "sender": sender.strip(),
                    "text": text.strip(),
                }
            else:
                if current_msg and not re.match(r"^\d{1,2}/\d{1,2}/\d{4},", line):
                    current_msg["text"] += "\n" + line

    if current_msg:
        messages.append(current_msg)

    return messages


def combine_consecutive(messages: list[dict]) -> list[dict]:
    if not messages:
        return []
    combined = [messages[0].copy()]
    for msg in messages[1:]:
        if msg["sender"] == combined[-1]["sender"]:
            combined[-1]["text"] += " " + msg["text"]
        else:
            combined.append(msg.copy())
    return combined


def split_into_sessions(messages: list[dict]) -> list[list[dict]]:
    if not messages:
        return []
    sessions, current = [], [messages[0]]
    for msg in messages[1:]:
        prev = current[-1]
        if (
            msg["datetime"]
            and prev["datetime"]
            and (msg["datetime"] - prev["datetime"]) > timedelta(hours=SESSION_GAP_HOURS)
        ):
            sessions.append(current)
            current = [msg]
        else:
            current.append(msg)
    if current:
        sessions.append(current)
    return sessions


def session_to_chunk(session: list[dict], contact_name: str) -> dict | None:
    clean = [m for m in session if not is_junk(m["text"])]
    if len(clean) < MIN_MESSAGES_PER_CHUNK:
        return None

    clean = combine_consecutive(clean)

    lines = []
    for msg in clean:
        tag = "You" if msg["sender"] == YOUR_NAME else msg["sender"]
        lines.append(f"[{tag}]: {msg['text']}")

    your_messages = [m["text"] for m in clean if m["sender"] == YOUR_NAME]
    other_senders = list({m["sender"] for m in clean if m["sender"] != YOUR_NAME})
    all_text = " ".join(m["text"] for m in clean)

    start_dt = clean[0]["datetime"]
    date_str = start_dt.strftime("%Y-%m-%d") if start_dt else "unknown"
    time_str = start_dt.strftime("%H:%M") if start_dt else "unknown"

    return {
        "id": f"{contact_name.replace(' ', '_')}_{date_str}_{uuid.uuid4().hex[:8]}",
        "text": "\n".join(lines),
        "metadata": {
            "source": "whatsapp",
            "contact": contact_name,
            "other_senders": other_senders,
            "date": date_str,
            "time": time_str,
            "language": detect_language(all_text),
            "topic": detect_topic([m["text"] for m in clean]),
            "message_count": len(clean),
            "your_messages": your_messages,
        },
    }


def process_chat(filepath: str) -> list[dict]:
    path = Path(filepath)
    contact_name = re.sub(
        r"WhatsApp[_ ]Chat[_ ]with[_ ]", "", path.stem, flags=re.IGNORECASE
    ).replace("_", " ").strip()

    print(f"  Parsing: {contact_name}")

    raw = parse_whatsapp(filepath)
    raw = [m for m in raw if m["datetime"]]
    print(f"    Messages: {len(raw)}")

    sessions = split_into_sessions(raw)
    print(f"    Sessions: {len(sessions)}")

    chunks = []
    for session in sessions:
        chunk = session_to_chunk(session, contact_name)
        if chunk:
            chunks.append(chunk)

    print(f"    Chunks:   {len(chunks)}")
    return chunks


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        chat_dir = Path(WHATSAPP_CHATS_DIR)
        if not chat_dir.exists():
            print(f"Error: directory '{WHATSAPP_CHATS_DIR}' not found.")
            print(f"Create it and drop your WhatsApp .txt exports inside,")
            print(f"or pass files directly: python wparser.py chat1.txt chat2.txt")
            return
        files = sorted(str(p) for p in chat_dir.glob("*.txt"))
        if not files:
            print(f"No .txt files found in {WHATSAPP_CHATS_DIR}/")
            return

    print(f"Found {len(files)} chat file(s)\n")

    all_chunks = []
    for filepath in files:
        chunks = process_chat(filepath)
        all_chunks.extend(chunks)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "rag_whatsapp.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nDone! {len(all_chunks)} chunks → {output_path}")


if __name__ == "__main__":
    main()
