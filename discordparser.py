"""
Discord plain-text export → RAG-ready JSONL chunks.

Usage:
    python discordparser.py                        # parses all .txt in DISCORD_CHATS_DIR
    python discordparser.py friend1.txt friend2.txt # parse specific files
"""

import json
import uuid
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

from config import (
    DISCORD_USERNAME, DISPLAY_NAME, DISCORD_CONTACT_MAP,
    SESSION_GAP_HOURS, MIN_MESSAGES_PER_CHUNK, MAX_MESSAGE_LEN,
    DISCORD_CHATS_DIR, OUTPUT_DIR,
)

# matches: [12/1/2020 9:40 PM] username
HEADER_RE = re.compile(r"^\[(\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}\s+[AP]M)\]\s+(.+)$")


def parse_timestamp(ts: str) -> datetime | None:
    try:
        return datetime.strptime(ts.strip(), "%m/%d/%Y %I:%M %p")
    except Exception:
        return None


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
        "college_logistics": ["class", "exam", "quiz", "submit", "assignment", "college", "marks", "test"],
        "tech_help": ["code", "bug", "error", "git", "install", "python", "linux", "server", "api"],
        "casual_banter": ["game", "play", "lmao", "lol", "bruh", "meme", "bro", "fr", "ngl"],
    }
    for topic, keywords in topics.items():
        if any(w in combined for w in keywords):
            return topic
    return "general"


def resolve_display_name(username: str) -> str:
    if username == DISCORD_USERNAME:
        return DISPLAY_NAME
    return DISCORD_CONTACT_MAP.get(username, username)


def clean_text(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None
    text = re.sub(r"@\w+", "", text).strip()
    if re.match(r"^https?://\S+$", text):
        return None
    text = re.sub(r"https?://\S+", "", text).strip()
    if text == "{Attachments}":
        return None
    if not text:
        return None
    if len(text) > MAX_MESSAGE_LEN:
        return None
    return text


def parse_txt(filepath: str) -> list[dict]:
    messages = []
    current_msg = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            stripped = line.strip()

            match = HEADER_RE.match(stripped)
            if match:
                if current_msg is not None:
                    messages.append(current_msg)

                ts_str, username = match.groups()
                current_msg = {
                    "_username": username.strip(),
                    "_display": resolve_display_name(username.strip()),
                    "_dt": parse_timestamp(ts_str),
                    "_raw_lines": [],
                }
            else:
                if current_msg is not None:
                    current_msg["_raw_lines"].append(stripped)

    if current_msg is not None:
        messages.append(current_msg)

    processed = []
    for msg in messages:
        raw = "\n".join(line for line in msg["_raw_lines"] if line)
        text = clean_text(raw)
        if text:
            processed.append({**msg, "_text": text})

    return processed


def split_into_sessions(messages: list[dict]) -> list[list[dict]]:
    if not messages:
        return []
    sessions, current = [], [messages[0]]
    for msg in messages[1:]:
        prev_dt = current[-1].get("_dt")
        curr_dt = msg.get("_dt")
        if prev_dt and curr_dt and (curr_dt - prev_dt) > timedelta(hours=SESSION_GAP_HOURS):
            sessions.append(current)
            current = [msg]
        else:
            current.append(msg)
    if current:
        sessions.append(current)
    return sessions


def combine_consecutive(messages: list[dict]) -> list[dict]:
    if not messages:
        return []
    combined = [messages[0].copy()]
    for msg in messages[1:]:
        if msg["_username"] == combined[-1]["_username"]:
            combined[-1]["_text"] += " " + msg["_text"]
        else:
            combined.append(msg.copy())
    return combined


def session_to_chunk(session: list[dict], contact_name: str) -> dict | None:
    clean = combine_consecutive(session)
    if len(clean) < MIN_MESSAGES_PER_CHUNK:
        return None

    lines = [f"[{m['_display']}]: {m['_text']}" for m in clean]
    your_messages = [m["_text"] for m in clean if m["_username"] == DISCORD_USERNAME]
    other_senders = list({m["_display"] for m in clean if m["_username"] != DISCORD_USERNAME})
    all_text = " ".join(m["_text"] for m in clean)

    start_dt = clean[0].get("_dt")
    date_str = start_dt.strftime("%Y-%m-%d") if start_dt else "unknown"
    time_str = start_dt.strftime("%H:%M") if start_dt else "unknown"

    return {
        "id": f"discord_{contact_name.replace(' ', '_')}_{date_str}_{uuid.uuid4().hex[:8]}",
        "text": "\n".join(lines),
        "metadata": {
            "source": "discord",
            "contact": contact_name,
            "other_senders": other_senders,
            "date": date_str,
            "time": time_str,
            "language": detect_language(all_text),
            "topic": detect_topic([m["_text"] for m in clean]),
            "message_count": len(clean),
            "your_messages": your_messages,
        },
    }


def process_discord_txt(filepath: str) -> list[dict]:
    path = Path(filepath)
    print(f"  Parsing: {path.name}")

    messages = parse_txt(filepath)
    print(f"    Messages: {len(messages)}")

    contact_name = path.stem
    for m in messages:
        if m["_username"] != DISCORD_USERNAME:
            contact_name = m["_display"]
            break

    sessions = split_into_sessions(messages)
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
        chat_dir = Path(DISCORD_CHATS_DIR)
        if not chat_dir.exists():
            print(f"Error: directory '{DISCORD_CHATS_DIR}' not found.")
            print(f"Create it and drop your Discord .txt exports inside,")
            print(f"or pass files directly: python discordparser.py chat1.txt")
            return
        files = sorted(str(p) for p in chat_dir.glob("*.txt"))
        if not files:
            print(f"No .txt files found in {DISCORD_CHATS_DIR}/")
            return

    print(f"Found {len(files)} chat file(s)\n")

    all_chunks = []
    for filepath in files:
        chunks = process_discord_txt(filepath)
        all_chunks.extend(chunks)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "rag_discord.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nDone! {len(all_chunks)} chunks → {output_path}")


if __name__ == "__main__":
    main()
