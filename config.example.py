# ── PERSONAL AI CHATBOT — CONFIGURATION ──────────────────────────────────────
#
# 1. Copy this file:  cp config.example.py config.py
# 2. Edit config.py with your real details
# 3. config.py is gitignored — your info stays private
# ──────────────────────────────────────────────────────────────────────────────

# ── YOUR IDENTITY ─────────────────────────────────────────────────────────────
# Your name EXACTLY as it appears in WhatsApp exports
YOUR_NAME = "Your Name Here"

# Your Discord username (lowercase, as it appears in exports)
DISCORD_USERNAME = "your_discord_username"

# How you want your name displayed in the dataset
DISPLAY_NAME = "Your Name Here"

# Map discord usernames to friendly display names
# Add entries like: "their_username": "Their Name"
DISCORD_CONTACT_MAP = {
    # "friend_username": "Friend Name",
}

# ── SYSTEM PROMPT (for fine-tuning dataset) ───────────────────────────────────
# Describe your texting style so the model knows what to mimic
SYSTEM_PROMPT = (
    "You are {name}. Reply casually like them — "
    "short, chill, gen-z texting style."
).format(name=DISPLAY_NAME)

# ── PARSING SETTINGS ──────────────────────────────────────────────────────────
SESSION_GAP_HOURS = 2       # hours of silence before a new "session" starts
MIN_MESSAGES_PER_CHUNK = 2  # minimum messages to keep a session
CONTEXT_WINDOW = 5          # how many past messages as context (fine-tune)

# ── CLEANING SETTINGS ─────────────────────────────────────────────────────────
MAX_CHUNK_CHARS = 5000      # split chunks larger than this
MIN_CHUNK_CHARS = 20        # drop chunks smaller than this
MAX_MESSAGE_LEN = 1500      # drop single messages longer than this (pastes/walls)

# ── DIRECTORIES ───────────────────────────────────────────────────────────────
# Where your raw WhatsApp .txt exports live
WHATSAPP_CHATS_DIR = "chats/whatsapp"

# Where your raw Discord .txt exports live
DISCORD_CHATS_DIR = "chats/discord"

# Where processed data goes
OUTPUT_DIR = "output"
