# Personal AI Chatbot

Build a personalized AI chatbot that mimics your texting style, trained on your own WhatsApp and Discord messages.

This repo handles the full pipeline: **exporting → parsing → cleaning → RAG inference** with optional fine-tuning dataset generation.

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/youruser/Personal-AI-ChatBot.git
cd Personal-AI-ChatBot

# 2. Copy and edit the config
cp config.example.py config.py
# Then open config.py and fill in your name (see Configuration below)

# 3. Drop your chat exports into the right folders
mkdir -p chats/whatsapp chats/discord

# 4. Run the full data pipeline
python run_all.py

# 5. Run the chatbot
export GEMINI_API_KEY='your-key-here'
python RAG.py
```

---

## Configuration

Open **`config.py`** and fill in your details. This is the only file you need to edit:

```python
# Your name EXACTLY as it appears in WhatsApp exports
YOUR_NAME = "Jane Doe"

# Your Discord username (lowercase)
DISCORD_USERNAME = "janedoe"

# Display name used in the dataset
DISPLAY_NAME = "Jane Doe"

# Map Discord usernames to readable names
DISCORD_CONTACT_MAP = {
    "coolfriend99": "Alex",
    "gamerboy":     "Sam",
}

# Customize the system prompt to match your vibe
SYSTEM_PROMPT = "You are Jane. Reply casually like her — short, chill, gen-z texting style."
```

All other settings (session gaps, chunk sizes, etc.) have sane defaults but are tunable in the same file.

---

## Exporting Your Chats

### WhatsApp

1. Open a chat in WhatsApp
2. Tap **⋮ → More → Export chat → Without media**
3. Drop the `.txt` file into `chats/whatsapp/`
4. Repeat for every chat you want

### Discord

Discord doesn't have built-in export. Use [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter) and export as **Plain Text (.txt)**. Drop files into `chats/discord/`.

---

## Pipeline Overview

```
                         DATA PIPELINE
                         ─────────────
chats/whatsapp/*.txt ──→ wparser.py ──→ output/rag_whatsapp.jsonl ──┐
                                                                     ├→ combine.py → clean.py → output/rag_cleaned.jsonl
chats/discord/*.txt  ──→ discordparser.py ──→ output/rag_discord.jsonl ┘
                                                            │
                                                  (optional) cleanv2.py → output/rag_rechunked.jsonl

                         INFERENCE
                         ─────────
                  output/rag_cleaned.jsonl ──→ RAG.py ──→ live chatbot


                         FINE-TUNING (optional)
                         ──────────────────────
              chats/whatsapp/*.txt ──→ data.py ──→ output/finetune_dataset.jsonl
```

### What each script does

| Script | Purpose |
|---|---|
| `config.py` | Central configuration — edit this first |
| `run_all.py` | Runs the full data pipeline in one command |
| `wparser.py` | Parses WhatsApp `.txt` exports into RAG chunks |
| `discordparser.py` | Parses Discord `.txt` exports into RAG chunks |
| `combine.py` | Merges chunks from all sources into one file |
| `clean.py` | Removes artifacts, spam, URLs, oversized chunks |
| `cleanv2.py` | *(optional)* Re-chunks with sliding window for finer retrieval |
| `data.py` | Builds ChatML fine-tuning dataset from WhatsApp exports |
| `RAG.py` | The actual chatbot — retrieval + Gemini generation |

---

## Running the Data Pipeline

### One command (recommended)

```bash
python run_all.py              # RAG dataset only
python run_all.py --finetune   # RAG + fine-tuning dataset
```

### Individual scripts

```bash
# Parse
python wparser.py
python discordparser.py

# Combine + Clean
python combine.py
python clean.py

# Optional: finer chunking for better retrieval
python cleanv2.py

# Fine-tuning dataset (WhatsApp only)
python data.py
```

Every script auto-discovers files from configured directories, but also accepts explicit paths:

```bash
python wparser.py "WhatsApp Chat with Alex.txt"
python clean.py output/rag_combined.jsonl -o output/rag_cleaned.jsonl
python cleanv2.py output/rag_cleaned.jsonl -o output/rag_rechunked.jsonl
```

---

## Running the Chatbot (RAG.py)

### Prerequisites

```bash
pip install google-genai duckduckgo-search chromadb sentence-transformers tqdm
```

### Setup

1. Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)
2. Set it as an environment variable:

```bash
export GEMINI_API_KEY='your-key-here'
```

### Run

```bash
python RAG.py
```

On first run, it embeds all your chunks into ChromaDB (takes a few minutes). After that, it loads from disk instantly.

You'll be prompted to pick a contact, then you can chat:

```
Available contacts:
  1. Alex (142 chunks)
  2. Sam (89 chunks)
  0. No filter (use all contacts)

Select contact number [0]: 1
Chatting as: Alex

Alex: yo wyd
You: nothing much just chilling

Alex: class cancel hai aaj
You: sahi hai free day then
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | Your Gemini API key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Which Gemini model to use |
| `DATASET_PATH` | `output/rag_cleaned.jsonl` | Path to your cleaned dataset |

---

## Output Formats

### RAG Chunks (`rag_cleaned.jsonl`)

```json
{
  "id": "Alex_2024-11-15_a3f8b2c1",
  "text": "[You]: hey you free tonight?\n[Alex]: yeah what's up\n[You]: wanna grab food",
  "metadata": {
    "source": "whatsapp",
    "contact": "Alex",
    "date": "2024-11-15",
    "language": "english",
    "topic": "casual_banter",
    "message_count": 5,
    "your_messages": ["hey you free tonight?", "wanna grab food"]
  }
}
```

### Fine-tuning Dataset (`finetune_dataset.jsonl`)

```json
{
  "messages": [
    {"role": "system", "content": "You are Jane. Reply casually..."},
    {"role": "user", "content": "hey you free tonight?"},
    {"role": "assistant", "content": "yeah lemme check"},
    {"role": "user", "content": "cool lmk"},
    {"role": "assistant", "content": "ok im down"}
  ]
}
```

---

## Project Structure

```
Personal-AI-ChatBot/
├── config.example.py      # ← copy to config.py, then edit
├── config.py              # your private config (gitignored)
├── .gitignore             # keeps your data out of git
├── run_all.py             # one-command data pipeline
├── wparser.py             # WhatsApp parser
├── discordparser.py       # Discord parser
├── combine.py             # merge sources
├── clean.py               # clean artifacts
├── cleanv2.py             # optional: sliding-window rechunk
├── data.py                # fine-tuning dataset builder
├── RAG.py                 # the chatbot
├── chats/
│   ├── whatsapp/          # drop WhatsApp .txt exports here
│   └── discord/           # drop Discord .txt exports here
├── output/                # processed datasets
│   ├── rag_whatsapp.jsonl
│   ├── rag_discord.jsonl
│   ├── rag_combined.jsonl
│   ├── rag_cleaned.jsonl  # ← main RAG dataset
│   ├── rag_rechunked.jsonl
│   └── finetune_dataset.jsonl
└── chroma_db/             # vector store (auto-created by RAG.py)
```

---

## Privacy

No personal information is hardcoded anywhere in the codebase. All names, contacts, and identity details live exclusively in `config.py`. Add this to your `.gitignore`:

```
config.py
chats/
output/
chroma_db/
```

The `clean.py` and `RAG.py` scripts also automatically redact phone numbers, emails, and card numbers from your dataset before indexing.

---

## Requirements

**Data pipeline:** Python 3.10+ (standard library only, no external deps)

**RAG chatbot:** Additional packages:
```bash
pip install google-genai duckduckgo-search chromadb sentence-transformers tqdm
```

A GPU is recommended for the embedding model (bge-m3). CPU works but is slower for initial indexing.

---

## Tips for Better Results

**Data quality > quantity.** A few hundred clean 1-on-1 conversations beat thousands of noisy group chat messages.

**Export your closest friends first.** Chats where you're most "yourself" produce the best training signal.

**Exclude group chats** unless small and you're very active in them.

**Check your config.** The most common issue is `YOUR_NAME` not matching exactly what WhatsApp uses. Open a `.txt` export and copy your name character-for-character.

**Tune the system prompt.** The `SYSTEM_PROMPT` in `config.py` shapes the chatbot's personality. Start simple, then iterate based on outputs.

**Use `cleanv2.py` if retrieval feels off.** The sliding-window rechunking creates smaller, overlapping chunks that can improve retrieval precision for shorter queries.

---

## License

MIT
