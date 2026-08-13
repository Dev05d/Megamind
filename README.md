# Megamind

**Megamind** is a 100% local, personal Retrieval-Augmented Generation (RAG) system. Point it at web pages, PDFs, YouTube videos, or Reddit threads, and it builds you a private, queryable knowledge base you can talk to in natural language — no cloud APIs, no external services, no data leaving your machine.

> **Note:** This project is under active development (currently v1.0). Expect rough edges, and expect this README to keep evolving alongside the code.

---

## Why Megamind

Most RAG tutorials wire together a handful of hosted APIs. Megamind takes the opposite approach: every piece of the pipeline — scraping, embedding, storage, retrieval, and generation — runs on your own hardware through [Ollama](https://ollama.com) and an embedded [Qdrant](https://qdrant.tech) instance. Your sources and your queries never leave your computer.

## Features

- **Fully local** — embeddings, OCR, routing, and chat all run through Ollama; the vector database runs embedded on disk, no server to stand up.
- **Multi-source ingestion** — feed it websites, YouTube videos, Reddit threads, or PDFs (including scanned/handwritten ones via OCR).
- **Hybrid search** — every chunk is stored as both a dense semantic vector and a sparse keyword vector, merged with Reciprocal Rank Fusion (RRF) so you get accurate matches whether your query shares exact words with a source or not.
- **Elastic semantic chunking** — a custom chunker that groups text along paragraph/sentence boundaries instead of blunt character cutoffs, and links neighboring chunks with sentence-level overlap so ideas don't get sliced mid-thought.
- **Smart query routing** — an LLM classifies each query (chit-chat, answerable from existing context, needs a fresh database hit, or needs a follow-up hit for new info on the same topic) so you're not hitting the vector database when you don't need to.
- **Token-aware memory management** — tracks real token usage via Ollama's API and trims the oldest chunks/messages once you approach your context limit, instead of blindly truncating.
- **No hallucinated answers** — if the database returns zero relevant chunks, Megamind returns an "information not found" response and skips the LLM call entirely.
- **Built-in maintenance tools** — swap embedding models (with automatic re-embedding or full migration), delete a source and all its chunks, switch chat/router models with auto-detected context limits, and inspect what's actually in your database.
- **Two ways to drive it** — a full arrow-key interactive terminal menu, or scriptable CLI flags for automation.

## How it works

**Ingestion pipeline:** a source (web page, PDF, video, or thread) is scraped and cleaned → split into chunks by the elastic chunker → embedded into dense + sparse vectors → written to the local Qdrant collection, with the source logged and timestamped for auditing.

**Query pipeline:** your question first goes to the query router, which decides whether the current conversation context already has enough information, whether the database needs to be searched for new chunks, or whether it's just chit-chat. If a database hit is needed, the query is embedded and searched with both dense and sparse vectors, the two result sets are merged with RRF, and the top chunks are handed to the memory manager, which fits them (and prior chat history) into the model's context window before streaming a response back from Ollama.

## Project structure

```
Megamind/
├── core/
│   ├── browser.py          # Headless-browser fetching (Playwright) w/ a spoofed user agent
│   ├── chunker.py           # The elastic semantic chunker
│   ├── config.py             # Loads settings from .env for the rest of the app
│   ├── memory_manager.py    # Tracks token usage and trims context when needed
│   ├── retriever.py          # The chat loop: routing, retrieval, streaming replies
│   ├── router.py             # Classifies queries into chitchat / context_sufficient / needs_retrieval / needs_novel_retrieval
│   ├── tracker.py            # Logs every ingested source with timestamps
│   └── vector_store.py       # Builds/queries the Qdrant collection (dense + sparse)
├── scrapers/
│   ├── scrape_pdf.py         # Renders PDF pages to images and OCRs them via Ollama
│   ├── scrape_reddit.py      # Pulls threads via Reddit's JSON endpoints through Playwright
│   ├── scrape_web.py         # Fetches + cleans a web page into markdown
│   └── scrape_youtube.py     # Pulls video metadata + transcript, no API key needed
├── tools/
│   ├── change_embedd_model.py # Swap the embedding model, re-embedding or migrating as needed
│   ├── delete_sources.py     # Remove a source and every chunk tied to it
│   ├── model_migrator.py     # Swap the chat/router models and auto-detect context limits
│   ├── show_db_info.py       # Summary of what's stored in the database
│   └── show_raw_data.py      # Inspect the raw chunks for a given source
├── sources/
│   ├── raw_pdfs/             # Local copies of ingested PDFs
│   ├── web.txt                # Log of ingested URLs + timestamps
│   ├── youtube.txt            # Log of ingested videos + timestamps
│   └── reddit.txt             # Log of ingested threads + timestamps
├── megamind_db/                # Embedded Qdrant collection (created on first run)
├── main.py                     # Entry point + CLI flags
├── menu.py                     # Interactive terminal menu (questionary)
├── requirements.txt
└── .env                         # Local configuration
```

## Getting started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally
- [Playwright](https://playwright.dev/python/) browser binaries (installed below)

You do **not** need to stand up a separate database server — Qdrant runs in embedded/local mode and stores its data on disk under `megamind_db/`.

### 1. Clone and install

```bash
git clone https://github.com/Dev05d/Megamind.git
cd Megamind

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
playwright install chromium
```

### 2. Pull the local models

Megamind ships configured for the following Ollama models by default — pull them, or swap in your own and update `.env` (the `model_migrator.py` tool can also do this for you interactively later):

```bash
ollama pull qwen3-embedding:4b   # embeddings
ollama pull lfm2.5:8b            # chat model
ollama pull llama3.1:8b          # query router model
ollama pull glm-ocr:latest       # PDF OCR
```

### 3. Configure

Edit `.env` to taste:

| Variable | Default | Description |
|---|---|---|
| `DB_PATH` | `./megamind_db` | Where the local vector database is stored |
| `COLLECTION_NAME` | `megamind` | Qdrant collection name |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server address (no trailing slash) |
| `EMBED_MODEL` | `qwen3-embedding:4b` | Embedding model |
| `CHAT_MODEL` | `lfm2.5:8b` | Main chat model |
| `CONTEXT_LIMIT` | `16000` | Chat model context budget (tokens) |
| `ROUTER_MODEL` | `llama3.1:8b` | Query router model |
| `ROUTER_CONTEXT` | `8192` | Router model context budget |
| `OCR_MODEL` | `glm-ocr:latest` | Vision model used to OCR PDF pages |
| `VECTOR_SIZE` | `2560` | Embedding dimension (must match `EMBED_MODEL`) |
| `TOP_K_CHUNKS` | `14` | Chunks returned per retrieval |
| `DENSE_THRESHOLD` | `0.5` | Minimum cosine similarity for a dense match |
| `CHUNK_SENTENCE_OVERLAP` | `2` | Sentence overlap between adjacent chunks |

## Usage

Launch with no arguments to get the interactive menu (ingestion, chat, and all maintenance tools in one place):

```bash
python main.py
```

Or drive it directly from the command line:

```bash
python main.py --ingest-web "https://example.com/some-article"
python main.py --ingest-youtube "https://www.youtube.com/watch?v=..."
python main.py --ingest-reddit "https://www.reddit.com/r/example/comments/..."
python main.py --ingest-pdf "./sources/raw_pdfs/paper.pdf"
python main.py --query "What did that article say about X?"
```

## Roadmap

Megamind's v1.0 foundation — hybrid retrieval, elastic chunking, query routing, and memory management — is in place. Next up:

- A non-LLM approach to query routing (the current router works, but leans on an extra model call that adds latency and compute)
- Continued refinement of ingestion quality across source types

## Development blog

The build process is documented in a blog series covering the reasoning behind each design decision:

- [Making of Megamind — Part 1](https://devanshmamoria.blogspot.com/2026/07/making-of-megamind-part-1.html) — architecture, chunking, hybrid search
- [Making of Megamind — Part 2](https://devanshmamoria.blogspot.com/2026/07/making-of-megamind-part-2.html) — token tracking, memory management, query routing
- [Making of Megamind — Part 3](https://devanshmamoria.blogspot.com/2026/08/making-of-megamind-part-3.html) — v1.0 wrap-up and full project breakdown

## License

This project is licensed under the [MIT License](LICENSE).

## Author

Built by [Devansh Mamoria](https://devanshmamoria.blogspot.com/).