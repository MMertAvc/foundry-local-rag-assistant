# 📚 Local RAG Assistant — Offline Document Q&A with Microsoft Foundry Local

A document question-answering chatbot that runs **100% offline** on your own
computer. It combines **Microsoft Foundry Local** (on-device LLM inference),
**Retrieval-Augmented Generation (RAG)**, and a **SQLite vector store** — no
cloud account, no API keys, no internet needed at question time.

> Built for the Microsoft Summer School program, following the
> [Building Your First Local RAG Application with Foundry Local](https://techcommunity.microsoft.com/blog/azuredevcommunityblog/building-your-first-local-rag-application-with-foundry-local/4501968)
> Tech Community example and the official
> [Foundry Local documentation](https://learn.microsoft.com/en-us/azure/foundry-local/).

## How it works

```
                 ┌──────────────────────── one machine, fully offline ───────────────────────┐
                 │                                                                            │
  User question ─┼─► CLI (main.py) ──► Pipeline (rag/pipeline.py)                             │
                 │   or Streamlit         │                                                   │
                 │   (app.py)             │ 1. RETRIEVE                                       │
                 │                        ▼                                                   │
                 │              Retriever (rag/retriever.py)                                  │
                 │              embed query ──► cosine similarity ranking                     │
                 │                        │                ▲                                  │
                 │                        │                │ chunks + embeddings              │
                 │                        │       SQLite (data/knowledge.db)                  │
                 │                        │                ▲                                  │
                 │                        │        Ingestion (rag/ingest.py)                  │
                 │                        │        chunk → embed → store                      │
                 │                        │                ▲                                  │
                 │                        │        Documents (data/docs/*.md)                 │
                 │                        │ 2. AUGMENT (top-k chunks into system prompt)      │
                 │                        ▼                                                   │
                 │              Foundry Local (rag/foundry.py)                                │
                 │              phi-3.5-mini (chat) + qwen3-embedding-0.6b (embeddings)       │
                 │                        │ 3. GENERATE                                       │
                 │                        ▼                                                   │
  Answer with  ◄─┼── grounded answer + source citations + similarity scores                   │
  citations      │                                                                            │
                 └────────────────────────────────────────────────────────────────────────────┘
```

1. **Ingest** — documents in `data/docs/` are split into overlapping ~900-character
   chunks, embedded locally (see *Embedding backends* below), and stored in SQLite.
2. **Retrieve** — the question is embedded with the same model; all chunks are
   ranked by cosine similarity and the top 3 above a 0.30 threshold are kept.
3. **Augment** — retrieved chunks (labeled with their source document) are
   injected into the system prompt with strict grounding rules.
4. **Generate** — `phi-3.5-mini` produces the answer locally, citing sources.
   If nothing relevant was retrieved, the assistant says it doesn't know —
   the model is never asked to guess.

## Quick start

Requires **Python 3.11+** (Windows, macOS, or Linux).

```bash
# 1. Clone and install
git clone https://github.com/MMertAvc/foundry-local-rag-assistant.git
cd foundry-local-rag-assistant
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

# 2. Build the knowledge base (first run downloads the models — needs internet ONCE)
python main.py ingest

# 3. Ask away — fully offline from here on
python main.py ask "What is cosine similarity used for in RAG?"
python main.py chat              # interactive loop
streamlit run app.py             # web UI (Streamlit)
python server.py                 # web UI (custom design) at http://127.0.0.1:8000
```

### Example session

```
$ python main.py ask "Why does RAG reduce hallucinations?"

When the prompt explicitly contains the relevant source text and the model is
instructed to answer only from it, the model is far less likely to invent
facts (source: 02-rag-pattern.md). RAG also enables source citations because
the application knows which document each passage came from
(source: 02-rag-pattern.md).

Retrieved context:
  - 02-rag-pattern.md (chunk 1, similarity 0.78)
  - 05-prompt-engineering.md (chunk 1, similarity 0.52)

(answered in 2.4s, fully offline)
```

## Project structure

```
foundry-local-rag-assistant/
├── main.py                 # CLI: ingest / ask / chat / stats
├── app.py                  # Streamlit web UI
├── server.py               # custom HTML/JS web UI (Starlette + uvicorn), port 8000
├── static/                 # custom UI assets (index.html, support.js, vendor/)
├── rag/
│   ├── config.py           # all tunable settings (models, chunking, top-k, prompt)
│   ├── foundry.py          # Foundry Local SDK wrapper (chat + embeddings)
│   ├── chunker.py          # paragraph-aware chunking with overlap
│   ├── ingest.py           # documents → chunks → embeddings → SQLite
│   ├── store.py            # SQLite vector store
│   ├── retriever.py        # cosine similarity ranking (pure Python)
│   └── pipeline.py         # retrieve → augment → generate
├── data/docs/              # the knowledge base source documents (.md / .txt)
├── tests/                  # 31 unit & integration tests (SDK mocked)
└── docs/TEST_REPORT.md     # test methodology and results
```

## Configuration

Every knob lives in `rag/config.py` and can be overridden with environment
variables — for example a faster, smaller chat model:

| Variable | Default | Purpose |
|---|---|---|
| `RAG_CHAT_MODEL` | `phi-3.5-mini` | chat model alias in the Foundry catalog |
| `RAG_EMBED_MODEL` | `qwen3-embedding-0.6b` | Foundry embedding model alias |
| `RAG_EMBEDDER` | `auto` | embedding backend: `auto` / `foundry` / `local` |
| `RAG_LOCAL_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | ONNX model used by the local fallback |
| `RAG_CHUNK_SIZE` | `900` | target chunk size (characters) |
| `RAG_CHUNK_OVERLAP` | `150` | overlap between consecutive chunks |
| `RAG_TOP_K` | `3` | retrieved chunks per question |
| `RAG_MIN_SIMILARITY` | `0.42` | below this, a chunk is treated as irrelevant |
| `RAG_TEMPERATURE` | `0.2` | low = consistent, factual answers |

```bash
# example: trade answer quality for speed on an older laptop
set RAG_CHAT_MODEL=qwen2.5-0.5b     # Windows (macOS/Linux: export ...)
python main.py chat
```

## Embedding backends (Windows 10 note)

Foundry Local's embedding models (including `qwen3-embedding-0.6b`) require
**Windows 11 24H2+ (build 26100)** — on older builds the service excludes them
from the catalog (`Skipping EP autoregistration ... Minimum supported build is
26100` in the log). Chat models are unaffected.

This project therefore treats embedding as a pluggable backend
(`rag/embedders.py`), selected by `RAG_EMBEDDER`:

- **`auto` (default)** — try the Foundry embedding model; if it's unavailable
  (e.g. Windows 10), fall back automatically to a local **ONNX Runtime**
  embedder via [`fastembed`](https://pypi.org/project/fastembed/)
  (`bge-small-en-v1.5`, ~130 MB, CPU-only). Both paths run 100% on-device —
  the offline guarantee is preserved either way, and it's the same ONNX
  technology Foundry Local itself is built on.
- **`foundry`** — force Foundry embeddings (fails loudly if unsupported).
- **`local`** — force the fastembed backend.

The knowledge base records which embedder built it; if you later query with a
different backend, the app refuses with a clear message instead of silently
returning garbage similarities — just re-run `python main.py ingest`.

## Using your own documents

Drop `.md` or `.txt` files into `data/docs/` and re-run
`python main.py ingest`. The knowledge base is rebuilt from scratch each time,
so edits and deletions are picked up automatically.

## Testing

The whole pipeline is covered by 31 tests that run without Foundry Local
installed — a deterministic `FakeEngine` stands in for the SDK, so tests are
fast and CI-friendly:

```bash
python -m pytest tests/ -v
```

Covered behaviors include: chunking respects size limits and loses no content,
SQLite round-trips embeddings exactly, cosine similarity matches hand-computed
values, ranking respects top-k and the similarity threshold, retrieved context
actually reaches the model's system prompt, and **unanswerable questions get
the fallback answer without ever calling the model**. See
[docs/TEST_REPORT.md](docs/TEST_REPORT.md) for the full report.

## Design decisions & limitations

**Why brute-force retrieval instead of a vector database?** At this scale
(dozens of chunks) scanning every vector takes microseconds, and the simple
code is easier to understand and debug. Beyond ~100k chunks you would want an
ANN index (e.g. HNSW) or a dedicated vector DB.

**Why a similarity threshold?** Without one, the top-k chunks are returned
even when none of them are actually relevant, and the model gets misled into
answering from noise. The 0.42 threshold plus the prompt's "say you don't
know" rule together keep the assistant honest.

**Known limitations:** answers are only as good as the documents; the small
chat model can occasionally phrase citations imperfectly; PDF/DOCX ingestion
is not implemented (convert to `.md`/`.txt` first); retrieval is single-turn —
follow-up questions don't inherit conversation context.

## References

- [What is Foundry Local?](https://learn.microsoft.com/en-us/azure/foundry-local/what-is-foundry-local) — Microsoft Learn
- [Get started with Foundry Local](https://learn.microsoft.com/en-us/azure/foundry-local/get-started) — Microsoft Learn
- [Tutorial: Build a RAG application](https://learn.microsoft.com/en-us/azure/foundry-local/) — Microsoft Learn
- [Building Your First Local RAG Application with Foundry Local](https://techcommunity.microsoft.com/blog/azuredevcommunityblog/building-your-first-local-rag-application-with-foundry-local/4501968) — Microsoft Tech Community
- [foundry-local-sdk on PyPI](https://pypi.org/project/foundry-local-sdk/)
- [Prompt engineering techniques](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/prompt-engineering) — Microsoft Learn

## License

MIT — see [LICENSE](LICENSE).

---

*Mustafa Mert Avcı · Microsoft Summer School 2026 · Local RAG Assistant final project*
