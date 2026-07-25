"""Central configuration for the Local RAG Assistant.

All tunable knobs live here so the rest of the code stays clean.
Override any value with an environment variable of the same name
prefixed with RAG_ (e.g. RAG_CHAT_MODEL=qwen2.5-0.5b).
"""

import os
from pathlib import Path

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = Path(os.getenv("RAG_DOCS_DIR", PROJECT_ROOT / "data" / "docs"))
DB_PATH = Path(os.getenv("RAG_DB_PATH", PROJECT_ROOT / "data" / "knowledge.db"))

# --- Foundry Local models ----------------------------------------------------
# Chat model: small (~3.8B) instruct model, good speed/quality balance on CPU.
CHAT_MODEL_ALIAS = os.getenv("RAG_CHAT_MODEL", "phi-3.5-mini")
# Embedding model used for both document chunks and user queries.
EMBED_MODEL_ALIAS = os.getenv("RAG_EMBED_MODEL", "qwen3-embedding-0.6b")

# Embedding backend selection:
#   auto    - try Foundry Local first, fall back to local ONNX (fastembed).
#             Foundry embedding models need Windows 11 24H2+ (build 26100);
#             on Windows 10 the fallback keeps everything 100% local.
#   foundry - Foundry Local only (fail loudly if unavailable)
#   local   - fastembed/ONNX only
EMBEDDER_MODE = os.getenv("RAG_EMBEDDER", "auto")
LOCAL_EMBED_MODEL = os.getenv("RAG_LOCAL_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

APP_NAME = "LocalRAGAssistant"

# --- Chunking ----------------------------------------------------------------
# Target chunk size in characters (~1-3 paragraphs) and overlap between chunks.
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))

# --- Retrieval -----------------------------------------------------------------
TOP_K = int(os.getenv("RAG_TOP_K", "3"))
# Chunks scoring below this cosine similarity are treated as irrelevant.
MIN_SIMILARITY = float(os.getenv("RAG_MIN_SIMILARITY", "0.42"))

# --- Generation ---------------------------------------------------------------
TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("RAG_MAX_TOKENS", "600"))

SYSTEM_PROMPT = """\
You are a helpful documentation assistant. Answer the user's question using
ONLY the context provided below. Follow these rules strictly:

1. Base your answer only on the context. Do not use outside knowledge.
2. If the context does not contain the answer, reply exactly:
   "I don't have that information in my knowledge base."
3. Cite the source of every fact you use, like: (source: <document name>).
4. Be concise, polite, and factual.

Context:
{context}
"""
