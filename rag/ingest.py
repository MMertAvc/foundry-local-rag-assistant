"""Ingestion pipeline: documents -> chunks -> embeddings -> SQLite.

Run via `python main.py ingest`. Re-running rebuilds the knowledge
base from scratch, so adding/editing documents is just: drop the file
into data/docs/ and re-ingest.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

from . import config
from .chunker import chunk_text
from .store import VectorStore

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".md", ".txt"}


def load_documents(docs_dir: Path = config.DOCS_DIR) -> List[Tuple[str, str]]:
    """Read all supported documents. Returns (file_name, text) pairs."""
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")
    documents = []
    for path in sorted(docs_dir.iterdir()):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.append((path.name, path.read_text(encoding="utf-8")))
    if not documents:
        raise FileNotFoundError(f"No .md/.txt documents found in {docs_dir}")
    return documents


def ingest(engine, store: VectorStore, docs_dir: Path = config.DOCS_DIR) -> int:
    """Chunk, embed and store every document. Returns total chunk count.

    `engine` needs an `embed_texts(List[str]) -> List[List[float]]` method.
    """
    documents = load_documents(docs_dir)
    store.clear()

    total = 0
    for file_name, text in documents:
        chunks = chunk_text(text)
        logger.info("Embedding %d chunks from %s ...", len(chunks), file_name)
        embeddings = engine.embed_texts(chunks)  # batch call = fewer round trips
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            store.add_chunk(file_name, index, chunk, embedding)
        total += len(chunks)

    store.commit()
    # Record which embedder built this knowledge base; queries must use the
    # same one (vectors from different models are not comparable).
    embedder_name = getattr(engine, "embedder_name", None)
    if embedder_name:
        store.set_meta("embedder", embedder_name)
    logger.info("Ingestion complete: %d chunks from %d documents.", total, len(documents))
    return total
