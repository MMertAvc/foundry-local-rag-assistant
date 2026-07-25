"""Pluggable embedding backends.

The original design uses Foundry Local's embedding models
(qwen3-embedding-0.6b). However, Foundry Local's embedding catalog
requires Windows 11 24H2 (build 26100) or newer; on Windows 10 the
service excludes embedding models entirely ("Skipping EP
autoregistration ... Minimum supported build is 26100").

To keep the assistant 100% offline on every OS, embedding is a
pluggable interface with two implementations:

  * FoundryEmbedder   - Foundry Local embedding model (Win11 24H2+)
  * FastEmbedEmbedder - ONNX Runtime-based local model via `fastembed`
                        (works everywhere Foundry Local's chat side works;
                        same ONNX technology Foundry Local itself uses)

Both run fully on-device. `RAG_EMBEDDER=auto` (default) tries Foundry
first and falls back automatically.

IMPORTANT: documents and queries must use the SAME embedder. The
embedder's `name` is recorded in the knowledge base at ingestion time
and checked at query time (see store.py / pipeline.py).
"""

from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)


class FoundryEmbedder:
    """Embeddings served by a Foundry Local embedding model."""

    def __init__(self, model, alias: str) -> None:
        self._client = model.get_embedding_client()
        self.name = f"foundry:{alias}"

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        response = self._client.generate_embeddings(texts)
        return [item.embedding for item in response.data]


class FastEmbedEmbedder:
    """Local ONNX embeddings via the `fastembed` package.

    Downloads the model once (~130 MB for bge-small-en-v1.5), caches it,
    then runs fully offline on CPU through ONNX Runtime.
    """

    def __init__(self, model_name: str) -> None:
        # Lazy import: only needed when this backend is actually used.
        from fastembed import TextEmbedding

        logger.info("Loading local ONNX embedding model '%s' ...", model_name)
        self._model = TextEmbedding(model_name=model_name)
        self.name = f"fastembed:{model_name}"

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]
