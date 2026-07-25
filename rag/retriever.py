"""Semantic retrieval: find the chunks most similar to a query.

We embed the query with the same model used for the documents, then
rank all stored chunks by cosine similarity. Pure-Python math keeps
the module dependency-free and easy to unit test.
"""

from __future__ import annotations

import math
from typing import List, NamedTuple

from . import config
from .store import StoredChunk, VectorStore


class ScoredChunk(NamedTuple):
    chunk: StoredChunk
    score: float


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two equal-length vectors, in [-1, 1]."""
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def rank_chunks(
    query_embedding: List[float],
    chunks: List[StoredChunk],
    top_k: int = config.TOP_K,
    min_similarity: float = config.MIN_SIMILARITY,
) -> List[ScoredChunk]:
    """Return the top_k chunks above the similarity threshold, best first."""
    scored = [
        ScoredChunk(chunk, cosine_similarity(query_embedding, chunk.embedding))
        for chunk in chunks
    ]
    scored.sort(key=lambda sc: sc.score, reverse=True)
    return [sc for sc in scored[:top_k] if sc.score >= min_similarity]


def get_top_chunks(
    query: str,
    engine,
    store: VectorStore,
    top_k: int = config.TOP_K,
) -> List[ScoredChunk]:
    """Embed `query` and return its most relevant stored chunks.

    `engine` is anything with an `embed_text(str) -> List[float]` method
    (the real FoundryEngine in production, a fake in tests).
    """
    query_embedding = engine.embed_text(query)
    return rank_chunks(query_embedding, store.all_chunks(), top_k=top_k)
