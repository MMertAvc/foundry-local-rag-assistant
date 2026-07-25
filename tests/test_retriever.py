"""Unit tests for cosine similarity and chunk ranking."""

import math

import pytest

from rag.retriever import cosine_similarity, get_top_chunks, rank_chunks
from rag.store import StoredChunk


def test_cosine_identical_vectors():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite_vectors():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_known_value():
    # angle of 45 degrees -> cos = sqrt(2)/2
    assert cosine_similarity([1.0, 0.0], [1.0, 1.0]) == pytest.approx(
        math.sqrt(2) / 2
    )


def test_cosine_zero_vector_is_safe():
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_cosine_length_mismatch_raises():
    with pytest.raises(ValueError):
        cosine_similarity([1.0], [1.0, 2.0])


def _chunk(cid, embedding):
    return StoredChunk(cid, f"doc{cid}.md", 0, f"content {cid}", embedding)


def test_rank_orders_by_similarity():
    chunks = [
        _chunk(1, [1.0, 0.0]),   # identical to query -> best
        _chunk(2, [1.0, 1.0]),   # 45 degrees
        _chunk(3, [0.0, 1.0]),   # orthogonal -> filtered by threshold
    ]
    ranked = rank_chunks([1.0, 0.0], chunks, top_k=3, min_similarity=0.3)
    assert [sc.chunk.id for sc in ranked] == [1, 2]
    assert ranked[0].score > ranked[1].score


def test_rank_respects_top_k():
    chunks = [_chunk(i, [1.0, 0.01 * i]) for i in range(10)]
    ranked = rank_chunks([1.0, 0.0], chunks, top_k=3, min_similarity=0.0)
    assert len(ranked) == 3


def test_get_top_chunks_end_to_end(engine, store):
    store.add_chunk("sqlite.md", 0, "sqlite database sqlite database",
                    engine.embed_text("sqlite database sqlite database"))
    store.add_chunk("rag.md", 0, "rag embedding rag",
                    engine.embed_text("rag embedding rag"))
    store.commit()

    results = get_top_chunks("tell me about the sqlite database", engine, store)
    assert results
    assert results[0].chunk.source == "sqlite.md"
