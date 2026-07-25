"""Unit tests for the SQLite vector store."""

from rag.store import VectorStore


def test_round_trip(store: VectorStore):
    store.add_chunk("doc.md", 0, "hello world", [0.1, 0.2, 0.3])
    store.commit()
    chunks = store.all_chunks()
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.source == "doc.md"
    assert chunk.chunk_index == 0
    assert chunk.content == "hello world"
    assert chunk.embedding == [0.1, 0.2, 0.3]


def test_count_and_clear(store: VectorStore):
    for i in range(5):
        store.add_chunk("a.md", i, f"chunk {i}", [float(i)])
    store.commit()
    assert store.count() == 5
    store.clear()
    assert store.count() == 0


def test_meta_round_trip_and_overwrite(store: VectorStore):
    assert store.get_meta("embedder") is None
    store.set_meta("embedder", "foundry:qwen3-embedding-0.6b")
    assert store.get_meta("embedder") == "foundry:qwen3-embedding-0.6b"
    store.set_meta("embedder", "fastembed:BAAI/bge-small-en-v1.5")
    assert store.get_meta("embedder") == "fastembed:BAAI/bge-small-en-v1.5"


def test_persists_across_connections(tmp_path):
    db = tmp_path / "persist.db"
    with VectorStore(db) as first:
        first.add_chunk("b.md", 0, "persisted", [1.0, 2.0])
        first.commit()
    with VectorStore(db) as second:
        assert second.count() == 1
        assert second.all_chunks()[0].content == "persisted"
