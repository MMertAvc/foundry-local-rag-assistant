"""Integration tests for ingestion + the full RAG pipeline (with FakeEngine)."""

from rag.ingest import ingest
from rag.pipeline import FALLBACK_ANSWER, answer_query, build_context
from rag.retriever import ScoredChunk
from rag.store import StoredChunk


def _write_docs(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "foundry.md").write_text(
        "Foundry model offline.\n\nFoundry runs the model offline on device.",
        encoding="utf-8",
    )
    (docs / "sqlite.md").write_text(
        "SQLite database storage.\n\nThe database is a single sqlite file.",
        encoding="utf-8",
    )
    return docs


def test_ingest_populates_store(engine, store, tmp_path):
    docs = _write_docs(tmp_path)
    total = ingest(engine, store, docs_dir=docs)
    assert total == store.count() > 0
    sources = {c.source for c in store.all_chunks()}
    assert sources == {"foundry.md", "sqlite.md"}


def test_reingest_replaces_old_data(engine, store, tmp_path):
    docs = _write_docs(tmp_path)
    first = ingest(engine, store, docs_dir=docs)
    second = ingest(engine, store, docs_dir=docs)
    assert first == second == store.count()


def test_answer_uses_retrieved_context(engine, store, tmp_path):
    docs = _write_docs(tmp_path)
    ingest(engine, store, docs_dir=docs)

    result = answer_query("how does the sqlite database work?", engine, store)

    assert result.answer.startswith("FAKE ANSWER")
    assert result.sources, "expected at least one retrieved chunk"
    assert result.sources[0].chunk.source == "sqlite.md"
    # The retrieved chunk text must be inside the system prompt sent to the LLM.
    system_prompt = engine.chat_calls[0]["system"]
    assert result.sources[0].chunk.content in system_prompt
    assert "[Document: sqlite.md]" in system_prompt


def test_unanswerable_question_gets_fallback(engine, store, tmp_path):
    docs = _write_docs(tmp_path)
    ingest(engine, store, docs_dir=docs)

    # No vocabulary overlap with the stored docs -> below the threshold.
    result = answer_query("what is the capital of France?", engine, store)

    assert result.answer == FALLBACK_ANSWER
    assert result.sources == []
    assert engine.chat_calls == []  # model was never called - no guessing


def test_ingest_records_embedder_name(engine, store, tmp_path):
    docs = _write_docs(tmp_path)
    ingest(engine, store, docs_dir=docs)
    assert store.get_meta("embedder") == engine.embedder_name


def test_embedder_mismatch_is_rejected(engine, store, tmp_path):
    import pytest

    docs = _write_docs(tmp_path)
    ingest(engine, store, docs_dir=docs)
    # Simulate opening the knowledge base with a different backend
    # (e.g. built on Win11 with Foundry, queried on Win10 with fastembed).
    engine.embedder_name = "fastembed:BAAI/bge-small-en-v1.5"

    with pytest.raises(RuntimeError, match="Rebuild it"):
        answer_query("how does the sqlite database work?", engine, store)


def test_empty_question_is_handled(engine, store):
    result = answer_query("   ", engine, store)
    assert "Please enter a question." == result.answer


def test_build_context_labels_sources():
    chunk = StoredChunk(1, "guide.md", 0, "Some content.", [1.0])
    context = build_context([ScoredChunk(chunk, 0.9)])
    assert "[Document: guide.md]" in context
    assert "Some content." in context
