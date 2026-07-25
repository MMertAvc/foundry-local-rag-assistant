"""Unit tests for document chunking."""

from rag.chunker import chunk_text, split_paragraphs


def test_split_paragraphs_drops_empty():
    text = "First para.\n\n\n\nSecond para.\n\n   \n\nThird."
    assert split_paragraphs(text) == ["First para.", "Second para.", "Third."]


def test_short_text_is_single_chunk():
    assert chunk_text("Hello world.", chunk_size=100) == ["Hello world."]


def test_chunks_respect_size_limit():
    paragraphs = "\n\n".join(f"Paragraph number {i} with some words." for i in range(40))
    chunks = chunk_text(paragraphs, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    # +2 tolerance for the joining newlines when overlap tail is added
    assert all(len(c) <= 200 + 52 for c in chunks)


def test_oversized_paragraph_is_split():
    long_paragraph = "word " * 500  # ~2500 chars, no blank lines
    chunks = chunk_text(long_paragraph, chunk_size=300, overlap=0)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_overlap_carries_context():
    paragraphs = "\n\n".join(f"Sentence {i} " + "x" * 80 for i in range(10))
    chunks = chunk_text(paragraphs, chunk_size=200, overlap=60)
    # The tail of chunk N should appear at the start of chunk N+1.
    for previous, current in zip(chunks, chunks[1:]):
        assert previous[-30:] in current


def test_no_content_lost():
    text = "\n\n".join(f"Unique-token-{i}" for i in range(50))
    chunks = chunk_text(text, chunk_size=120, overlap=20)
    joined = " ".join(chunks)
    for i in range(50):
        assert f"Unique-token-{i}" in joined
