"""End-to-end RAG pipeline: Retrieve -> Augment -> Generate.

`answer_query()` is the single entry point used by both the CLI
(main.py) and the Streamlit UI (app.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from . import config
from .retriever import ScoredChunk, get_top_chunks
from .store import VectorStore

FALLBACK_ANSWER = "I don't have that information in my knowledge base."


@dataclass
class RagAnswer:
    """The generated answer plus the evidence that produced it."""

    question: str
    answer: str
    sources: List[ScoredChunk] = field(default_factory=list)


def _check_embedder_match(engine, store: VectorStore) -> None:
    """Refuse to query if the knowledge base was built with a different
    embedding backend — vectors from different models live in different
    spaces and similarity scores would be meaningless."""
    stored = store.get_meta("embedder")
    current = getattr(engine, "embedder_name", None)
    if stored and current and stored != current:
        raise RuntimeError(
            f"Knowledge base was built with embedder '{stored}' but the "
            f"current engine uses '{current}'. Rebuild it with: "
            "python main.py ingest"
        )


def build_context(chunks: List[ScoredChunk]) -> str:
    """Format retrieved chunks into a context block for the prompt."""
    blocks = []
    for sc in chunks:
        blocks.append(f"[Document: {sc.chunk.source}]\n{sc.chunk.content}")
    return "\n\n---\n\n".join(blocks)


def answer_query(question: str, engine, store: VectorStore) -> RagAnswer:
    """Answer `question` using retrieved context and the local LLM.

    Steps:
      1. Retrieve: embed the question, rank stored chunks by similarity.
      2. Augment: inject the top chunks into the system prompt.
      3. Generate: one-turn chat completion on the local model.

    If nothing relevant is retrieved we short-circuit with the fallback
    answer instead of letting the model guess.
    """
    question = question.strip()
    if not question:
        return RagAnswer(question, "Please enter a question.")

    _check_embedder_match(engine, store)
    top_chunks = get_top_chunks(question, engine, store)
    if not top_chunks:
        return RagAnswer(question, FALLBACK_ANSWER)

    system_prompt = config.SYSTEM_PROMPT.format(context=build_context(top_chunks))
    answer = engine.chat(system_prompt, question)
    return RagAnswer(question, answer.strip(), top_chunks)
