"""Shared test fixtures.

FakeEngine replaces the Foundry Local SDK in tests: it produces
deterministic keyword-based embeddings and canned chat answers, so the
whole pipeline can be tested quickly with no models installed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest

# Make the project root importable when running `pytest` from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.store import VectorStore  # noqa: E402

# Simple keyword vocabulary -> one embedding dimension per keyword.
_VOCAB = ["foundry", "rag", "embedding", "sqlite", "prompt", "cosine",
          "offline", "model", "chunk", "database"]


class FakeEngine:
    """Drop-in replacement for FoundryEngine used in unit tests."""

    def __init__(self) -> None:
        self.chat_calls: List[dict] = []
        self.embedder_name = "fake:keyword-v1"

    def embed_text(self, text: str) -> List[float]:
        lowered = text.lower()
        vector = [float(lowered.count(word)) for word in _VOCAB]
        # Extra "unknown" dimension guarantees a non-zero vector when the
        # text contains no vocabulary words (keeps cosine well-defined
        # without creating fake similarity to real keyword dimensions).
        vector.append(0.001 if not any(vector) else 0.0)
        return vector

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]

    def chat(self, system_prompt: str, user_message: str) -> str:
        self.chat_calls.append(
            {"system": system_prompt, "user": user_message}
        )
        return f"FAKE ANSWER to: {user_message}"


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def store(tmp_path):
    s = VectorStore(tmp_path / "test.db")
    yield s
    s.close()
