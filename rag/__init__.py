"""Local RAG Assistant built on Microsoft Foundry Local.

A fully offline document Q&A assistant:
retrieve relevant chunks from a SQLite vector store,
augment the prompt with that context, and
generate an answer with an on-device LLM.
"""

__version__ = "1.0.0"
