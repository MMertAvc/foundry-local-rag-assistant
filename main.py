"""Local RAG Assistant - command line interface.

Usage:
    python main.py ingest              # build the knowledge base
    python main.py ask "..."           # answer a single question
    python main.py chat                # interactive Q&A loop
    python main.py stats               # show knowledge base info

Everything runs 100% offline on your machine via Microsoft Foundry Local.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from rag import config
from rag.ingest import ingest
from rag.pipeline import answer_query
from rag.store import VectorStore


def _make_engine():
    """Create the Foundry Local engine (loads/downloads models)."""
    from rag.foundry import FoundryEngine  # lazy import: needs the SDK

    print("Starting Foundry Local (first run downloads the models) ...")
    return FoundryEngine()


def cmd_ingest(_args) -> None:
    engine = _make_engine()
    with VectorStore() as store:
        started = time.perf_counter()
        total = ingest(engine, store)
        elapsed = time.perf_counter() - started
    print(f"Ingested {total} chunks into {config.DB_PATH} in {elapsed:.1f}s.")
    engine.close()


def _print_answer(result) -> None:
    print("\n" + result.answer + "\n")
    if result.sources:
        print("Retrieved context:")
        for sc in result.sources:
            print(f"  - {sc.chunk.source} (chunk {sc.chunk.chunk_index}, "
                  f"similarity {sc.score:.2f})")
        print()


def cmd_ask(args) -> None:
    engine = _make_engine()
    with VectorStore() as store:
        _require_ingested(store)
        started = time.perf_counter()
        result = answer_query(args.question, engine, store)
        elapsed = time.perf_counter() - started
    _print_answer(result)
    print(f"(answered in {elapsed:.1f}s, fully offline)")
    engine.close()


def cmd_chat(_args) -> None:
    engine = _make_engine()
    with VectorStore() as store:
        _require_ingested(store)
        print("\nLocal RAG Assistant ready. Ask a question, or type 'exit'.\n")
        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if question.lower() in {"exit", "quit", "q", ""}:
                break
            result = answer_query(question, engine, store)
            _print_answer(result)
    print("Goodbye!")
    engine.close()


def cmd_stats(_args) -> None:
    with VectorStore() as store:
        chunks = store.all_chunks()
        sources = sorted({c.source for c in chunks})
    print(f"Knowledge base : {config.DB_PATH}")
    print(f"Total chunks   : {len(chunks)}")
    print(f"Documents ({len(sources)}):")
    for source in sources:
        count = sum(1 for c in chunks if c.source == source)
        print(f"  - {source}: {count} chunks")


def _require_ingested(store: VectorStore) -> None:
    if store.count() == 0:
        print("Knowledge base is empty. Run: python main.py ingest")
        sys.exit(1)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        prog="local-rag-assistant",
        description="Offline document Q&A with Microsoft Foundry Local + RAG.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="(Re)build the knowledge base").set_defaults(
        func=cmd_ingest
    )
    ask = sub.add_parser("ask", help="Ask a single question")
    ask.add_argument("question", help="The question to answer")
    ask.set_defaults(func=cmd_ask)
    sub.add_parser("chat", help="Interactive Q&A loop").set_defaults(func=cmd_chat)
    sub.add_parser("stats", help="Show knowledge base statistics").set_defaults(
        func=cmd_stats
    )

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
