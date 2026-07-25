"""Local web server for the custom "Foundry RAG Assistant" UI.

Run with:
    python server.py

Serves the static UI (static/index.html + static/support.js) and a small
JSON API backed by the same rag/pipeline.py used by main.py and app.py.
The Foundry Local engine loads once at startup and is reused for every
request — loading it is expensive (~20-50s), answering with it is not.
"""

from __future__ import annotations

import time
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from rag import config
from rag.pipeline import answer_query
from rag.store import VectorStore

STATIC_DIR = Path(__file__).resolve().parent / "static"
SNIPPET_LENGTH = 240

_engine = None


def _get_engine():
    """Load the Foundry Local engine once and reuse it for every request."""
    global _engine
    if _engine is None:
        from rag.foundry import FoundryEngine

        print("Starting Foundry Local (first run downloads the models) ...")
        _engine = FoundryEngine()
    return _engine


async def api_stats(_request: Request) -> JSONResponse:
    with VectorStore(config.DB_PATH) as store:
        chunks = store.all_chunks()
    sources = sorted({c.source for c in chunks})
    engine = _get_engine()
    return JSONResponse(
        {
            "chunkCount": len(chunks),
            "docCount": len(sources),
            "documents": sources,
            "chatModel": config.CHAT_MODEL_ALIAS,
            "embedderName": engine.embedder_name,
            "isEmpty": len(chunks) == 0,
        }
    )


async def api_ask(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except ValueError:
        return JSONResponse({"error": "Invalid JSON body."}, status_code=400)

    question = (body.get("question") or "").strip()
    if not question:
        return JSONResponse({"error": "Please enter a question."}, status_code=400)

    engine = _get_engine()
    with VectorStore(config.DB_PATH) as store:
        started = time.perf_counter()
        try:
            result = answer_query(question, engine, store)
        except RuntimeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        elapsed = time.perf_counter() - started

    sources = [
        {
            "doc": sc.chunk.source,
            "chunkIndex": sc.chunk.chunk_index,
            "similarity": sc.score,
            "snippet": sc.chunk.content[:SNIPPET_LENGTH]
            + ("..." if len(sc.chunk.content) > SNIPPET_LENGTH else ""),
        }
        for sc in result.sources
    ]
    return JSONResponse(
        {"answer": result.answer, "elapsed": round(elapsed, 1), "sources": sources}
    )


routes = [
    Route("/api/stats", api_stats, methods=["GET"]),
    Route("/api/ask", api_ask, methods=["POST"]),
    Mount("/", app=StaticFiles(directory=str(STATIC_DIR), html=True), name="static"),
]

app = Starlette(routes=routes)


def main() -> None:
    _get_engine()  # load eagerly so the first request isn't slow
    print("Serving at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
