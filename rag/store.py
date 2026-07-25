"""SQLite-backed vector store.

Each row holds one document chunk and its embedding vector
(JSON-serialized). SQLite is serverless and single-file, which keeps
the whole knowledge base portable and 100% local.

For our scale (tens to hundreds of chunks) brute-force cosine
similarity in Python is more than fast enough; a dedicated vector
database only becomes necessary at much larger scale.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, NamedTuple

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT NOT NULL,      -- document file name (used for citations)
    chunk_index INTEGER NOT NULL,   -- position of the chunk inside the doc
    content     TEXT NOT NULL,      -- the chunk text itself
    embedding   TEXT NOT NULL       -- JSON-encoded list of floats
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,         -- e.g. 'embedder'
    value TEXT NOT NULL
);
"""


class StoredChunk(NamedTuple):
    id: int
    source: str
    chunk_index: int
    content: str
    embedding: List[float]


class VectorStore:
    """Minimal CRUD layer over the `chunks` table."""

    def __init__(self, db_path: Path | str = config.DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def clear(self) -> None:
        """Remove all chunks (used when re-ingesting)."""
        self._conn.execute("DELETE FROM chunks")
        self._conn.commit()

    def add_chunk(
        self, source: str, chunk_index: int, content: str, embedding: List[float]
    ) -> None:
        self._conn.execute(
            "INSERT INTO chunks (source, chunk_index, content, embedding) "
            "VALUES (?, ?, ?, ?)",
            (source, chunk_index, content, json.dumps(embedding)),
        )

    def commit(self) -> None:
        self._conn.commit()

    def all_chunks(self) -> List[StoredChunk]:
        """Load every chunk with its embedding (fine at our scale)."""
        rows = self._conn.execute(
            "SELECT id, source, chunk_index, content, embedding FROM chunks"
        ).fetchall()
        return [
            StoredChunk(r[0], r[1], r[2], r[3], json.loads(r[4])) for r in rows
        ]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    # --- Metadata (e.g. which embedder built this knowledge base) -----------

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self._conn.commit()

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def close(self) -> None:
        self._conn.close()

    # Context-manager support: `with VectorStore() as store: ...`
    def __enter__(self) -> "VectorStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
