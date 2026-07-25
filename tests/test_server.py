"""Tests for the custom web UI's JSON API (server.py), using FakeEngine.

Runs without Foundry Local installed: the module-level engine and DB path
are monkeypatched to the same FakeEngine/tmp-store fixtures used by the
rest of the test suite (see tests/conftest.py).
"""

import pytest
from starlette.testclient import TestClient

import server
from rag.ingest import ingest


def _write_docs(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "foundry.md").write_text(
        "Foundry model offline.\n\nFoundry runs the model offline on device.",
        encoding="utf-8",
    )
    return docs


@pytest.fixture
def client(engine, store, tmp_path, monkeypatch):
    docs = _write_docs(tmp_path)
    ingest(engine, store, docs_dir=docs)
    monkeypatch.setattr(server, "_engine", engine)
    monkeypatch.setattr(server.config, "DB_PATH", store.db_path)
    return TestClient(server.app)


def test_stats_reflects_real_store(client, engine):
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["chunkCount"] > 0
    assert data["documents"] == ["foundry.md"]
    assert data["chatModel"]
    assert data["embedderName"] == engine.embedder_name
    assert data["isEmpty"] is False


def test_ask_returns_answer_with_sources(client):
    response = client.post(
        "/api/ask", json={"question": "how does foundry run models?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answer"].startswith("FAKE ANSWER")
    assert data["sources"]
    assert data["sources"][0]["doc"] == "foundry.md"
    assert isinstance(data["sources"][0]["similarity"], float)
    assert "elapsed" in data


def test_ask_rejects_embedder_mismatch(client, engine):
    engine.embedder_name = "different:embedder"
    response = client.post(
        "/api/ask", json={"question": "how does foundry run models?"}
    )
    assert response.status_code == 400
    assert "Rebuild it" in response.json()["error"]


def test_ask_rejects_empty_question(client):
    response = client.post("/api/ask", json={"question": "   "})
    assert response.status_code == 400
