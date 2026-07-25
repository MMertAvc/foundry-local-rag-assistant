# Test Report — Local RAG Assistant

Date: 2026-07-25 · Suite: `pytest tests/` · Result: **31 passed, 0 failed** (0.83s)

## Methodology

The pipeline was designed so that everything except the model calls is pure
Python. A deterministic `FakeEngine` (tests/conftest.py) replaces the Foundry
Local SDK during testing: it embeds text into a keyword-count vector and
returns canned chat answers while recording every prompt it receives. This
makes the tests fast, reproducible, and runnable in CI without any models —
while still exercising the real chunking, storage, retrieval, and pipeline
code end to end.

## Automated test results

| Area | Test | Verifies | Result |
|---|---|---|---|
| Chunking | test_split_paragraphs_drops_empty | blank paragraphs removed | ✅ |
| Chunking | test_short_text_is_single_chunk | no unnecessary splitting | ✅ |
| Chunking | test_chunks_respect_size_limit | chunk size honored | ✅ |
| Chunking | test_oversized_paragraph_is_split | long paragraphs split at word boundaries | ✅ |
| Chunking | test_overlap_carries_context | boundary context preserved | ✅ |
| Chunking | test_no_content_lost | every token survives chunking | ✅ |
| Store | test_round_trip | embedding stored and read back exactly | ✅ |
| Store | test_count_and_clear | re-ingestion starts clean | ✅ |
| Store | test_persists_across_connections | knowledge base survives restarts | ✅ |
| Retrieval | test_cosine_identical/orthogonal/opposite/known_value | math matches hand-computed values | ✅ |
| Retrieval | test_cosine_zero_vector_is_safe | no division-by-zero crash | ✅ |
| Retrieval | test_cosine_length_mismatch_raises | wrong-model mixups fail loudly | ✅ |
| Retrieval | test_rank_orders_by_similarity | best chunk first, weak ones filtered | ✅ |
| Retrieval | test_rank_respects_top_k | never more than k chunks | ✅ |
| Retrieval | test_get_top_chunks_end_to_end | query → embedding → correct document | ✅ |
| Pipeline | test_ingest_populates_store | docs → chunks → DB | ✅ |
| Pipeline | test_reingest_replaces_old_data | idempotent rebuilds | ✅ |
| Pipeline | test_answer_uses_retrieved_context | retrieved text reaches the system prompt with source labels | ✅ |
| Pipeline | test_unanswerable_question_gets_fallback | off-topic question → "I don't know", model never called | ✅ |
| Pipeline | test_empty_question_is_handled | edge case: blank input | ✅ |
| Pipeline | test_ingest_records_embedder_name | knowledge base remembers its embedding backend | ✅ |
| Pipeline | test_embedder_mismatch_is_rejected | querying with a different backend fails loudly with a fix hint | ✅ |
| Store | test_meta_round_trip_and_overwrite | meta table stores/updates the embedder id | ✅ |
| Pipeline | test_build_context_labels_sources | citations possible | ✅ |
| Server API | test_stats_reflects_real_store | /api/stats reports real chunk/doc counts and embedder name | ✅ |
| Server API | test_ask_returns_answer_with_sources | /api/ask returns the answer, sources, and elapsed time | ✅ |
| Server API | test_ask_rejects_embedder_mismatch | /api/ask returns 400 on embedder mismatch instead of a stack trace | ✅ |
| Server API | test_ask_rejects_empty_question | /api/ask returns 400 on blank input | ✅ |

## Manual acceptance checklist (with real models)

These scenarios are checked live on a machine with Foundry Local models
downloaded, per the Week 5 plan:

1. Answerable question ("What is Foundry Local?") returns a grounded answer
   citing `01-foundry-local-overview.md`, in roughly 1–3 seconds on a typical
   laptop CPU.
2. Unanswerable question ("What is the capital of France?") returns the
   fallback answer, with no fabricated content.
3. General/vague question ("Tell me about this project") retrieves
   `06-project-architecture.md` and answers from it.
4. Airplane-mode test: with Wi-Fi disabled after ingestion, `ask` and `chat`
   keep working — confirming zero network dependency.
5. Streamlit UI: sidebar shows document/chunk counts; each answer exposes its
   retrieved chunks and similarity scores in the expander.
6. Custom web UI (`server.py` + `static/`): sidebar shows the same real
   chunk/doc counts and the real chat/embedder model names; the
   Retrieve → Augment → Generate indicator and source citations render
   correctly with real data (verified with Playwright, screenshots + zero
   console errors).
7. Offline check: with every request to a host other than 127.0.0.1 blocked
   (Playwright route interception), the custom web UI still loads and
   answers correctly — confirming zero network dependency, including the
   page's own JS runtime (React is vendored locally, not loaded from a CDN).

## Issues found and fixed during testing

Initial fake-embedding design mapped "no known keywords" onto a real keyword
dimension, which created phantom similarity between off-topic queries and
stored chunks — caught by `test_unanswerable_question_gets_fallback` and fixed
by adding a dedicated "unknown" dimension. This mirrors a real-world lesson:
retrieval quality bugs show up as wrong answers, so the fallback path needs
explicit tests.

A second real-world issue surfaced on the target machine: Foundry Local's
embedding models require Windows 11 24H2+ (build 26100) and are excluded from
the catalog on Windows 10 ("Skipping EP autoregistration ... Minimum supported
build is 26100"). Fixed by making the embedding backend pluggable
(`rag/embedders.py`) with an automatic fallback to a local ONNX embedder
(fastembed / bge-small-en-v1.5), plus a meta guard that rejects querying a
knowledge base built with a different backend. Both new behaviors are covered
by the three tests added above.

A third issue: the Foundry Local Python SDK looks for downloaded models in a
per-app cache directory (derived from `Configuration(app_name=...)`), which is
separate from the shared cache the `foundry` CLI downloads into — so a model
already fetched via `foundry model download` still failed with "Model path
does not exist" when loaded through the SDK. Fixed by pointing
`model_cache_dir` at the same shared cache the CLI uses. The SDK's default
device selection also picked a GPU/DirectML variant that fails to load on
this machine ("DML provider requested, but the installed GenAI has not been
built with DML support"); fixed by explicitly selecting the CPU variant.

A fourth issue surfaced only in the custom web UI: its runtime loads React
from a CDN (unpkg.com) by default, which silently broke the project's
zero-internet-dependency guarantee. Fixed by vendoring React/ReactDOM locally
(hash-verified against the CDN originals) and removing the Google Fonts CDN
call — see manual check 7 above.
