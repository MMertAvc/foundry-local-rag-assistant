# Project Architecture: Local RAG Assistant

This project is a document Q&A assistant that runs entirely on one machine
with no internet dependency at question time. It follows the architecture from
the Microsoft Tech Community example of a local RAG support agent.

## Layers

The application has four layers, all local. The client interface is either a
command line loop (main.py) or a Streamlit web page (app.py). The pipeline
layer (rag/pipeline.py) orchestrates a query: it calls the retriever, builds
the augmented prompt, and invokes the model. The data layer is a SQLite file
(data/knowledge.db) holding document chunks and their embeddings. The AI layer
is Microsoft Foundry Local for the chat model, plus a pluggable embedding
backend: Foundry Local's embedding model on Windows 11 24H2+, or a local ONNX
Runtime embedder (fastembed with bge-small-en-v1.5) on Windows 10 — both
fully on-device.

## Ingestion flow

Ingestion runs once up front (and again whenever documents change). Each
document in data/docs is split into overlapping chunks of roughly 900
characters. Every chunk is embedded in a single batch call to the embedding
model, and each chunk plus its vector is inserted into SQLite.

## Query flow

When the user asks a question, the pipeline embeds the question with the same
embedding model, ranks all stored chunks by cosine similarity, and keeps the
top three that pass the similarity threshold. Those chunks, labeled with their
source document names, are injected into the system prompt. The local chat
model (phi-3.5-mini) then generates the answer, citing sources. If no chunk
passes the threshold, the assistant answers that it does not have the
information, without calling the model at all.

## Design decisions

Chunk overlap of about 150 characters prevents facts near a boundary from
being split away from their context. A similarity threshold of 0.30 filters
out weak matches so the model is not misled by irrelevant context. Brute-force
retrieval was chosen over a vector database because the collection is small
and the simpler code is easier for beginners to understand and debug.
