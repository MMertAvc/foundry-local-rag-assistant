# Embeddings and Vector Search

A text embedding is a numeric vector representation of a piece of text that
captures its semantic meaning. An embedding model maps sentences with similar
meaning to vectors that are close together in vector space, even when they
share no words. For example, "How do I reset my password?" and "I forgot my
login credentials" produce nearby vectors.

## Cosine similarity

The standard way to compare two embeddings is cosine similarity: the cosine of
the angle between the two vectors. It ranges from -1 to 1, where values close
to 1 mean the texts are semantically similar. Cosine similarity is computed as
the dot product of the two vectors divided by the product of their magnitudes.

## Semantic search in a RAG pipeline

In a RAG pipeline, every document chunk is embedded once during ingestion and
stored. At question time the query is embedded with the same model, and the
system ranks all stored chunks by cosine similarity to the query vector. The
top-K chunks (typically 2 to 5) become the context for generation.

It is critical to use the same embedding model for documents and queries.
Vectors from different models live in different spaces and cannot be compared
meaningfully.

## Scale considerations

For small collections (up to a few thousand chunks), brute-force comparison of
the query against every stored vector is fast and perfectly adequate. Larger
collections need approximate nearest neighbor indexes or dedicated vector
databases such as those with HNSW indexes. This project intentionally uses the
brute-force approach because the knowledge base is small and the code stays
simple and readable.

## The embedding models in this project

This project selects its embedding backend automatically. On Windows 11 24H2
and newer it uses the qwen3-embedding-0.6b model from the Foundry Local
catalog. On Windows 10, where Foundry Local excludes embedding models (they
require build 26100), it falls back to bge-small-en-v1.5 running locally
through ONNX Runtime via the fastembed package. Both options are small, fast
on a laptop CPU, and fully on-device, so the offline guarantee holds either
way. Because vectors from different models are not comparable, the knowledge
base records which backend built it and must be re-ingested if the backend
changes.
