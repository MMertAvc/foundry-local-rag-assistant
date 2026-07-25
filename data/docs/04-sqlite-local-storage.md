# SQLite for Local Data Storage

SQLite is a serverless, self-contained SQL database engine. The entire
database lives in a single file on disk, there is no separate server process
to install or manage, and it is the most widely deployed database engine in
the world - it runs inside phones, browsers, and operating systems.

## Why SQLite fits this project

A local RAG assistant needs to persist document chunks and their embedding
vectors between runs. SQLite is ideal because it requires zero setup (Python
ships with the built-in `sqlite3` module), it is cross-platform, and the whole
knowledge base can be copied or backed up by copying one file.

## Storing embeddings

SQLite has no native vector type, so embeddings are serialized before storage.
The simplest approach, used in this project, is JSON encoding: the list of
floats becomes a TEXT column. An alternative is packing the floats into a BLOB
with Python's `struct` module, which is more compact. For small collections
the difference does not matter.

## The schema used here

The knowledge base has a single table called `chunks` with five columns: an
autoincrementing `id`, the `source` document file name (used for citations),
the `chunk_index` inside that document, the chunk `content` text, and the
JSON-encoded `embedding`. Retrieval loads all rows and computes similarity in
Python, which is the recommended pattern at this scale.

## Basic operations

Creating the table uses `CREATE TABLE IF NOT EXISTS` so the code can run
safely on every start. Inserts use parameterized queries with `?` placeholders
to avoid SQL injection issues. Re-ingestion simply deletes all rows first and
rebuilds the table, which keeps the update logic trivial.
