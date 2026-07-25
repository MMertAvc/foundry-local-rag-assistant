# The Retrieval-Augmented Generation (RAG) Pattern

Retrieval-Augmented Generation, or RAG, is an AI design pattern that grounds a
language model's answers in your own documents. Instead of hoping the model
memorized the right facts during training, RAG retrieves relevant passages at
question time and hands them to the model as context.

## The three steps

RAG has three steps that give the pattern its name. Retrieve: given a user
question, find the most relevant passages in a document collection, usually
with embedding-based semantic search. Augment: insert those passages into the
model's prompt as context, typically inside the system message. Generate: ask
the model to answer the question using only the provided context.

## Why RAG reduces hallucinations

A plain language model answers from its training data, which may be outdated,
incomplete, or simply wrong about a niche topic. When the prompt explicitly
contains the relevant source text and instructs the model to answer only from
it, the model is far less likely to invent facts. RAG also enables source
citations, because the application knows exactly which document each retrieved
passage came from.

## RAG versus fine-tuning

Fine-tuning changes the model's weights and requires training data, compute,
and time. RAG requires none of that: updating the assistant's knowledge is as
simple as adding a document to the collection and re-running ingestion. For
document Q&A use cases, RAG is almost always the simpler and cheaper choice.

## When retrieval fails

If no stored passage is similar enough to the question, the correct behavior
is to say "I don't know" rather than to guess. A good RAG application enforces
this with a similarity threshold in the retriever and an explicit instruction
in the system prompt.
