# Prompt Engineering for Q&A Assistants

How retrieved information is presented to the model matters as much as the
retrieval itself. Prompt engineering is the practice of writing instructions
that reliably steer the model toward the desired behavior.

## System versus user messages

Chat-style models accept messages with roles. The system message sets the
assistant's persona and rules and is the right place for RAG instructions and
the retrieved context. The user message carries the actual question. Keeping
these separated makes the prompt easier to maintain and the model more likely
to follow the rules.

## Rules that make RAG answers trustworthy

Three instructions dramatically improve answer quality in a RAG assistant.
First, restrict the model to the provided context: "Answer using only the
context below; do not use outside knowledge." Second, give it permission to
refuse: "If the context does not contain the answer, say you don't know."
Without this escape hatch, models tend to fabricate plausible-sounding
answers. Third, require citations: "Name the source document for every fact,"
which lets users verify answers themselves.

## Temperature and determinism

Temperature controls randomness in generation. For factual Q&A a low
temperature (0.0 to 0.3) is preferred: answers become more consistent and
stick closer to the context. Higher temperatures suit creative writing, not
document assistants.

## Iterating on prompts

Prompts are code and should be tested like code. A practical workflow is to
keep a small set of test questions (some answerable from the documents, some
deliberately not answerable) and re-run them after every prompt change,
checking that answerable questions get grounded answers and unanswerable ones
get the fallback response.
