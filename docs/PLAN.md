# Plan

How to run it: [README](../README.md). Why it is built this way: [DESIGN.md](DESIGN.md).

## MVP

- [x] Upload PDFs (files or a folder), ingest in the background with progress, restart safe
- [x] Parse with PyMuPDF, chunk per page, embed locally, index in Postgres + pgvector
- [x] Chat with follow-ups, hybrid search + reranking, abstains when the reports do not cover it
- [x] FTE and sustainability goals pre-extracted at ingestion, driven by a config registry
- [x] Every quote checked verbatim against the PDF text; unverified answers are downgraded
- [x] Anthropic or OpenAI behind one interface
- [x] Three-page UI with the cited page shown next to the answer
- [x] Unit, parser-quality and integration tests; CI

## Todo

**Evaluation** (results and recommendations: [MVP_EVAL.md](MVP_EVAL.md))

- [x] Gold set of 52 questions over six reports (three companies, 2024 and 2025), including unanswerable ones
- [x] Labelled FTE figures and sustainability goals for those reports
- [x] Measure retrieval recall, answer correctness, abstention and verbatim rate (Anthropic)
- [ ] The same for the OpenAI provider
- [ ] Human agreement check of the LLM judge

**RAG**

- [ ] Multi-search for comparison questions across companies
- [ ] Expand a chunk to its page when it looks truncated
- [ ] Apply the tuned retrieval settings: 60 candidates, top 12, abstain below 0.2
- [ ] Fix or drop full-text search (AND query matches nothing for half the questions)
- [ ] Strip repeated page furniture (navigation bars) before chunking

**Extraction**

- [ ] Detect `max_tokens` truncation; raise the extraction output cap
- [ ] Better recall for goals: map-reduce over more candidate chunks
- [ ] More registry entries
- [ ] Re-run extraction without re-ingesting

**Verification**

- [ ] Check that the numbers in an answer appear in its verified quotes
- [ ] Show what differed on a fuzzy match
- [ ] Highlight the quote on the page image

**UI**

- [ ] Stream the answer
- [ ] Rename conversations
- [ ] Browser smoke test
- [ ] Narrow-screen layout
