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

**Evaluation**

- [ ] Golden set of about 40 questions, including unanswerable ones
- [ ] Hand-labelled FTE figures and goals for the five reports
- [ ] Measure retrieval recall, answer correctness, abstention and verbatim rate, per provider

**RAG**

- [ ] Multi-search for comparison questions across companies
- [ ] Expand a chunk to its page when it looks truncated
- [ ] Tune candidates, top-k and the abstention threshold once the eval exists

**Extraction**

- [ ] Better recall for goals by scanning all sustainability sections
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
