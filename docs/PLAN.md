# Plan

How to run it: [README](../README.md). Why it is built this way: [MVP_DESIGN.md](MVP_DESIGN.md).
How well it works: [MVP_EVAL.md](MVP_EVAL.md).

## MVP

- Ingestion pipeline:
  - [x] Upload PDFs (files or a folder), ingest in the background with progress, restart safe
  - [x] Parse PDFs with `PyMuPDF`: text and tables (infographics are not read in the MVP)
  - [x] Chunk per page, embed locally with `BAAI/bge-m3`, index in Postgres + pgvector
  - [x] Detect company name and fiscal year from the first pages of the report
  - [x] Pre-extract FTE and sustainability goals with the same retrieval as the chat
- Chat pipeline:
  - [x] Rewrite initial question + follow-ups into a one standalone question (query)
  - [x] Hybrid search (vector search + Postgres FTS), fused with RRF, reranked
    with `BAAI/bge-reranker-v2-m3`; abstain when the best rerank score is too low
  - [x] Structured output from the LLM: answer, citations (chunk id + quote + status `answered`/`partial`/`not_found`)
  - [x] Verbatim check of every quote: exact match, then fuzzy (RapidFuzz); answers without
    a verified quote become "not stated in the report"
  - [x] Anthropic and OpenAI behind one interface (local and smaller models are out of scope
    for the MVP)
- UI:
  - [x] Three pages: reports (ingestion), extracted data, chat
  - [x] Cited PDF page shown next to the answer
- Tests:
  - [x] Unit tests, parser quality on labelled pages, API integration test, CI
- Evaluation: out of scope for the MVP

## Final

**Evaluation** (results: [MVP_EVAL.md](MVP_EVAL.md))

- Gold set:
  - [x] Six reports: 3 companies × 2 fiscal years (2024, 2025)
  - [x] 52 questions: factoid, comparison, follow-up and unanswerable
  - [x] Labelled FTE figures and sustainability goals per report
- Ingestion:
  - [x] Time and cost per stage, metadata accuracy, parse and chunk statistics
- Retrieval:
  - [x] Measure hit@k, MRR and facts-in-context; ablate each stage
  - [x] Tune candidates, top-k and the abstention threshold; compare chunk sizes
- Extraction:
  - [x] Measure FTE accuracy and goal recall and precision against the gold labels
- Answers:
  - [x] Measure answer correctness, abstention, groundedness and verbatim rate (Anthropic)
  - [ ] Anthropic vs OpenAI vs local model
  - [ ] Human agreement check of the LLM judge

**Ingestion**

- [x] Embed all chunks of a report in one call (47% faster, identical vectors): slices of 512
- [ ] Strip repeated page furniture (navigation bars) before chunking

**Extraction**

- [x] Detect `max_tokens` truncation; raise the extraction output cap to 16,000
- [x] Better recall for goals: map-reduce over more candidate chunks
- [x] Clarify the registry instructions: FTE with or without temporary staff; goals dated
  after the fiscal year only
- [x] More registry entries: revenue, net profit, dividend per share (with gold labels)

**Retrieval**

- [x] Apply the tuned settings: 60 candidates, top 12, abstain below 0.2
- [x] Multi-search for comparison questions across companies and years (sub-queries)
- [ ] Filter each sub-query by its company and year, with the ingested company list in query analysis and a clear answer for unknown companies (try)
- [ ] Abstain when the question names a company that has no ingested report
- [ ] Fix or drop full-text search (the AND query matches nothing for half the questions)
- [ ] Expand a chunk to its page when it looks truncated
- [ ] Decide how (and should) to inject extracted data about the company to 

**Verification**

- [ ] Check that the numbers in an answer appear in its verified quotes
- [ ] Show what differed on a fuzzy match

**UI**

- [ ] Highlight the quote on the page image
- [ ] Stream the answer
- [ ] Sorting and filtering on the extracted data page
