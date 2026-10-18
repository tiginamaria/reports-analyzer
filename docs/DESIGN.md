# Annual Report RAG Analyzer: Design

This document explains how the application is built, why each part is built the way it is,
and what I would change next. It follows the order of [PLAN.md](PLAN.md): the ingestion
pipeline, the chat pipeline, the UI, the tests and the evaluation. For every component there
is a "how it works today" part and a "what could be improved" part.

Related documents: 
* [README](../README.md) explains how to run it
* [PLAN.md](PLAN.md) lists what is done and what is open
* [EVAL.md](EVAL.md) reports how well every stage works on a gold set of six reports and 57 questions.

## 1. What the application does

A local full-stack Retrieval-Augmented Generation (RAG) application for company annual
reports in PDF form. A user can:

1. **Upload reports** (one PDF or a whole folder). Each report is parsed, chunked, embedded
   and indexed in the background.
2. **Browse extracted data**: a table of datapoints that were pulled out of every report at
   ingestion time (FTE, revenue, net profit, dividend per share, sustainability goals), each
   with a quote and a link to the page it came from.
3. **Chat with the reports**: ask questions in natural language, including follow-ups and
   comparisons across companies and years. Every answer cites passages of the report, and
   the cited page opens next to the answer.

The one design principle that runs through everything: **every figure the application shows
comes with a verbatim quote, and the quote is checked in code against the PDF text**.

## 2. Architecture

```mermaid
flowchart LR
    user([Browser]) --> fe["React + Vite<br/>Reports · Extracted data · Chat"]
    fe -- "HTTP / SSE" --> api["FastAPI"]

    subgraph backend[Backend, one process]
        api --> ing["Ingestion pipeline<br/>parse · detect · chunk · embed · extract · verify"]
        api --> chat["Chat pipeline<br/>rewrite · retrieve · rerank · answer · verify"]
        ing --> emb["Local models<br/>bge-m3 embedder<br/>bge reranker"]
        chat --> emb
        ing --> llm["LLM provider interface"]
        chat --> llm
    end

    ing --> db[("Postgres + pgvector<br/>reports · pages · chunks<br/>datapoints · conversations")]
    chat --> db
    ing --> files[("PDF store on disk<br/>named by SHA-256")]

    llm --> claude["Claude API"]
    llm --> openai["OpenAI API"]
```

**Three services, one command.** `docker compose up` starts Postgres with the pgvector
extension, the FastAPI backend and the React frontend. The frontend container is nginx: it
serves the built app and proxies `/api` to the backend, so the browser talks to one origin
and there is no CORS configuration. The Vite dev server proxies the same way when the app
runs on the host.

**One backend process, several threads.** The HTTP API, ingestion, the local embedding and
reranking models and the calls to the LLM provider all live in one FastAPI process; there
is no task queue and no separate model server. Each upload starts its own ingestion as a
background task, which runs in a worker thread after the response is sent, so several
reports ingest in parallel while the API keeps answering requests; the two local models are
loaded once and shared between threads behind a lock. This keeps the system small to start
and debug, and it sets the main limits of the current design, which come back in the
improvements below: an ingestion dies with the process, and ingestion and chat share one
CPU or GPU.

**Two pipelines share the same building blocks.** Ingestion and chat both use the same
hybrid search, the same reranker, the same LLM interface and the same quote verifier. The
pre-extraction of datapoints is literally "run the chat's retrieval with a fixed set of
queries and a fixed instruction". That keeps the code small and means an improvement to
retrieval helps both features.

**Where state lives.** Postgres holds everything relational, the vectors and the full-text
index. PDFs are stored on a bind-mounted folder under their content hash. Both survive a
`docker compose down`. The local models are cached on a named volume, because the first
download is about 4.5 GB.

### Request flows

*Upload.* The browser posts one or many files. For each file the backend checks the PDF
magic bytes, hashes it, rejects duplicates, writes the file under its hash, creates a report
row in status `pending` and schedules a background task. The response lists what started and
what was rejected and why; the Reports page then polls every 2.5 seconds while anything is
still ingesting.

*Chat.* The browser posts the question and gets a stream of server-sent events: a
`conversation` event with the conversation id, `stage` events (`analyzing`, `retrieving`,
`generating`) that drive the status line in the UI, then exactly one `answer` or `error`.
The pipeline is blocking (database, local models, LLM calls), so the endpoint runs it in a
worker thread and passes events to the event loop through a queue. The answer is stored
before it is sent, in the same shape the UI gets when it reloads a conversation, so what the
user saw live is what they see later.

## 3. Tech stack

| Layer | Choice | Why this one |
|---|---|---|
| Backend | Python 3.12, FastAPI, Pydantic v2, SQLAlchemy 2 + Alembic, `sse-starlette`, `uv` | Typed end to end: the same Pydantic models define the LLM output schemas, the API responses and the validation. One process, async HTTP, server-sent events for the chat |
| Database | Postgres 16 + pgvector | One store for relational data, vectors and full-text search, with real transactions and cascading deletes. No second system to keep in sync |
| PDF parsing | PyMuPDF | Fast on CPU, gives text blocks in reading order and table boundaries, no external service. Scored 1.00 on a labelled set of 26 hard pages (see Tests) |
| Embeddings | `BAAI/bge-m3`, run locally through sentence-transformers | No API key, no per-token cost, multilingual, 8k-token context. The index stays valid when the LLM provider changes |
| Reranker | `BAAI/bge-reranker-v2-m3`, local cross-encoder | The biggest single quality gain in retrieval (+12 points hit@8 over vector search alone). Its score is a probability, so it can also drive abstention |
| LLM | Claude (default) or OpenAI, selected in `.env` | Strong structured output and prompt caching. Two providers behind one interface prove the design is not tied to a vendor |
| Frontend | React 19 + Vite + TypeScript, Tailwind v4, react-router 7 | Three pages, a typed API client, a hand-written SSE parser; no component library |
| PDF viewing | The backend renders a cited page to PNG | One request per citation instead of loading a 400-page PDF in the browser; the full PDF is one click away |
| Tooling | ruff, mypy, pytest; eslint, prettier, tsc; Makefile | Lint and type checks on both sides |

Model roles, configurable in `backend/app/config.py`:

| Role | Anthropic | OpenAI | Used for |
|---|---|---|---|
| Main model | `claude-sonnet-5-5` | `gpt-5` | Answers, datapoint extraction |
| Small model | `claude-haiku-4-5` | `gpt-5-mini` | Company and year detection, query rewriting |

The split keeps the cheap, frequent, easy calls on the small model and the calls where
reading comprehension matters on the main one. The earlier evaluation tried Haiku for
answers: 25% of its quotes failed verification, so Sonnet stayed.

## 4. Ingestion pipeline

Ingestion runs as a FastAPI background task with its own database session and moves a report
through the statuses `pending`, `parsing`, `chunking`, `embedding`, `extracting`, `ready`
(or `failed`). After every stage, and within the long stages after every batch, the status
and a progress percentage are written to the report row. If anything raises, the error text
is stored on the report and shown in the UI. Reports that are still in progress when the
process starts are marked failed with the message "interrupted by a restart", because an
in-process task cannot be resumed.

On an M-series Mac a 400-page report takes about 4 minutes end to end (16% parsing, 22%
embedding, 62% extraction); on CPU in Docker, 10 to 15 minutes. The LLM cost per report is
16 to 25 cents, almost all of it in extraction.

### 4.1 Upload and storage

The upload box accepts files or a folder; folders are walked
recursively in the browser and every PDF is sent in one multipart request. Each file is
checked for the `%PDF-` header, hashed with SHA-256 and stored as `<hash>.pdf`. The hash is a
unique column, so the same bytes can never be ingested twice, even if two uploads race (the
loser gets a clear "already ingested as Shell 2025" rejection). Rejections never fail the
batch: the response lists started reports and rejected files separately. Deleting a report
removes the row, and the database cascades to its pages, chunks and datapoints without
loading thousands of vectors into memory; the file is unlinked afterwards.

### 4.2 Parsing

PyMuPDF returns text blocks with their coordinates. Blocks are sorted
top to bottom, left to right. Sorting whole blocks rather than lines matters: it keeps each
paragraph intact and keeps the two columns of a two-column page in order. Tables come from
PyMuPDF's table finder, but only boundaries are used: a detected table is kept only when it
has at least 2 rows and 3 numeric cells, which drops layout grids and navigation bars that
the finder also reports. For each kept table the raw text inside its boundary is stored
rather than the cell grid, because cell extraction merges the columns of borderless tables,
which is how most annual-report tables are drawn. Each page ends up with two texts: the full
text (the source of truth for quote verification) and a body text with table regions removed
(what gets chunked). Words hyphenated across lines are rejoined.

The parser has its own quality dataset: 26 pages cut from the real reports (financial
tables, five-year tables, infographics, two-column text, pages with navigation bars),
labelled with text that must be recovered, cells that must land in a table, pages that must
have no table, and reading-order pairs. All four categories score 1.00.

**Improvements**

- *Page furniture.* Running headers and navigation bars ("HOME CONTENTS MANAGEMENT BOARD
  REPORT ...") are read as text on almost every page of some reports. In CM.com's reports
  the bar sits at the start of about half of the chunks and dilutes their embeddings; the
  evaluation traces two extraction misses to it. Stripping lines that repeat on 20% or more
  of a report's pages before chunking is cheap and needs only a re-ingest.
- *Infographics and charts.* Key-figures pages drawn as graphics come out as digits without
  labels, so a query naming the metric cannot find them. Pages with little text and many
  numbers could go to a vision model that returns the figures with their labels; a
  layout-aware parser (Docling, or PyMuPDF's layout mode) is the cheaper first step.
- *Table representation.* A table chunk is the raw text in reading order; a row's label and
  its column header can be far apart in the text. Rendering tables as rows with the header
  repeated would help both retrieval and the model's reading of them (two of the four chat
  failures in the evaluation are tables the model skimmed).

### 4.3 Chunking, embedding and indexing

Each page body is cut into windows of 400 words with 60 words of
overlap, so a sentence cut by a window boundary is whole in the next one. Each table is one
chunk of its own, never cut and indexed once. Chunks under 5 words are dropped. A chunk
never spans pages, because the page is the citation unit. No section structure is inferred:
headings in these reports are too inconsistent to split on reliably, and the earlier
evaluation found 400/60 better than 200 and no worse than 600.

Chunk ids are readable, for example `shell-2025-a1b2c3-p87-c412` (company, year, a fragment
of the report id, page, running number), because the LLM has to copy them into citations
and a readable id is copied more reliably than a UUID.

Before embedding, every chunk gets a contextual header: `[Shell | Annual Report 2025 |
p. 87]`. The same header is put in front of the chunk wherever the text is used: embedding,
reranking and the answer model's context.

Embedding runs locally with `bge-m3`, normalised vectors, in slices of 512 chunks per
`encode` call. Progress is updated after every slice. The
vectors go into a pgvector column; a generated `tsvector` column with a GIN index provides
full-text search over the same rows. On a Mac the models run on Metal; loading a second
model while the first is encoding crashes Metal, so model loading and inference share one
lock.

**Improvements**

- *An approximate-nearest-neighbour index.* Vector search today is an exact scan ordered by
  cosine distance. At 5,000 chunks that is milliseconds; at hundreds of reports it would
  need an HNSW index, which pgvector supports with one migration.
- *Small-to-big retrieval.* The full page text is already stored. A chunk that looks cut
  (ends mid-sentence, or is a table fragment) could be expanded to its whole page before it
  is sent to the model. The page is the citation unit anyway.
- *Chunk headers with a section name.* The header carries company, year and page. If
  section titles could be recovered reliably (from the PDF outline when the report has one),
  adding "Section: Climate" to the header would help the many questions that are about a
  topic rather than a figure.
- *Parallel ingestion.* Embedding is single-threaded and the GPU is shared with the chat.
  Running two reports at once gives little on one GPU; the real gain is in extraction (below).

### 4.4 Metadata detection

The small model reads the first three pages (2,500 characters each)
and the file name and returns the company's short name, the fiscal year and the title as a
JSON object. This runs right after parsing, before chunking, because the chunk ids and the
contextual headers need the company and year. If the call fails, the report keeps the file
name as its company and year 0, so it stays visible and can be deleted. In the evaluation
it was right for 6 of 6 reports and takes about one second.

**Improvements**

- The detected name is what later scoping relies on ("ABN AMRO" in a question has to match
  "ABN AMRO" in the header). A normalised company list, or letting the user correct the
  detected values in the Reports table, would make that robust against "Shell plc" versus
  "Shell".
- Fiscal years that do not match calendar years (a report "for the year ended 31 March
  2025") are stored as one integer. That is enough for these reports, but a start and end
  date would be the general answer.

### 4.5 Pre-extraction of datapoints

The datapoints the application extracts are defined in a
configuration file, `extraction/registry.yaml`, not in code. Each entry has a key, a label, a
flag for single or multi-value, a list of retrieval queries, the number of chunks to pass to
the model (`top_k`) and extraction instructions in plain English. Five entries ship: FTE,
revenue, net profit and dividend per share (single value, 3 queries and 10 chunks each) and
sustainability goals (multi-value, 6 queries and 32 chunks).

For each entry the extractor runs every query through the same hybrid search and reranker
the chat uses, restricted to the report being ingested and with the reranker floor turned
off (recall matters more than precision here). It keeps a few chunks per query, takes the
union, cuts it to `top_k` and passes the chunks to the main model in page order, so the model
reads the report front to back. The model returns a list of items, each with a label, the
value exactly as written, unit, period, qualifiers (for example baseline year and scope), a
verbatim quote and the id of the chunk it was copied from. Items that repeat the same label,
value and period, or the same quote, are deduplicated. Every quote is then verified against
the stored page text (see 5.4) and the result is stored with a `verified`, `fuzzy` or `failed`
badge. The value is also parsed to a number (handling "23,500", "1.2 million", "(3,786)" for a
loss) for sorting and for the evaluation.

The instructions in the registry carry the domain knowledge: prefer the group-wide year-end
FTE over the average, "total operating income" is the bank's revenue, a dividend per share is
the sum of interim and final, a goal whose target year is already past is a result rather
than a goal. The shared system prompt adds the rules that hold for every entry: only use
the chunks, never round or rescale a value, prefer the most precise statement of a figure.

Results on the six evaluation reports: every single-value figure extracted was exact or a
defensible alternative the report also states, none was wrong, one was missing. Goal
precision is 0.97 and recall 0.67 (0.84 on the headline goals).

**Improvements**

- *Run the extraction calls concurrently.* The ten calls per report are independent, and
  extraction is 62% of ingestion time. Four in parallel would take a report from about 4
  minutes to about 2 at the same cost.
- *Page-level extraction for goals.* The ceiling on goal recall is retrieval: the queries
  reach only 66% of the pages that hold a gold goal, and the model can only list what it is
  given. Retrieval finds the pages well enough; the better design is to use the candidate
  chunks to locate the sustainability pages and then send those whole pages (or the whole
  chapter when the table of contents names one) in one call. A batched map-reduce over more
  chunks was tried during the evaluation and is not in the current code; page expansion
  attacks the cause rather than the symptom.
- *Revenue precision.* Two of the revenue "alternatives" are rounded figures taken from prose
  when the income statement was among the chunks. The prompt now says to prefer the most
  precise figure; a query that names the income statement explicitly is the config-level fix.
- *User-defined datapoints.* The registry is a file in the repository. A small editor in the
  UI, or at least an upload of a registry file, would let an analyst add a field without a
  deployment.

## 5. Chat pipeline

The chat answers a question in five steps: rewrite the question so it stands alone and split
it into sub-queries, retrieve candidate chunks for each sub-query, rerank them and decide
whether to abstain, generate a structured answer with citations, and verify every quote. The
median question takes 7.5 seconds (about 1 s rewriting, 4 s retrieving and reranking, 2.5 s
generating) and costs about 1.8 cents.

### 5.1 Query rewriting and decomposition

Retrieval needs a standalone query, but users write "and the year
before?". The small model gets the conversation so far and the latest question and returns
two things. First, the question rewritten to stand alone, in the user's language, with "it",
"they" and "that year" resolved; the answer is written for this question and it is shown in
the UI as "searched for ...". Second, a list of sub-queries: one per company, fiscal year
and topic the question is about, each a standalone English question naming its company and
year. "ASML vs ABN AMRO headcount in 2025" becomes two sub-queries, as does "ASML net sales
2024 vs 2025". The model fills the company and year fields before it writes the question,
and if the question text still lacks them, retrieval appends them ("... ASML annual report
2025"), phrased like the chunk headers, so every search names the report it is about. At
most ten sub-queries are searched. If the call fails, the raw question is searched as is, so
the chat keeps working.

What the rewrite sees is deliberately asymmetric: every earlier question (up to 30) but only
the answers within the last six messages. Questions are short and carry the company and year
a follow-up refers to, often many turns back; answers are long and only matter for what was
just said. The raw questions are used rather than their earlier rewrites, so one wrong
rewrite never contaminates the next.

Decomposition is the change that made comparison questions work: 63% correct with one query
in the earlier evaluation, 100% with sub-queries now, because each side of the comparison
gets its own retrieval budget and is not crowded out by the other's scores.

**Improvements**

- *Tell the rewrite which companies exist.* The model does not know the corpus. If it got the
  list of ingested companies and years, it could map "the bank" to ABN AMRO and say plainly
  when a question names a company that has no report, instead of letting retrieval score
  low and abstain.
- *An agentic variant for open comparisons.* "Which company has the most employees?" needs
  one search per ingested company; today the rewrite has to guess them. Giving the model a
  `search_reports(query, company, year)` tool and letting it call it several times is the
  natural extension, at the cost of latency and predictability. The current fixed pipeline
  is easier to evaluate, which is why it came first.

### 5.2 Hybrid retrieval, reranking and abstention

For each sub-query, two searches run over all indexed reports (or
over the reports the user selected in the sidebar, if any): a pgvector cosine search on the
embedded query and a Postgres full-text search that matches any word of the query, ranked by
`ts_rank_cd`. Each returns 60 candidates. The two rankings are fused with Reciprocal Rank
Fusion (k = 60), a method that only looks at ranks, so the two very different score scales
never have to be calibrated against each other. The fused top 60 go to the cross-encoder
reranker, which reads query and chunk together and returns a probability that the chunk
answers the query. Chunks below 0.2 are dropped and the top 8 are kept per sub-query; the
sub-queries' chunks are merged, a chunk found twice kept once with its best score.

The 0.2 floor is the first abstention gate. In the evaluation, questions about companies
outside the corpus scored 0.10 to 0.18 while every answerable question's best chunk scored
0.96 or more. When nothing survives the floor, or there is no indexed report to search, the
pipeline answers "not stated in the report" without calling the main model and says which of
the two it was. Plausible questions the reports do not answer (the ATM count of a bank) score
high anyway because the reports discuss the topic, so for those the model has to say so, and
in the evaluation it did for every one.

Reports are searchable as soon as their index is complete, while extraction is still running.
A user filter is never widened: if it matches no indexed report, the answer is "not stated",
not a search over everything.

Retrieval numbers on the gold set: a gold page among the retrieved chunks for 95.7% of
answerable questions, every key fact retrieved for 93.5%, MRR 0.61.

**Improvements**

- *Filter each sub-query by its company and year.* This is the one change with a clearly
  measured cause. The rewrite already outputs both fields, but they are only appended to the
  search text. The result is "year leakage": 79% of retrieved chunks are from the right
  company and year, and the rest are mostly the other year of the same company, because a
  2025 report restates 2024 figures as comparatives and the reranker cannot tell them apart.
  Passing company and year as a report filter was measured at 0.98 against 0.90 for the gold
  page in context. It was held back because it fails silently on an unknown company or a
  wrong year; the fix is to combine it with the ingested-company list in the rewrite and a
  fallback to the unfiltered search when the filter matches nothing.
- *Keep 12 chunks instead of 8.* The earlier evaluation chose 12 (hit@k 0.95 against 0.90);
  the shipped constant went back to 8 when sub-queries multiplied the context. With the year
  filter in place, the extra chunks would all be from the right report.
- *Fix or drop full-text search.* It was added for exact terms (FTE, Scope 3, EUV) where
  embeddings are weak, but whole-sentence queries make it nearly useless: in the ablation,
  hybrid plus reranker equals vector plus reranker. Building the full-text query from the
  sub-query's key terms, or a BM25 index over chunk text, would make it earn its place;
  otherwise it is one search per sub-query that can go.
- *Reranker throughput.* Reranking 60 pairs per sub-query is the largest part of latency
  (about 60 ms per pair on Metal). Batching the sub-queries' pairs into one call, or a
  smaller reranker for the first cut, would bring a three-sub-query question from 13 s
  towards 6 s.

### 5.3 Answer generation with structured output

The main model gets the whole conversation as real user and
assistant turns, and the final user message holds this question's chunks (each prefixed with
its citable id and its contextual header) followed by the standalone question. Earlier turns
carry only question and answer text, never their chunks, so every answer is grounded in and
verified against this turn's evidence alone. Sending the whole conversation keeps the prompt
prefix append-only, which is what makes the provider's prompt cache work: the system prompt
and all earlier turns are served from cache and only the final message is paid in full.

The reply is not free text. The model must return a JSON object that validates against a
Pydantic schema: a status (`answered`, `partial`, `not_found`), the answer as short plain
prose in the user's language with every figure carrying its unit and period, and a list of
citations, each a chunk id and a quote copied character for character from that chunk. The
instructions say to prefer the sentence or table row that contains the number, never to use
outside knowledge, and to use `partial` when the report gives a related but different figure
(mitigation spending when adaptation spending was asked).

Results: 91.3% of answerable questions correct, 95.7% correct or partial, 0% incorrect, all
11 unanswerable questions handled without inventing a figure.

**Improvements**

- *Check the numbers in the prose, not only the quotes.* Quote verification proves the quotes
  are real; it says nothing about the sentence around them. In the evaluation 16% of correct
  answers mention a side figure (a baseline, a one-off) that the model read in a chunk but
  did not quote. A code check that every number in the answer appears in a verified quote,
  with scale normalisation so "28.3 billion" matches "28,262.9 million", would flag these as
  "figure not quoted". A prototype caught every such case with some false alarms on rescaled
  figures.
- *Streaming the answer.* The stages stream, the answer does not, because structured JSON is
  validated whole. Streaming the `answer` field as it is generated and attaching citations at
  the end is possible with incremental JSON parsing, and would make the 2 to 3 seconds of
  generation feel shorter.
- *Table reading.* Both generation failures in the evaluation were tables the model skimmed
  (it used a rounded prose figure when the exact table row was in context). Asking the model
  to check table chunks cell by cell for the asked metric and year is a prompt change worth
  one evaluation run; a better table rendering (4.2) is the structural fix.

### 5.4 Quote verification

Every quote, from the chat and from extraction, goes through the
same function. It answers one question: is this text really on that page of that report?

1. A citation of a chunk id the model was never given fails immediately.
2. The sources are the stored text of the chunk's page (the source of truth) and the chunk's
   own text. The second matters for tables: a table chunk is text clipped to the table region
   and can read differently from the page's block text. A match in either counts.
3. Quote and sources are normalised: Unicode NFKC, typographic quotes and dashes unified,
   ligatures expanded, soft hyphens removed, whitespace collapsed, lower-cased. Models do not
   reproduce PDF typography and should not fail for it.
4. An exact substring match is `verified`. Otherwise a RapidFuzz `partial_ratio`, the best
   alignment of the quote against any window of the source, is computed; at 0.92 or above the
   quote is `fuzzy`, below it `failed`. A quote longer than its source is compared whole, so
   a short table chunk hidden inside a long invented quote cannot score 1.

What the status does: a verified citation gets a green badge, a fuzzy one an amber badge and
still counts as support, a failed one is shown struck through and counted in the message. An
`answered` reply whose citations all failed is downgraded to `not_found` and its text is
replaced by "not stated in the report", because it states figures nothing in the report
supports (the original text is kept in the message metadata for debugging). Datapoints are
stored whatever the status and show it as a badge, so a failed one is visible rather than
silently dropped.

The threshold was chosen on a generated set of 570 quote variants from the fixture reports:
real quotes with small edits passed 96% of the time, invented text was rejected 100%. In the
evaluation, 106 of 106 chat citations verified exactly.

**Improvements**

- *Fuzzy is a warning, not a proof.* A changed digit or an inserted "not" is a small edit and
  scores above 0.92. That is why fuzzy matches are shown in amber. A stricter check for
  fuzzy matches (the digits of the quote must appear unchanged in the matched window) would
  close the digit case; it was not needed in practice because exact matches dominate.
- *Show the difference.* When a quote is fuzzy, showing what differed would let the user
  judge it in a second instead of opening the page.
- *Highlight the quote on the page image.* The citation opens the right page; the user still
  has to find the sentence. PyMuPDF can search a page for text and return rectangles, so a
  highlighted page image is one more endpoint.

### 5.5 LLM providers

Anthropic and OpenAI sit behind one small interface: a system prompt,
optional earlier turns, the current user message, a Pydantic schema and a `small` flag that
selects the model. Each adapter uses its provider's native JSON-schema mode (every property
required, no additional properties). The reply is validated with Pydantic; on failure the
validation error is sent back once as a "repair retry", and a second failure raises a clear
error. Provider exceptions (authentication, rate limit, connection) are mapped to one error
type with a message the UI can show.

The Anthropic adapter streams every response, so a long goal list (tens of thousands of
tokens) is not cut by the HTTP timeout, detects truncation at `max_tokens` and reports it as
an error rather than letting invalid JSON fail twice, and places cache breakpoints on the
system prompt and the last earlier turn. OpenAI caches shared prefixes automatically, so its
adapter only has to keep the prefix byte-identical. Token usage, including cache reads and
writes, is logged for every call and the evaluation harness hooks into it to compute cost.

**Improvements**

- *Local and smaller models.* The interface would take an Ollama or vLLM adapter in an
  afternoon; what is missing is the evaluation of whether a local model keeps quotes verbatim
  (Haiku did not, 25% of its quotes failed). The provider comparison in the plan is open.
- *Cost and tracing in the product.* Usage is logged, not stored. Writing tokens and cost per
  call to the database, or sending traces to a tool like Langfuse, would answer "what did
  this report cost to ingest" from the UI.
- *Retries with backoff.* A rate-limit error is reported to the user; the ingestion's
  extraction step logs it and continues without that datapoint. A short retry with backoff
  would recover most of these.

## 6. Data model

Six tables in Postgres, created by one Alembic migration:

- `reports`: company, fiscal year, title, original file name, SHA-256 (unique), page count,
  status, progress, error. Company and year are empty until metadata detection has run.
- `pages`: one row per PDF page with its full text. The source of truth for verbatim checks
  and the unit of citation.
- `chunks`: readable id, report, page, kind (`text` or `table`), content, embedding
  (pgvector), word count, and a generated `tsvector` column with a GIN index for full-text
  search.
- `datapoints`: report, registry key, label, value as written, parsed numeric value, unit,
  period, qualifiers (JSON), quote, chunk id, page, verification status, the model that
  produced it.
- `conversations` and `messages`: chat history. An assistant message stores its status, its
  citations with their verification results, and metadata such as the standalone question,
  the sub-queries, the retrieved chunk ids and the best reranker score, so a reloaded
  conversation shows exactly what was shown live and a bad answer can be debugged from the
  database alone.

Deletes cascade in the database. Everything survives a restart.

## 7. User interface

Three pages behind a header that always shows the configured provider,
models and reranker, so a demo states what produced the answers.

- *Reports.* Drop PDFs or a folder, upload in one request, see per-file rejections, a status
  badge and a progress bar per report, polling only while something is ingesting, delete with
  a confirmation.
- *Extracted data.* One section per registry entry, across all companies, with a report
  filter. Single-value entries (FTE, revenue, net profit, dividend) show the value with its
  unit, period and qualifiers; multi-value entries (goals) list the items. Every row has the
  verification badge, the quote and a page link that opens the page image in a side panel.
- *Chat.* Saved conversations in a sidebar, a report filter, a status line that follows the
  pipeline's stage events ("Searching the reports for ..."), the answer with its status
  badge, citation chips that open the cited page, failed quotes struck through, and the
  standalone question the search used, so a wrong rewrite is visible.

The cited page is a PNG rendered by the backend at 144 dpi and cached for a day; the full PDF
opens at that page in a new tab. The SSE client is a small hand-written parser with a 60-second
idle timeout, because a proxy can keep the browser's side of a dead connection open.

**Improvements**

- Highlight the quote on the page image (5.4).
- Stream the answer text (5.3).
- Sorting, filtering and CSV export on the Extracted data page; a cross-company comparison
  view (one row per company, one column per year) for the single-value entries.
- Let the user correct detected metadata and re-run extraction from the Reports page.
- A per-answer "show the chunks" toggle for debugging; the data is already in the message.

## 8. Tests

- *Unit tests* run offline, with no database and no models: chunk windows and tables, quote
  verification including the fuzzy cases, RRF merging, number parsing, the provider's repair
  retry and cache breakpoints, how much history the rewrite gets, the report-scope rules, and
  the cleanup of extracted units.
- *Rewrite behaviour tests* run against the real small model (skipped without an API key)
  and cover the cases that go wrong in practice: a company and year named only many turns
  back, companies and years mixed across sub-queries, topics merged or split beyond what was
  asked, questions about all companies, and how many sub-queries one question may produce.
- *Parser quality* is a test with a floor per category over the 26 labelled pages, so a
  parser change cannot silently regress. A separate command prints the full scorecard with
  every failing label.
- *An integration test* runs the FastAPI app against a real Postgres and the configured
  provider: upload a generated PDF, wait for ingestion, check the datapoints, chat, check
  conversation ordering, render a page, delete. It is skipped without an API key.
- *Frontend*: TypeScript type checks, ESLint with type-aware rules, Prettier. No browser tests.
- `make test` and `make lint` run all of it.

**Improvements**

- *Continuous integration.* The checks exist but no CI configuration is in the repository.
  A GitHub Actions workflow with a pgvector service container and a small embedding model
  would run lint, unit tests, migrations and the integration test on every push.
- *A fake provider.* The integration test needs a real API key. A deterministic fake provider
  that returns canned JSON for known prompts would let the whole pipeline run in CI for free
  and make the test reproducible.
- *Browser tests* for the three main flows (upload, open a citation, follow-up question) with
  Playwright.

## 9. Evaluation

Evaluation was out of scope for the MVP and is the main addition of the final version. The
principle: measure every stage on its own, with the application's own code and settings, so a
failure can be attributed to parsing, retrieval, extraction or generation rather than to "the
RAG".

- *Corpus and gold set.* Six reports: ABN AMRO (a bank), ASML (a semiconductor equipment
  maker) and CM.com (a small software company), fiscal years 2024 and 2025. Two years per
  company force the system to tell a 2025 report from the 2024 figures it restates. 57
  questions: 30 factoids (20 answered by text, 10 by a table), 8 comparisons across years and
  companies, 5 multi-metric questions, 3 follow-ups with a conversation history, 11
  unanswerable (9 plausible figures the reports do not disclose, 2 about companies outside the
  corpus). Each answerable question has a gold answer, key facts and evidence quotes with
  pages. For extraction, one label per report and single-value entry with the acceptable
  alternatives the report also states, and 123 sustainability goals. Questions were written by
  hand per case and multiplied across companies with LLM help, phrased away from the report's
  wording; a script checks that every evidence quote is on its stated page, so nothing in the
  gold set is invented.
- *Per stage.* Ingestion: time and cost per stage, metadata accuracy, chunk statistics.
  Retrieval: hit@k, MRR, facts in context, report precision and recall, per question type,
  with ablations of each stage and parameter sweeps through command-line overrides.
  Extraction: exact, acceptable alternative, wrong, missing, spurious, scored in code by
  comparing numbers with their scale; goals scored by an LLM judge for recall and precision.
  End to end: correctness and groundedness judged by a stronger model (`claude-opus-5-5`)
  than the generator, plus deterministic checks of status, verification and latency.
- *Headline results.* 91.3% correct, 95.7% correct or partial, 0% incorrect, 0 hallucinations
  on unanswerable questions, 106 of 106 citations verified verbatim, 7.5 s median, 1.8 cents
  per question. Comparisons went from 63% to 100% with query decomposition. Full detail and
  the prioritised next steps are in [EVAL.md](EVAL.md).

**Improvements**

- *Human agreement with the judge.* One LLM judge, no human check. Reading 20 judged answers
  would put an error bar on the 91%.
- *Statistical power.* 46 answerable questions means one question moves a rate by two
  points; differences under five points are noise. More questions per type, especially
  comparisons and multi-metric, before tuning further.
- *Provider comparison.* Only Anthropic was evaluated; OpenAI and a local model are in the
  plan.

## 10. Alternatives considered

| Chosen | Alternative | Why not (for this project) |
|---|---|---|
| Plain provider SDKs and a small own interface | LangChain, LlamaIndex | Heavy abstraction for a five-step pipeline; harder to debug, to explain and to evaluate stage by stage |
| Own provider interface | LiteLLM | A large dependency that hides exactly the structured-output and caching differences this design relies on |
| Quote and verify in code | Claude Citations API | Guaranteed verbatim, but Claude-only; the design had to work for two providers |
| Postgres + pgvector | Chroma, Qdrant, Weaviate, SQLite | A second store to keep in sync with the relational data, or weaker full-text search and no cascading deletes |
| PyMuPDF | Docling, `pymupdf4llm`, LlamaParse, a vision LLM per page | Slow on CPU, third-party upload of the PDF, or high cost on 400 pages; PyMuPDF reached 1.00 on the labelled pages. A vision step for infographic pages only is the planned complement |
| Fixed word windows per page | Heading-based sections, semantic chunking | Headings in these reports are unreliable; windows with overlap are predictable, measured as good as larger ones, and the page is the citation unit anyway |
| Contextual header per chunk | Plain chunk text | Measured: +12 points hit@8, +33 on table questions |
| Local `bge-m3` embeddings | OpenAI or Voyage embeddings | Ties the index to one vendor; re-embedding 400 pages on every provider change |
| Local cross-encoder reranker | Cohere Rerank, LLM reranking | Cohere needs a key and sends report text out; an LLM reranker costs more per query. The local model is the biggest measured gain and gives a calibrated abstention score |
| Hybrid search with RRF | Vector only | Kept for exact terms, although the ablation shows it adds little as implemented; see 5.2 |
| Query decomposition into sub-queries | Agentic tool loop over a `search_reports` tool | Fixed pipeline first: easier to evaluate and to show in the UI; the loop is the next step for open comparisons |
| Rely on chunk headers for scoping | Filter each sub-query by company and year | Measured better, held back because it fails silently on unknown companies; planned together with a corpus list in the rewrite |
| Structured JSON output | Free text plus regex for citations | Status and citations as typed fields make verification and the UI simple and robust |
| RapidFuzz `partial_ratio` | `difflib.SequenceMatcher` | No dependency, but 40 times slower and misses quotes with a typo in the first word |
| Server-rendered page images | `react-pdf` / pdf.js in the browser | A 400-page PDF per citation is slow; one PNG per cited page is not |
| Server-sent events for the chat | WebSocket, polling | One-directional and plain HTTP; works through nginx with buffering off |
| Polling for ingestion progress | SSE or WebSocket | A progress bar every 2.5 s only while something ingests is enough and simpler |
| In-process background task | Celery + Redis, arq | An extra service to run and explain, while worker threads already ingest several reports in parallel; the cost is no resumption after a restart |
| React + Vite | Streamlit, Next.js | Streamlit is faster to write but weak at citation and PDF control; Next.js adds a server the app does not need |
| Chunk RAG | Whole report in the context window, GraphRAG | Slow and costly on 400 pages, no page-level citations; GraphRAG out of scope |
| LLM judge with a stronger model | Human grading only | 57 questions times many configurations; the judge is checked by reading every non-correct verdict, a human agreement study is open |

## 11. Known limitations and risks

| Limitation | Mitigation today | Fix |
|---|---|---|
| Charts and infographics are not read | Stated; only text and tables | Vision step for digit-heavy pages (4.2) |
| A verified quote can still be misread | Verification badge, page link | Number check on the answer prose (5.3) |
| Fuzzy matching tolerates a changed digit | Shown in amber | Digit check on fuzzy matches (5.4) |
| Year leakage wastes up to half of the context | Chunk headers name the year; no wrong answer in the evaluation | Per-sub-query report filter (5.2) |
| Long `unit` or `period` values fail the datapoint insert | Report marked failed, re-upload | Truncate or widen the columns (4.5) |
| Goal recall at 0.67 | Precision 0.97, so what is shown is right | Page-level extraction (4.5) |
| A restart kills an in-process ingestion | Marked failed with a clear message | Resume from the last stage, or a queue (4.1) |
| First run downloads 4.5 GB of models | Cached on a volume, stated in the README | Smaller models as an option |
| Embedding on CPU is slow | Background with progress; Metal on a Mac | A GPU, or a hosted embedding option behind the same interface |
| Cloud APIs see report content | The reports are public | Local model adapter (5.5) |
| Two providers double the test surface | Shared prompts, Pydantic validation | A fake provider in CI (8) |
