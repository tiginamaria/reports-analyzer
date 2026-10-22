# Annual Report RAG Analyzer

A local full-stack Retrieval-Augmented Generation (RAG) application for analysing company
annual reports (PDF). Upload a report, ask questions in a chat, and browse the datapoints
that were pre-extracted at ingestion (FTE, revenue, net profit, dividend per share and
sustainability goals). Every answer and every
datapoint carries a verbatim quote that is checked, in code, against the PDF text.

Design, pipeline details, alternatives and risks: [docs/DESIGN.md](docs/DESIGN.md).
Done and todo, by area: [docs/PLAN.md](docs/PLAN.md).
Evaluation of every stage, with results and improvements: [docs/EVAL.md](docs/EVAL.md).

## What it does

**Ingestion** ([pipeline.py](backend/app/ingestion/pipeline.py))

1. Parse the PDF into page texts and tables with PyMuPDF:
   [parser.py](backend/app/ingestion/parser.py)
2. Cut each page's text (tables removed) into fixed-size overlapping word windows; each table is its own chunk, indexed once:
   [chunker.py](backend/app/ingestion/chunker.py)
3. Embed with a hosted embedding model (`text-embedding-3-large` by default; local `bge-m3`
   is an option), index vectors and full text in Postgres + pgvector:
   [embedder.py](backend/app/retrieval/embedder.py), [models/](backend/app/models/)
4. Pre-extract datapoints from the config registry and verify every quote:
   [extractor.py](backend/app/extraction/extractor.py), [registry.yaml](backend/app/extraction/registry.yaml)

**Chat** ([pipeline.py](backend/app/chat/pipeline.py), SSE endpoint in [chat.py](backend/app/api/routers/chat.py))

1. Query rewriting: the small model turns the question and the conversation into a
   standalone question and one sub-query per company, fiscal year and topic, so a
   comparison searches each side separately:
   [pipeline.py](backend/app/chat/pipeline.py), [schemas.py](backend/app/chat/schemas.py)
2. Hybrid retrieval per sub-query: vector + full-text search fused with Reciprocal Rank
   Fusion: [search.py](backend/app/retrieval/search.py)
3. Optional cross-encoder reranking (`RERANKER_PROVIDER=local`) with abstention below a
   score threshold; the sub-queries' chunks are merged:
   [embedder.py](backend/app/retrieval/embedder.py)
4. Structured answer generation with a quote and chunk id per claim. The whole conversation
   is sent as real turns, with this question's chunks in the last message only; the system
   prompt and earlier turns are served from the provider's prompt cache:
   [pipeline.py](backend/app/chat/pipeline.py), [schemas.py](backend/app/chat/schemas.py)
5. Quote verification against the stored page text; answers with no verified quote become "not stated in the report":
   [verify.py](backend/app/verify.py)

**Extracted data**: FTE, revenue, net profit, dividend per share and sustainability goals
per report with quote, page link and
verification badge: [datapoints.py](backend/app/services/datapoints.py),
[DataPage.tsx](frontend/src/pages/data/DataPage.tsx). New fields go in
[registry.yaml](backend/app/extraction/registry.yaml).

**Persistence**: Postgres on a named docker volume, PDFs on a bind mount stored as
`<sha256>.pdf` with the original name kept in the database, restart safe:
[docker-compose.yml](docker-compose.yml), [storage.py](backend/app/services/storage.py).

**LLM providers**: OpenAI (default) or Anthropic behind one interface with JSON-schema
structured output and a Pydantic repair retry: [base.py](backend/app/llm/base.py),
[anthropic_provider.py](backend/app/llm/anthropic_provider.py),
[openai_provider.py](backend/app/llm/openai_provider.py).

**Background work that stays responsive**: ingestion reports progress after every page,
every embedding batch and every extraction step (units done, elapsed time, latest
activity), deleting a report cancels its ingestion before the next batch or write, and
local models, when used, are held for one batch at a time with chat queries going first:
[pipeline.py](backend/app/ingestion/pipeline.py),
[cancellation.py](backend/app/ingestion/cancellation.py),
[embedder.py](backend/app/retrieval/embedder.py).

## Run it

### Option A: everything in Docker (recommended for a first run)

Prerequisites: Docker Desktop with at least 4 GB of memory for containers and an OpenAI API
key (used for the LLM and for embeddings). An Anthropic key can take over the LLM part.

```bash
git clone <this repo> && cd reports-analyzer
cp .env.example .env
```

Edit `.env` and set the key:

```
OPENAI_API_KEY=sk-...
```

To answer with Claude instead, keep the OpenAI key for embeddings and add:

```
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
# only for user-scoped keys (sk-ant-usr-...): the workspace id from the Anthropic console
ANTHROPIC_WORKSPACE_ID=
```

Then build and start the three services (Postgres with pgvector, the FastAPI backend, the
React frontend served by nginx):

```bash
docker compose up --build -d
docker compose logs -f backend      # wait for "Application startup complete"
```

Open http://localhost:8080.

No model is downloaded: embeddings come from a hosted endpoint and reranking is off by
default, so the backend image stays small and idles at a few hundred MB. The database lives
on the `pgdata` volume and uploaded PDFs in `data/store/` (named by content hash), so
`docker compose down` and `up` keep everything. The sample reports themselves live in
`data/reports/` under the names listed in `data/reports/SOURCES.md`.

### Option B: backend and frontend on the host

Prerequisites: Docker (for Postgres), Python 3.12+ with [uv](https://docs.astral.sh/uv/),
Node 20+.

```bash
docker compose up -d db

cp .env.example .env            # set LLM_PROVIDER and the API key as above
cd backend
uv sync                         # dev tools and the optional local models (dependency groups)
uv run alembic upgrade head
uv run uvicorn app.main:app --reload --port 8000

# second terminal
cd frontend
npm install
npm run dev                     # http://localhost:5173, proxies /api to :8000
```

To run the retrieval models in-process instead (no embedding key, the cross-encoder
reranker on, GPU through Metal on a Mac), set `RETRIEVAL_MODELS=local`; the `local-models`
dependency group (torch and sentence-transformers) is installed by `uv sync` on the host. In
Docker add `--build-arg LOCAL_MODELS=1` to the backend build.

### Load the annual reports

On the **Reports** page, drop a PDF or a whole folder of PDFs onto the upload box (or click
it to choose files) and press **Upload**. A folder uploads every PDF inside it. The company
and fiscal year are detected from the first pages of each report by the LLM. From the
command line, the same endpoint takes one or many files:

```bash
curl -F "files=@data/reports/shell-annual-report-2025.pdf" localhost:8000/api/reports
```

Ingestion runs in the background: parse, chunk, embed, then extract the registry datapoints
(FTE, revenue, net profit, dividend per share, sustainability goals) with the LLM. The
Reports page shows the stage, units done of the total (pages, chunks, datapoints), the
elapsed time and the worker's latest activity, updated after every page and every embedding
batch. With hosted embeddings a report takes about 1.5 to 3 minutes, most of it the
extraction calls. Deleting a report while it is ingesting cancels the ingestion.
`data/reports/` holds eight sample reports (ABN AMRO, ASML and CM.com for 2024 and 2025,
Shell and Heineken for 2025); `data/reports/SOURCES.md` lists where they came from.

### Provider options

| `LLM_PROVIDER` | Needs | Models |
|---|---|---|
| `openai` (default) | `OPENAI_API_KEY` | `gpt-5` for answers and extraction, `gpt-5-mini` for query analysis; `OPENAI_REASONING_EFFORT=low` by default (their reasoning tokens count against the output limit, so the adapter adds headroom) |
| `anthropic` | `ANTHROPIC_API_KEY` (plus `ANTHROPIC_WORKSPACE_ID` for user-scoped keys) | `claude-sonnet-5-5` for answers and extraction, `claude-haiku-4-5` for query analysis |

The retrieval stack is one switch, independent of the LLM:

| `RETRIEVAL_MODELS` | Embeddings | Reranker | Needs |
|---|---|---|---|
| `openai` (default) | hosted `text-embedding-3-large` (batched, bounded concurrency, timeouts, retries) | none: the fused hybrid order is kept, no score floor | `OPENAI_API_KEY` (or `EMBEDDING_API_KEY`) |
| `local` | `BAAI/bge-m3` in-process | `BAAI/bge-reranker-v2-m3` cross-encoder in-process, with the abstention floor | the `local-models` dependency group (installed by `uv sync` on the host; `--build-arg LOCAL_MODELS=1` in Docker); GPU through Metal on a Mac |

Each setting the switch fills can still be overridden on its own: `EMBEDDING_PROVIDER`,
`EMBEDDING_MODEL` (`EMBEDDING_DIMENSIONS` shortens OpenAI vectors, `EMBEDDING_BASE_URL`
points at any OpenAI-compatible server), `RERANKER_PROVIDER` and `RERANKER_MODEL`; the
hosted request shape is `EMBEDDING_BATCH_SIZE`, `EMBEDDING_CONCURRENCY`,
`EMBEDDING_TIMEOUT_SECONDS` and `EMBEDDING_MAX_RETRIES`.

Each report records which embedding model produced its vectors, and only reports matching
the configured model are searched. After changing the embedding settings, rebuild the
existing reports with `cd backend && uv run python -m app.reembed` (`--dry-run` lists them);
the header and the Reports page flag reports that need it. The remaining knobs (chunk sizes,
retrieval depth, verification threshold) are constants in the backend modules; settings in
`backend/app/config.py` can be overridden in the repo-root `.env` (a `backend/.env`, if
present, takes precedence).

### Check that it works

```bash
curl localhost:8000/api/health            # {"status":"ok"}
curl localhost:8000/api/status            # provider and models in use
```

## Using it

- **Reports**: upload one PDF or a folder, watch ingestion progress (stage, units done,
  elapsed time, latest activity), delete; deleting a report stops its ingestion.
- **Extracted data**: cross-company tables of FTE, revenue, net profit, dividend per share
  and sustainability goals. Click a page
  number to open that page in a side panel. Badges: `verified` (exact substring of the page
  text after normalisation), `fuzzy match` (small whitespace or typography differences),
  `quote not found` (the quote is not in the report text).
- **Chat**: ask in natural language. Optional report filter in the sidebar; otherwise the
  company and year in the question narrow the search. Citation chips open the cited page;
  failed quotes are shown struck through. If every quote fails verification the answer is
  downgraded to "not stated in the report". Before anything runs, the chat answers
  greetings directly and tells apart "upload a report", "reports are still processing"
  and "no relevant evidence found" without calling the model; local models that still have
  to load are shown as their own stage.

Example questions:

- How many FTEs did CM.com have in 2025?
- What are Heineken's emission reduction targets?
- How much did Shell spend on climate change adaptation in 2025? (not disclosed as a figure;
  the system should abstain or point to the related taxonomy disclosure)

## API

`http://localhost:8000/docs` for the OpenAPI UI. Main endpoints:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/reports` | Upload one or many PDFs (`files`); returns started reports and per-file rejections |
| `GET` | `/api/reports`, `/api/reports/{id}` | Index status and progress, plus the outcome of every datapoint's extraction |
| `POST` | `/api/reports/{id}/extract` | Extract the failed datapoints (or the `keys` given) again without re-ingesting the PDF |
| `GET` | `/api/reports/{id}/pdf` | The PDF (use `#page=N`) |
| `GET` | `/api/reports/{id}/pages/{n}` | Page `n` rendered as PNG; what the citation panel shows |
| `GET` | `/api/datapoints?report_id=&key=` | Pre-extracted datapoints |
| `POST` | `/api/chat` | Server-sent events: `conversation`, `stage`…, `answer` |
| `GET` | `/api/conversations` | Saved conversations |

## Tests

```bash
make test     # unit tests (offline) + integration tests (Postgres, hosted embeddings, needs the API keys)
make lint     # ruff, mypy, tsc
```

**Parser quality** has its own dataset: `backend/tests/fixtures/pdf/` holds 26 pages cut from
the real reports (financial tables, five-year tables, infographics, two-column text,
pages with navigation bars), labelled in `backend/tests/fixtures/parsing.yaml` with text that
must be recovered, table cells that must land in a table, pages without tables, and
reading-order pairs. `test_parser_quality.py` holds
each category at a floor so parser changes cannot silently regress, and

```bash
cd backend && uv run python -m tests.parsing_quality
```

prints the full scorecard with every failing label. Current scores: text, table cells,
false tables and reading order all at 1.00.

## Evaluation

[docs/EVAL.md](docs/EVAL.md) evaluates every stage on six reports (ABN AMRO, ASML, CM.com,
fiscal years 2024 and 2025), 57 gold questions and labelled datapoints, with
`claude-opus-5-5` as judge, for the default configuration and for the local retrieval
profile. Headline results:

| Stage | Default: gpt-5, hosted embeddings, no reranker | Local retrieval profile with Claude |
|---|---|---|
| Answers | 65% of answerable questions correct, 85% correct or partial, 3 incorrect; all 11 unanswerable questions handled, 0 hallucinations; 84/93 citations verified verbatim, 9 rejected; 9.0 s median, $0.014 per question | 91% correct, 96% correct or partial, 0 incorrect; 106/106 citations verified; 7.5 s, $0.018 |
| Retrieval | A gold page among the retrieved chunks for 72% of questions, in 0.4 s (91% with the local reranker added to the same embeddings) | 96%, in 4.8 s |
| Extraction | 17 of 24 single values exact, 4 defensible alternatives, 1 missing, none wrong; sustainability goals recall 0.66, precision 0.85 | 19 exact, 2 alternatives, 1 missing, none wrong; recall 0.56, precision 0.97 |
| Ingestion | Company and year detected 6/6; 2,073 pages in 7.8 minutes and $1.04 | 6/6; 19 minutes and $1.01 on a laptop GPU |

The harness and how to rerun it or run a parameter experiment:
[backend/eval/README.md](backend/eval/README.md).

## Project layout

```
backend/app
  api/          async FastAPI routers (reports, datapoints, chat, system), deps, error mapping
  services/     use cases: upload/storage, extraction retry, deletion, chat persistence, startup
  repositories/ every SQL query, on the async session
  schemas/      Pydantic API types      models/   SQLAlchemy tables
  ingestion/    parser (PyMuPDF), chunker, pipeline, cancellation
  retrieval/    hosted/local embedders + optional reranker, hybrid search with RRF
  extraction/   registry.yaml, extractor, metadata detection
  chat/         analyse -> retrieve -> rerank -> generate -> verify
  llm/          provider interface, Anthropic, OpenAI
  verify.py     verbatim quote check (exact, fuzzy, failed)
frontend/src
  api/          fetch wrapper, server-sent events parser, typed endpoints
  pages/        reports (upload + table), data (datapoint sections), chat (sidebar, messages, useChat)
  components/   header, badges, table, PDF page panel
  lib/          report helpers, file picking, formatting
```
