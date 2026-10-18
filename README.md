# Annual Report RAG Analyzer

A local full-stack Retrieval-Augmented Generation (RAG) application for analysing company
annual reports (PDF). Upload a report, ask questions in a chat, and browse the datapoints
that were pre-extracted at ingestion (FTE and sustainability goals). Every answer and every
datapoint carries a verbatim quote that is checked, in code, against the PDF text.

Design, pipeline details, alternatives and risks: [docs/DESIGN.md](docs/DESIGN.md).
Done and todo, by area: [docs/PLAN.md](docs/PLAN.md).
Evaluation of every stage, with the chosen parameters: [docs/MVP_EVAL.md](docs/MVP_EVAL.md).

## What it does

**Ingestion** ([pipeline.py](backend/app/ingestion/pipeline.py))

1. Parse the PDF into page texts and tables with PyMuPDF:
   [parser.py](backend/app/ingestion/parser.py)
2. Cut each page's text (tables removed) into fixed-size overlapping word windows; each table is its own chunk, indexed once:
   [chunker.py](backend/app/ingestion/chunker.py)
3. Embed locally with `bge-m3`, index vectors and full text in Postgres + pgvector:
   [embedder.py](backend/app/retrieval/embedder.py), [models.py](backend/app/models.py)
4. Pre-extract datapoints from the config registry and verify every quote:
   [extractor.py](backend/app/extraction/extractor.py), [registry.yaml](backend/app/extraction/registry.yaml)

**Chat** ([pipeline.py](backend/app/chat/pipeline.py), SSE endpoint in [chat.py](backend/app/api/chat.py))

1. Query analysis: company, fiscal year, follow-up rewriting:
   [pipeline.py](backend/app/chat/pipeline.py)
2. Hybrid retrieval: vector + full-text search fused with Reciprocal Rank Fusion:
   [search.py](backend/app/retrieval/search.py)
3. Cross-encoder reranking, abstain below a score threshold:
   [embedder.py](backend/app/retrieval/embedder.py)
4. Structured answer generation with a quote and chunk id per claim. The whole conversation
   is sent as real turns, with this question's chunks in the last message only; the system
   prompt and earlier turns are served from the provider's prompt cache:
   [pipeline.py](backend/app/chat/pipeline.py), [schemas.py](backend/app/chat/schemas.py)
5. Quote verification against the stored page text; answers with no verified quote become "not stated in the report":
   [verify.py](backend/app/verify.py)

**Extracted data**: FTE and sustainability goals per report with quote, page link and
verification badge: [datapoints.py](backend/app/api/datapoints.py),
[DataPage.tsx](frontend/src/pages/data/DataPage.tsx). New fields go in
[registry.yaml](backend/app/extraction/registry.yaml).

**Persistence**: Postgres on a named docker volume, PDFs on a bind mount stored as
`<sha256>.pdf` with the original name kept in the database, restart safe:
[docker-compose.yml](docker-compose.yml), [reports.py](backend/app/api/reports.py).

**LLM providers**: Anthropic (default) or OpenAI behind one interface with JSON-schema
structured output and a Pydantic repair retry: [base.py](backend/app/llm/base.py),
[anthropic_provider.py](backend/app/llm/anthropic_provider.py),
[openai_provider.py](backend/app/llm/openai_provider.py).

## Run it

### Option A: everything in Docker (recommended for a first run)

Prerequisites: Docker Desktop with at least 8 GB of memory for containers, and an Anthropic
or OpenAI API key.

```bash
git clone <this repo> && cd reports-analyzer
cp .env.example .env
```

Edit `.env` and set the provider and key:

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

The first backend start downloads about 4.5 GB of local models (`bge-m3` embeddings and
`bge-reranker-v2-m3`) into the `models` volume, which takes a few minutes. The database lives
on the `pgdata` volume and uploaded PDFs in `data/store/` (named by content hash), so
`docker compose down` and `up` keep everything. The sample reports themselves live in
`data/reports/` under the names listed in `data/reports/SOURCES.md`.

### Option B: backend and frontend on the host (faster ingestion on Apple silicon)

Prerequisites: Docker (for Postgres), Python 3.12+ with [uv](https://docs.astral.sh/uv/),
Node 20+.

```bash
docker compose up -d db

cp .env.example .env            # set LLM_PROVIDER and the API key as above
cd backend
uv sync --group dev
uv run alembic upgrade head
uv run uvicorn app.main:app --reload --port 8000

# second terminal
cd frontend
npm install
npm run dev                     # http://localhost:5173, proxies /api to :8000
```

On a Mac the embedder and reranker run on the GPU through Metal, which makes ingestion
several times faster than in Docker.

### Load the annual reports

On the **Reports** page, drop a PDF or a whole folder of PDFs onto the upload box (or click
it to choose files) and press **Upload**. A folder uploads every PDF inside it. The company
and fiscal year are detected from the first pages of each report by the LLM. From the
command line, the same endpoint takes one or many files:

```bash
curl -F "files=@data/reports/shell-annual-report-2025.pdf" localhost:8000/api/reports
```

Ingestion runs in the background with a progress bar: parse, chunk, embed, then extract FTE
and sustainability goals with the LLM. A 400-page report takes about 2 minutes on an M-series
Mac and 10 to 15 minutes on CPU in Docker. The sample set in `data/reports/SOURCES.md` lists
where the five example reports came from.

### Provider options

| `LLM_PROVIDER` | Needs | Models |
|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` (plus `ANTHROPIC_WORKSPACE_ID` for user-scoped keys) | `claude-sonnet-5-5` for answers and extraction, `claude-haiku-4-5` for query analysis |
| `openai` | `OPENAI_API_KEY` | `gpt-5` / `gpt-5-mini` |

Model names, chunk sizes, retrieval depth and the verification threshold are all settings in
`backend/app/config.py` and can be overridden in the repo-root `.env` (a `backend/.env`, if
present, takes precedence).

### Check that it works

```bash
curl localhost:8000/api/health            # {"status":"ok"}
curl localhost:8000/api/status            # provider and models in use
```

## Using it

- **Reports**: upload one PDF or a folder, watch ingestion progress, delete.
- **Extracted data**: cross-company table of FTE and sustainability goals. Click a page
  number to open that page in a side panel. Badges: `verified` (exact substring of the page
  text after normalisation), `fuzzy match` (small whitespace or typography differences),
  `quote not found` (the quote is not in the report text).
- **Chat**: ask in natural language. Optional report filter in the sidebar; otherwise the
  company and year in the question narrow the search. Citation chips open the cited page;
  failed quotes are shown struck through. If every quote fails verification the answer is
  downgraded to "not stated in the report".

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
| `GET` | `/api/reports`, `/api/reports/{id}` | Status and progress |
| `GET` | `/api/reports/{id}/pdf` | The PDF (use `#page=N`) |
| `GET` | `/api/reports/{id}/pages/{n}` | Page `n` rendered as PNG; what the citation panel shows |
| `GET` | `/api/datapoints?report_id=&key=` | Pre-extracted datapoints |
| `POST` | `/api/chat` | Server-sent events: `conversation`, `stage`…, `answer` |
| `GET` | `/api/conversations` | Saved conversations |

## Tests and CI

```bash
make test     # unit tests (offline) + integration test (Postgres, real embedder, needs an LLM API key)
make lint     # ruff, mypy, tsc
```

**Parser quality** has its own dataset: `backend/tests/fixtures/pdf/` holds 26 pages cut from
the five real reports (financial tables, five-year tables, infographics, two-column text,
pages with navigation bars), labelled in `backend/tests/fixtures/parsing.yaml` with text that
must be recovered, table cells that must land in a table, pages without tables, and
reading-order pairs. `test_parser_quality.py` holds
each category at a floor so parser changes cannot silently regress, and

```bash
cd backend && uv run python -m tests.parsing_quality
```

prints the full scorecard with every failing label. Current scores: text, table cells,
false tables and reading order all at 1.00.

GitHub Actions runs the same with a pgvector service container.

## Project layout

```
backend/app
  api/          FastAPI routers: reports, datapoints, chat + conversations
  ingestion/    parser (PyMuPDF), chunker, pipeline
  retrieval/    embedder + reranker, hybrid search with RRF
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

## Limitations

- Charts and infographics are not read; only text and tables that PyMuPDF recovers.
  Chunking is deliberately simple: fixed windows with overlap and no section structure.
  A golden question set with retrieval and answer metrics is the missing piece that would
  tell whether heading-based sections or a layout-aware parser earn their complexity.
- Verification proves a quote exists; it does not prove the model read it correctly.
- Comparison questions across many companies rely on a single retrieval pass with the
  company filter removed; an agentic multi-search loop is the natural next step.
