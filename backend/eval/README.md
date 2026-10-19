# Evaluation

Offline evaluation of the pipeline as it ships, stage by stage: ingestion, retrieval,
extraction and the end-to-end chat. Each script runs the app's own code with the app's
settings; an experiment overrides parameters through command-line arguments and lands in its
own result file. Results, analysis and improvements: [docs/EVAL.md](../../docs/EVAL.md).

Everything runs against a separate database (`reports_eval`, or `EVAL_DATABASE_URL`), so the
reports ingested through the UI are never touched. Importing any `eval` module redirects
`DATABASE_URL` before the app creates its engine.

## Corpus and gold set

- [corpus.yaml](corpus.yaml): ABN AMRO, ASML and CM.com, fiscal years 2024 and 2025. The 2025
  reports are the app's samples; the 2024 reports (also in `data/reports/`, sources in
  `SOURCES.md`) force retrieval and scoping to tell two years of one company apart.
- [gold/qa.yaml](gold/qa.yaml): 57 questions: 30 factoid, 8 comparison (across years and
  across companies), 5 multi-metric (three figures of one report in one sentence, each from
  a different section), 3 follow-ups with a conversation history, 11 unanswerable (9
  plausible but not disclosed, 2 about companies outside the corpus). Each answerable
  question has a short gold answer, key facts as the report writes them, and evidence quotes
  with pages.
- [gold/extraction.yaml](gold/extraction.yaml): one label per report for each single-value
  registry entry (year-end FTE, revenue, net profit, dividend per share), each with the
  acceptable alternatives the report also states (average FTE, a rounded figure, a final
  dividend alone) or a null value where the report states none (CM.com pays no dividend);
  and 123 sustainability goals (41 marked `core`).

Labels were drafted per company with LLM help from the page text and then checked in code:
`gold_check` fails if any evidence quote is not on its page in `pages.text`, or a key fact is
in no evidence quote. Questions were phrased away from the evidence wording, so retrieval is
not rewarded for keyword overlap.

## Running

```bash
docker compose up -d db
docker exec reports-analyzer-db-1 psql -U reports -c "create database reports_eval"
cd backend
DATABASE_URL=postgresql+psycopg://reports:reports@localhost:5432/reports_eval uv run alembic upgrade head

uv run python -m eval.ingest_eval          # ingest the corpus with stage timings, then score
uv run python -m eval.gold_check           # gold quotes are on their pages
uv run python -m eval.retrieval_eval       # what the chat retrieves, scored against the gold pages
uv run python -m eval.extraction_eval      # run the registry on each report as ingestion does, score and cost it
uv run python -m eval.e2e_eval             # full chat pipeline + LLM judge
```

Every script runs the pipeline as configured in `.env`: the LLM provider (`LLM_PROVIDER`)
and the retrieval profile (`RETRIEVAL_MODELS`: hosted embeddings without a reranker, or
bge-m3 with the cross-encoder). A run without overrides therefore evaluates the shipped
defaults; the reranker, its score floor and the abstention gate only take part in the
`local` profile. The judge is always `claude-opus-5-5`, so `ANTHROPIC_API_KEY` is needed
even when the app answers with OpenAI.

Results land in `eval/results/<stage>.json`. The two profiles were run on their own
databases, since vectors of different models do not compare, and kept apart by `--tag`:

| Files | Profile | Run |
|---|---|---|
| `ingestion.json`, `retrieval.json`, `extraction.json`, `e2e.json` | `LLM_PROVIDER=anthropic`, `RETRIEVAL_MODELS=local` (claude-sonnet-5-5, bge-m3, bge reranker) | 2026-10-18 |
| `*_openai.json` | `LLM_PROVIDER=openai`, `RETRIEVAL_MODELS=openai` (gpt-5, text-embedding-3-large, no reranker), the defaults | 2026-10-19, `EVAL_DATABASE_URL=...reports_eval_openai`, `--tag openai` on every script |
| `retrieval_openai_vector.json`, `retrieval_openai_rerank.json` | Ablations on that database: `--search vector`, and `RERANKER_PROVIDER=local` on top of the hosted embeddings | 2026-10-19 |

The only cache (`eval/cache/`) holds the query rewrites, one file per rewrite model, so
retrieval runs with different parameters search the same queries.

### Experiments

Every override lands in its own file (`retrieval_candidates40_top_k8.json`, or `--tag name`),
so the result of the app's settings is never overwritten. `--limit N` runs the first N
questions as a smoke test.

| Script | Arguments | Meaning |
|---|---|---|
| `retrieval_eval`, `e2e_eval` | `--candidates`, `--top-k`, `--min-score`, `--rrf-k` | The constants of `retrieval/search.py` |
| | `--search vector`, `--search fulltext` | One search method alone instead of hybrid |
| | `--no-rerank` | Keep the fused order; no score floor (what the `openai` profile does anyway) |
| | `--max-subqueries N` | `0` searches the standalone question only (no decomposition) |
| `e2e_eval` | `--model`, `--workers` | Answer model (of the configured provider); questions in parallel (1 for honest latency) |
| `extraction_eval` | `--top-k`, `--candidates`, `--min-score`, `--per-query` | The tunable fields of a registry entry, applied to every entry |
| | `--keys fte sustainability_goal` | A subset of the registry |
| | `--stored` | Score what ingestion wrote; no model calls |

Chunking (`CHUNK_MAX_WORDS`, `CHUNK_OVERLAP_WORDS`, the contextual header) changes the index,
so it is evaluated by re-ingesting: point `EVAL_DATABASE_URL` at another database, change
the constants in `ingestion/chunker.py`, run `ingest_eval` and then `retrieval_eval`.

## What is measured

| Stage | Script | Metrics |
|---|---|---|
| Ingestion | `ingest_eval` | Seconds per stage and their share, LLM tokens and cost, hosted embedding tokens and cost, metadata detection accuracy, chunks per report (table chunks, median and longest chunk, duplicates), datapoints written; a report whose stage raises is marked failed as the app does, with the stage and error recorded, and the run goes on |
| Retrieval | `retrieval_eval` | hit@k (a gold page among the retrieved chunks), MRR, facts-in-context (every key fact in the retrieved text of the right reports), report precision (share of chunks from the right company and year), report recall (share of the gold reports with a chunk, for comparisons); in all, per question type and text vs table; sub-queries and chunks per question, retrieval seconds p50/p90; for the abstention floor, the lowest best score on an answerable question and the share of unanswerable questions that retrieved nothing |
| Extraction | `extraction_eval` | Per datapoint: exact / acceptable alternative / wrong / missing, or correctly nothing where the report states none; whether a gold page was among the candidate chunks; quote verified; chunks passed to the model, tokens, seconds and cost. Goals: recall (all and core), precision (real goal with a verified quote), candidate-page recall, the same cost figures. Totals per report and for the corpus |
| Answers | `e2e_eval` | Judge correctness (correct, partial, incorrect, abstained) and groundedness in the cited quotes; false abstention; unanswerable questions handled and hallucinated; cited and retrieved gold page; citations verified / fuzzy / failed; latency in all and per stage; tokens and cost |

The judge is `claude-opus-5-5`, a different and stronger model than either generator, with
a rubric that is strict on figures and periods and lenient on wording. Costs are estimated
from list prices in `common.py` (cached input at a tenth; OpenAI's reasoning tokens count as
output).

## Limits

- 46 answerable questions: one question moves a rate by about 2 points (12.5 on the 8
  comparisons). Treat differences under about 5 points as noise and compare runs per
  question, which the result files allow.
- Gold pages list where an analyst found the answer, not every page that restates it;
  `facts` is the lenient check for that.
- Short key facts (e.g. `"48"`) can match unrelated text in the facts check.
- One LLM judge, no human agreement study.
