# Evaluation

Offline evaluation of the pipeline, stage by stage: ingestion, retrieval, extraction and
the end-to-end chat. Results and recommendations: [docs/MVP_EVAL.md](../../docs/MVP_EVAL.md).

Everything runs against a separate database (`reports_eval`, or `EVAL_DATABASE_URL`), so the
reports ingested through the UI are never touched. Importing any `eval` module redirects
`DATABASE_URL` before the app creates its engine.

## Corpus and gold set

- [corpus.yaml](corpus.yaml): ABN AMRO, ASML and CM.com, fiscal years 2024 and 2025. The 2025
  reports are the app's samples; the 2024 reports (also in `data/reports/`, sources in `SOURCES.md`)
  force retrieval and scoping to tell two years of one company apart.
- [gold/qa.yaml](gold/qa.yaml): 52 questions: 30 factoid, 8 comparison (across years and
  across companies), 3 follow-ups with a conversation history, 11 unanswerable (9 plausible
  but not disclosed, 2 about companies outside the corpus). Each answerable question has a
  short gold answer, key facts as the report writes them, and evidence quotes with pages.
- [gold/extraction.yaml](gold/extraction.yaml): year-end FTE per report with the acceptable
  alternatives (average FTE, headcount, payroll only), and 123 sustainability goals
  (41 marked `core`).

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

uv run python -m eval.ingest_eval                  # ingest the corpus with stage timings (~25 min on M-series)
uv run python -m eval.gold_check                   # gold quotes are on their pages
uv run python -m eval.retrieval_eval               # stage ablation + parameter sweeps (offline after first run)
uv run python -m eval.extraction_eval stored goals_k48 goals_queries goals_k48_queries
uv run python -m eval.e2e_eval baseline tuned      # full chat pipeline + LLM judge
uv run python -m eval.variants words200 words600 noheader   # alternative indexes
uv run python -m eval.retrieval_eval --variant words200     # ... and their retrieval scores
uv run python -m eval.embed_bench                  # embedding throughput
```

Results land in `eval/results/*.json`. Caches (`eval/cache/`: parsed documents, rankings,
reranker scores, query analyses) make reruns of the retrieval sweeps free.

## What is measured

| Stage | Script | Metrics |
|---|---|---|
| Ingestion | `ingest_eval` | Seconds per stage, LLM tokens and cost, metadata detection accuracy, empty pages, text kept vs lost around tables, repeated header lines, chunk length distribution, duplicates |
| Embedding | `embed_bench` | Chunks per second for the batching scheme, fp16 and length cap; cosine to current vectors |
| Retrieval | `retrieval_eval` | hit@k (a gold page in the top k), MRR, facts-in-context (every key fact in the retrieved text of the right reports), report precision (share of chunks from the right company and year); per question type and text vs table; three scopes: the pipeline's, oracle (gold reports) and global; abstention threshold curve |
| Scope | `retrieval_eval` | Does the pipeline's company/year scope contain the gold reports |
| Extraction | `extraction_eval` | FTE exact / acceptable alternative / wrong, cited page, verification; goals recall (all and core), precision (real goal with a verified quote), candidate-page recall (did retrieval surface the gold page) |
| Answers | `e2e_eval` | Judge correctness (correct, partial, incorrect, abstained), groundedness in the cited quotes, key facts in the answer, cited gold page, false abstention, abstention and hallucination on unanswerable questions, citation verified/fuzzy/failed, latency per stage, tokens and cost |

The judge is `claude-opus-5-5`, a different and stronger model than the generator, with a
rubric that is strict on figures and periods and lenient on wording.

## Limits

- 41 answerable questions: one question moves a rate by about 2.4 points (12.5 on the 8
  comparisons). Treat differences under about 5 points as noise; compare configs per
  question, as the scripts do on the same set.
- Gold pages list where an analyst found the answer, not every page that restates it;
  `facts` is the lenient check for that.
- Short key facts (e.g. `"48"`) can match unrelated text in the facts check.
- One LLM judge, no human agreement study.
