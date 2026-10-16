# Evaluation

How well the pipeline works, stage by stage, what limits it, and which parameters to use.
The harness, the gold set and how to rerun everything: [backend/eval/README.md](../backend/eval/README.md).
Raw results: `backend/eval/results/*.json`. Run on 2026-10-16, Apple M-series (MPS),
`claude-sonnet-5-5` for answers and extraction, `claude-haiku-4-5` for query analysis,
`claude-opus-5-5` as judge.

## Summary

"Current" below means the MVP as evaluated. Recommendations 1 and 2 (60 candidates, top 12,
abstain below 0.2) have since been applied in `retrieval/search.py`; `e2e_eval current`
evaluates the code as it is.

- **Answers are trustworthy; coverage is the gap.** On 52 questions the current pipeline is
  correct on 88% of answerable questions, partially correct on another 10%, never
  wrong, and never invents a figure for a question the reports do not answer. All 121
  citations verified. Failures are missing information, not false information.
- **Two parameter changes take it to 93% correct and 100% correct-or-partial** (every
  factoid right) for +1.3 s median latency and +0.8 cents per question: rerank 60
  candidates instead of 40 and keep 12 chunks instead of 8. A rerank threshold of 0.2
  instead of 0.003 answers questions about companies outside the corpus without calling the
  model. A repeat run reproduced every verdict (0 of 52 flips).
- **The cross-encoder carries retrieval**: +12 points hit@8 over vector search. The
  contextual chunk header is worth another 12 points (and 33 on table questions).
  **Full-text search contributes nothing**: `websearch_to_tsquery` ANDs every term, so 54%
  of real questions match zero chunks.
- **Comparisons are the weak spot** (63% correct): one query cannot retrieve both sides of
  "ASML vs ABN AMRO" or "2024 vs 2025". This needs query decomposition, not tuning.
- **Extraction**: FTE is right or defensibly right for all six reports (4/6 the exact
  year-end figure). Sustainability goals have precision 1.0 but recall 0.50 (0.76 on the
  headline goals); half of the gold goals' pages never reach the model. Giving it more chunks
  exposes a latent bug: goal lists overflow the 6,000-token output cap and the extraction
  fails silently.
- **Ingestion**: 2,073 pages in 21.5 minutes, $0.82 of LLM calls, metadata detection 6/6.
  Embedding is 48% of the time and runs 47% faster with one `encode` call (identical
  vectors).

## Setup

**Corpus.** Six reports: ABN AMRO, ASML and CM.com, fiscal years 2025 (from `data/reports`)
and 2024 (added to `data/reports`; sources in `data/reports/SOURCES.md`). A bank, an
industrial and a small tech company, 159 to 508 pages. Two years per company make year
scoping matter, and the 2025 reports restate 2024 figures, which is a realistic trap.

**Gold set.** 52 questions: 30 factoids (10 answered by a table), 8 comparisons (6 across
years, 2 across companies), 3 follow-ups with a conversation history, 11 unanswerable (9
plausible but not disclosed, such as ABN AMRO's ATM count or TSMC's share of ASML revenue;
2 about companies outside the corpus). Plus year-end FTE with acceptable alternatives
per report, and 123 sustainability goals (41 headline). Labels were drafted per company with
LLM help from the page text, phrased away from the source wording, and verified in code:
every one of 213 evidence quotes is on its stated page in `pages.text`.

**Metrics.**

| Stage | Metric |
|---|---|
| Retrieval | hit@k (a gold page in the top k), MRR, facts-in-context (all key facts in the retrieved text of the right reports), report precision |
| Answers | Judge verdict (correct, partial, incorrect, abstained); grounded (every figure in a cited quote); citation verification; false abstention; hallucination on unanswerable; latency; cost |
| Extraction | FTE exact / alternative / wrong; goal recall (all, headline), precision (real goal with verified quote), candidate-page recall |

**Noise.** 41 answerable questions: one question is 2.4 points, one comparison 12.5 points.
Configurations are compared on the same questions, and conclusions below rest on effects of
5 points or more, or on per-question inspection.

## Ingestion

| Report | Pages | Chunks | Tables | Total | Parse | Embed | Extract | LLM cost |
|---|---|---|---|---|---|---|---|---|
| ABN AMRO 2025 | 434 | 928 | 197 | 292 s | 54 s | 155 s | 82 s | $0.14 |
| ABN AMRO 2024 | 508 | 1,134 | 215 | 328 s | 67 s | 198 s | 62 s | $0.15 |
| ASML 2025 | 350 | 838 | 158 | 200 s | 48 s | 85 s | 66 s | $0.17 |
| ASML 2024 | 410 | 960 | 232 | 275 s | 102 s | 106 s | 65 s | $0.15 |
| CM.com 2025 | 159 | 343 | 81 | 93 s | 17 s | 34 s | 40 s | $0.11 |
| CM.com 2024 | 212 | 422 | 69 | 101 s | 18 s | 42 s | 39 s | $0.11 |
| **Total** | 2,073 | 4,625 | 952 | 21.5 min | 24% | **48%** | 28% | $0.82 |

- **Metadata detection**: company and fiscal year right for all 6.
- **Parsing**: body text plus tables keeps 99.5 to 99.8% of each page's words, so nothing is
  lost or doubled around tables. CM.com has 11 to 26 near-empty pages (infographics and
  dividers) whose figures are not read.
- **Repeated page furniture**: in CM.com, 46 to 50% of chunks start with the navigation bar
  ("HOME CONTENTS MANAGEMENT BOARD REPORT ..."). It dilutes embeddings, and it is one cause
  of a miss below (the key-figures page).
- **Chunks**: median 210 to 320 words; 5 to 16% under 50 words (page tails, small tables);
  the largest table chunks are 710 words, over the reranker's 1,024-token window.
- **How embedding works today** (`pipeline._embed`). A report's chunks are cut into slices
  of 32 (`embedding_batch_size`). Each slice is one `Embedder.embed_documents` call, which
  runs one `model.encode(texts, batch_size=32, normalize_embeddings=True)` under the shared
  model lock, and the progress bar moves after every slice. A 928-chunk report makes 29
  calls. The text embedded is the chunk behind its contextual header.
- **No truncation at embedding time.** bge-m3 loads with `max_seq_length = 8192`; the
  longest chunk in the sample below is 1,295 tokens (median 480), so every chunk is embedded
  whole, including the largest 710-word tables. The only cap is the reranker's
  `RERANK_MAX_TOKENS = 1024`, which cuts the few longest table chunks when they are scored at
  chat time, not when they are indexed.
- **Embedding throughput** (512 random chunks with their headers, `embed_bench`, one run per
  scheme after a warm-up, MPS):

  | Scheme | Chunks/s | Min cosine to current |
  |---|---|---|
  | Current: one `encode` per 32 chunks | 8.8 | 1.0 (reference) |
  | One `encode` for all chunks (sorted by length) | 12.9 (+47%) | 1.0 |
  | One call, fp16 | 14.2 (+61%) | 0.9998 |
  | One call, max 1,024 tokens | 12.8 | 0.946 |

  Chunks/s is 512 divided by the wall time of the run. Min cosine compares each chunk's
  vector with its vector under the current scheme and reports the lowest of the 512, so a
  single changed chunk shows.

  - The cost is padding, not length. Every chunk in a GPU batch is padded to the longest
    one, and `encode` sorts by length only within a call. A 32-chunk call cannot regroup
    with other calls, so a 100-word page tail is padded to a 700-word table.
  - One call over all chunks lets `encode` sort all of them by length first, so batches are
    tight. Same model and inputs, identical vectors: free speed.
  - fp16 adds cheaper arithmetic; 0.9998 is a numerical difference too small to change a
    ranking, though retrieval on an fp16 index was not measured.
  - Capping at 1,024 tokens is not faster, because only 1 of the 512 chunks is longer and
    there is almost nothing to cut. The 0.946 is that one chunk, whose cut tail moved its
    vector. Accuracy lost for no speed: ruled out.
  - Embedding is 48% of ingestion, so one call saves about 48% × (1 − 1/1.47) ≈ 15% of
    total ingestion with no change to the index. Progress can still be reported through a
    callback or by embedding in slices of about 512 chunks.
  - During ingestion ABN AMRO embedded at about 6 chunks/s because other work shared the
    CPU; 8.8 is the clean figure. Each scheme ran once, so treat ±1 chunk/s as noise.

## Retrieval

The 41 answerable questions; the 11 unanswerable ones are used only for the abstention
threshold below.

**How a configuration is run** (`eval/retrieval_eval.py`, replaying the app's stages):

1. **Query**: the standalone question from query analysis (Haiku), as the app searches it.
2. **Scope**: only the reports matching the analysis' company and year (on average 1.15
   reports). "What was ASML's net income in 2025?" searches ASML 2025 only. The oracle
   scope (gold reports) and the global scope (all six) are reported in the results file.
3. **Candidates**: each search method returns its best 40 chunks in that scope.
4. **Fusion**: with two methods the lists are merged by Reciprocal Rank Fusion, a chunk
   scoring 1/(60 + rank) for each list it is in, and cut back to 40.
5. **Rerank** (where stated): the cross-encoder rescores the 40 and they are re-sorted.
6. **Context**: the top 8 are what the answer model would see.

**Metrics**, all against the gold evidence pages (where the labeller found the answer, with a
verified quote):

- **hit@k**: share of questions with at least one of the top k chunks on a gold page of
  the right report. 0.90 is 37 of 41.
- **MRR** (mean reciprocal rank): 1 / rank of the first chunk on a gold page (1.0 at rank
  1, 0.5 at rank 2, 0.33 at rank 3, 0 if none in the top k), averaged. 0.64 means the first
  useful chunk is usually at rank 1 or 2, which matters because the model reads the top
  most closely.
- **facts**: share of questions where every key fact (e.g. "15.4" for the CET1 ratio)
  appears in the retrieved text from the right reports. Looser than hit@k (a figure
  restated on a page the labeller did not list counts), stricter for comparisons (both
  figures must be retrieved).
- **report precision** (in the results file): share of retrieved chunks from the right
  company and year.

One question moves a rate by about 2.4 points, so 0.88 vs 0.90 is noise.

**Stage ablation** (top 8 unless stated):

| Configuration | hit@8 | MRR | facts |
|---|---|---|---|
| Full-text only (current, AND) | 0.22 | 0.15 | 0.15 |
| Full-text only, OR of terms | 0.37 | 0.20 | 0.39 |
| Vector only | 0.78 | 0.45 | 0.71 |
| Hybrid RRF, no rerank | 0.73 | 0.43 | 0.68 |
| Vector + rerank | 0.90 | 0.63 | 0.85 |
| **Hybrid + rerank (current)** | 0.90 | 0.64 | 0.85 |
| Hybrid OR + rerank | 0.88 | 0.66 | 0.85 |

- **Full-text, AND (0.22, the app's keyword search)**. `websearch_to_tsquery` requires every
  term of the question in the chunk. "How strong was ABN AMRO's core capital position (CET1)
  at year-end 2025?" becomes roughly *strong & abn & amro & core & capit & posit & cet1 &
  year & end & 2025*; the answer chunk says "CET1 ratio of 15.4% at year-end" without
  "strong" or "position", so it does not match. Of all 52 rewritten questions, 28 match no
  chunk at all and 41 fewer than 5: users ask in their words, not the report's.
- **Full-text, OR (0.37)**. Any term may match, ranked by `ts_rank_cd` (many query terms
  close together). Everything matches now, but within an ABN AMRO 2025 scope "ABN", "AMRO"
  and "2025" are in almost every chunk, so the ranking favours chunks dense in common words.
- **Vector only (0.78, MRR 0.45)**. bge-m3 handles paraphrase ("capital position" to "CET1
  ratio") and finds a gold page for 32 of 41, but often at rank 3 to 8 rather than 1.
- **Hybrid without rerank (0.73)**. When full-text returns nothing (most questions) fusion
  is the vector list; when it returns a few loose matches, fusion lifts them into the top 8
  above good vector hits. Adding keyword search costs about 2 questions.
- **Vector + rerank (0.90, MRR 0.63)**. The cross-encoder reads question and chunk together,
  so it judges whether the chunk answers the question, not only whether it is on topic. It
  is the biggest single gain: 5 more questions (+12 points hit) and the right chunk moved up
  (+0.18 MRR).
- **Hybrid + rerank, the current app (0.90, MRR 0.64)**. The same as vector + rerank:
  full-text rarely adds a useful candidate, and the reranker discards the noise it adds. In
  practice the app runs vector + rerank and pays for an extra query.
- **Hybrid OR + rerank (0.88, MRR 0.66)**. Keyword candidates now take slots in the 40-cut
  from vector hits and one question loses its gold chunk, while MRR rises slightly; both are
  within noise, and the end-to-end run confirmed answers did not improve.
- **What is still missed at 0.90** (4 questions): `abn-2024-cfo` is a gold-label gap (the
  name is on other pages too and facts passes); ASML EUV units and CM.com ARR had the answer
  at fusion rank 41 to 60, which 60 candidates fix; the CM.com revenue comparison needs the
  2025 key-figures infographic, a badly chunked page (see End to end).
- **Takeaway**: the reranker does the work, so give it more to choose from (the sweep
  below). Keyword search, meant for exact terms such as "Scope 3" or "FTE", needs its query
  built from key terms rather than the whole sentence, or should be dropped.

**Parameter sweeps** (hybrid + rerank):

| | hit@k | MRR | facts |
|---|---|---|---|
| candidates 10 / 20 / **40** / 60 / 80 | 0.76 / 0.83 / **0.90** / 0.98 / 0.98 | 0.58 / 0.63 / **0.64** / 0.69 / 0.69 | 0.68 / 0.76 / **0.85** / 0.88 / 0.88 |
| top_k 3 / 5 / **8** / 12 / 16 | 0.73 / 0.88 / **0.90** / 0.95 / 0.95 | | 0.76 / 0.78 / **0.85** / 0.85 / 0.90 |
| RRF k 10 / 30 / **60** / 100 | 0.90 for all | | |
| **candidates 60 + top_k 12 (tuned)** | **0.98** | **0.69** | 0.88 (factoids 1.00) |

- For 3 questions the gold chunk was at fusion rank 41 to 60, so it never reached the
  reranker; beyond 60 nothing is gained. Reranking costs about 70 ms per pair on MPS, so 60 candidates
  add about 1.4 s.
- RRF k does not matter here: with full-text contributing little, fusion barely reorders.
- Balancing the top_k across the reports in a comparison's scope made comparisons worse
  (facts 0.50 to 0.38): the 2025 report usually states both years, and forced slots for the
  2024 report pushed out the chunk with both figures.

**Chunking variants** (re-chunked and re-embedded, tuned settings):

| Index | Chunks | hit@12 | facts | table facts | Rerank ms/pair |
|---|---|---|---|---|---|
| 200 words, 30 overlap | 7,453 | 0.95 | 0.88 | 0.87 | 78 |
| **400 / 60 (current)** | 4,625 | **0.98** | 0.88 | 0.80 | 67 |
| 600 / 90 | 3,756 | 0.98 | 0.90 | 0.87 | 115 |
| 400 / 60, no contextual header | 4,625 | 0.85 | 0.76 | **0.47** | 82 |

- The contextual header `[Company | Annual Report 2025 | p. 87]` is essential. A table chunk
  is numbers and labels; without the header it does not say whose numbers they are.
- 400 to 600 words is a plateau. 600 does as well at the default top 8, but each passage
  costs 70% more to rerank and often exceeds the reranker's 1,024-token window. 400 stays.

**Scope.** The pipeline's company and year scope contains the gold report for 88% of
questions and averages 1.15 reports. The misses are follow-ups and cross-year comparisons
scoped to one year; they were still answered because the 2025 report restates 2024. Without
scoping (all six reports), report precision falls from 1.00 to 0.83, so scoping is what
keeps 2024 and 2025 apart.

**Abstention threshold.** Best rerank score for every answerable question: 0.949 or more.
Out-of-corpus questions (Shell, Heineken): 0.10 and 0.18. In-corpus unanswerable questions:
0.77 to 0.995. The current threshold of 0.003 never fires. A threshold of 0.2 catches the
out-of-corpus questions with no false abstention. Plausible unanswerable questions cannot be
caught this way; the model has to say so, and it does (below).

## End to end

| | Baseline (current) | Tuned | Tuned + OR full-text | Tuned + Haiku answers |
|---|---|---|---|---|
| Correct (answerable) | 87.8% | **92.7%** | 90.2% | 87.8% |
| Correct or partial | 97.6% | **100%** | 100% | 90.2% |
| Incorrect | 0% | 0% | 0% | 7.3% |
| False abstention | 0% | 0% | 0% | 24.4% |
| Factoid / comparison / follow-up correct | 93 / 63 / 100% | **100 / 63 / 100%** | | 97 / 50 / 100% |
| Text / table questions correct | 92 / 80% | 96 / 87% | | |
| Unanswerable handled correctly | 100% | 100% | 100% | 100% |
| Grounded (every figure in a cited quote) | 84.6% | 88.5% | 84.6% | 69.2% |
| Citations verified | 100% | 100% | 99.2% | 74.6% |
| Latency p50 / p90 | 6.1 / 10.3 s | 7.4 / 9.1 s | 7.7 / 10.0 s | 6.3 / 7.8 s |
| Per question: input tokens, cost | 7.1k, 2.4¢ | 9.7k, 3.2¢ | | 0.8¢ |

Tuned is candidates 60, top_k 12, rerank threshold 0.2. Median stage times for tuned:
analysis 1.0 s, retrieval 3.1 s (embedding, two searches and 60 rerank pairs), generation
3.2 s.

- **Tuned vs baseline**: the two gains are retrieval misses the offline eval predicted (ASML
  EUV units, CM.com ARR); nothing got worse. A second run of tuned reproduced every verdict.
- **OR full-text**: no gain, one question noisier. Not worth it on this set; see
  recommendations for what to do instead.
- **Haiku for answers**: a quarter of the cost, but a quarter of its quotes are not
  verbatim. The verifier downgrades those answers to "not found" (the false abstentions), and
  it got three questions wrong, among them calling a lower dividend "higher". The safety net
  held: nothing was hallucinated on the unanswerable questions. Keep Sonnet for answers;
  Haiku is fine for query analysis.
- **Ungrounded answers** are all side figures: the right answer plus a number from a chunk
  it did not quote (e.g. a final dividend of €1.84). Quote verification checks quotes, not
  the answer text. A deterministic check (does every number in the answer appear in a
  verified quote, `eval/number_check.py`) flagged all 5 answers the judge found ungrounded,
  plus 9 false alarms (rescaled figures such as 28,262.9 million as 28.3 billion, numbers
  from the conversation history). Good enough for a warning badge, not for blocking.

**What still fails** (tuned):

| Question | What happened | Cause |
|---|---|---|
| CM.com revenue 2024 to 2025 | 2024 figure only, says 2025 not found | 2025 figure is on the key-figures infographic (p. 2) |
| CM.com year-end FTE 2024 to 2025 | Uses average FTE, says year-end 2025 not stated | Same page: "FTE Year-End 2025 616" sits in a 388-word chunk of marketing prose and navigation bar |
| ASML vs ABN AMRO employees | ABN AMRO average instead of year-end FTE | One query for two companies |
| ASML vs CM.com revenue | Partial | One query for two companies |
| ASML gross margin 2024 to 2025 | Partial | Scoped to one year |

## Extraction

**FTE** (registry: group-wide, year-end FTE):

| Report | Extracted | Gold | Outcome | Cause |
|---|---|---|---|---|
| ABN AMRO 2025 | 23,126 | 23,126 | Exact | |
| ABN AMRO 2024 | 21,976 | 21,976 | Exact | |
| ASML 2025 | 43,520 | 44,209 | Alternative: payroll only, excludes temporary staff | Definition |
| ASML 2024 | 44,027 | 44,027 | Exact | |
| CM.com 2025 | 648 | 616 | Alternative: average FTE | Gold page never retrieved (same p. 2 as above) |
| CM.com 2024 | 666 | 666 | Exact | |

4/6 exact, 6/6 exact or a defensible alternative, all quotes verified. The instruction
should say whether temporary staff count.

**Sustainability goals.** Precision is 1.0 (every extracted item is a real, verified goal),
recall is the problem:

| Experiment | Recall | Headline recall | Precision | Candidate-page recall | Cost (6 reports) |
|---|---|---|---|---|---|
| **Current** (top 24 chunks, 5 queries) | 0.50 | 0.76 | 1.00 | 0.54 | in ingestion |
| + 3 queries | 0.53 | 0.73 | 0.94 | 0.59 | $0.70 |
| top 48 (6,000-token cap) | 0.58 | 0.84 | 0.83 | 0.71 | $1.37 |
| top 48, 16,000-token cap | 0.53 | 0.81 | 0.94 | 0.71 | $0.97 |
| top 48 + queries, 6,000 cap | 0.36 | 0.53 | 0.58 | 0.54 | $1.42 |
| top 48 + queries, 16,000 cap | 0.55 | 0.78 | 0.84 | 0.74 | $1.07 |

- **Retrieval limits recall first**: with 24 chunks, half the gold goals' pages are never
  shown to the model. The misses cluster by theme: ABN AMRO's sector intensity targets
  (mortgages, real estate, oil and gas, shipping), ASML's product energy and circularity
  targets, CM.com's diversity and training targets.
- **More chunks raise candidate recall by 17 points but goal recall by only 3 to 8**: one
  model call over 48 chunks lists the prominent goals and skims the rest. The two runs on
  identical chunks differ by 4 points, so extraction is noisier than chat.
- **Bug**: goal lists from 48 chunks overflow `max_tokens=6000` (JSON runs about 1.25
  characters per token). The provider does not check `stop_reason == "max_tokens"`, so the
  cut-off reply looks like invalid JSON and the repair retry overflows again. Both ASML
  reports lost every goal in one run. Today's top 24 is already close: the ASML 2025 goals
  call used about 5,000 tokens.
- Precision drops with more chunks because FY2025 reports describe 2025 targets that are now
  results. The instruction should say that a target dated in or before the fiscal year is
  not a goal.

## Recommendations

Ordered by value for effort. "Measured" means verified in this evaluation; "expected" means
supported by the failure analysis but not yet run.

| # | Change | Where | Evidence | Effort |
|---|---|---|---|---|
| 1 | **Applied.** `RETRIEVAL_CANDIDATES` 40 to **60**, `RERANK_TOP_K` 8 to **12** | `retrieval/search.py` | Measured: correct 87.8 to 92.7%, factoids 100%, +1.3 s, +0.8¢ per question | Constants |
| 2 | **Applied.** `RERANK_MIN_SCORE` 0.003 to **0.2** | `retrieval/search.py` | Measured: out-of-corpus questions skip the LLM, no false abstention (min answerable score 0.949) | Constant |
| 3 | Check `stop_reason == "max_tokens"` and raise a clear error; raise the extraction cap to 16,000 | `llm/anthropic_provider.py`, `extraction/extractor.py` | Measured: truncation makes goal extraction fail silently | Small |
| 4 | Embed all chunks of a report in one `encode` call (or slices of 512) | `ingestion/pipeline.py` | Measured: +47% embedding throughput, identical vectors; about 15% off total ingestion | Small |
| 5 | Query decomposition for comparisons: when the analysis names several companies or years, retrieve per company and year and merge | `chat/pipeline.py`, `chat/schemas.py` | Expected: fixes 3 of the 5 remaining failures; comparisons are 63% correct | Medium |
| 6 | Map-reduce goal extraction: batches of about 12 candidate chunks, then dedupe; widen candidates to the sustainability chapter | `extraction/extractor.py` | Expected: candidate recall is the cap at 24 chunks, a single call is the cap at 48 | Medium |
| 7 | Strip lines repeated on 20% or more of a report's pages before chunking | `ingestion/parser.py` or `chunker.py` | Expected: CM.com navigation bar in half its chunks; behind the key-figures misses | Small, re-ingest |
| 8 | Fix or remove full-text search: OR query (as tested) or drop it | `retrieval/search.py` | Measured: AND query matches nothing for 54% of questions; OR did not improve answers | Small |
| 9 | Warn when a number in the answer is in no verified quote | `chat/pipeline.py`, UI | Measured: catches 5/5 ungrounded answers, 9 false alarms until scale normalisation | Small |
| 10 | Registry wording: FTE including or excluding temporary staff; goals dated after the fiscal year | `extraction/registry.yaml` | Measured: both FTE alternatives and the precision drop come from this | Config |
| 11 | Parallelise ingestion: the two extraction calls concurrently, parse pages in a process pool | `ingestion/pipeline.py`, `parser.py` | Expected: extraction and parse are 52% of ingestion | Medium |

Kept as they are, with evidence: chunk size 400 words with 60 overlap (200 is worse, 600 is
no better and costs more to rerank), the contextual header (+12 points), RRF k = 60 (no
effect), Sonnet for answers (Haiku: 25% unverifiable quotes), Haiku for query analysis.

**Chosen parameters**: candidates 60, top_k 12, rerank threshold 0.2, chunk 400/60 with
header; for extraction, a 16,000-token output cap whatever the top_k, and top_k 48 only
together with map-reduce extraction.

## Limits of this evaluation

- 52 questions over 3 companies. Enough to rank configurations with large effects, not to
  separate ones within about 5 points; comparisons (8) are especially thin.
- The gold set was drafted with LLM help. Quotes and pages are checked in code, but the
  choice of what counts as "the" answer (e.g. internal vs total FTE) is one labeller's.
- One LLM judge, a different and stronger model than the generator, without a human
  agreement study. Its verdicts were spot-checked on every failure listed above.
- Latency was measured on one machine with the models warm and one question at a time.
- The OpenAI provider was not evaluated.
