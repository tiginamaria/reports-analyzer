# Evaluation

The evaluation report of the annual reports analysis application described in [DESIGN.md](DESIGN.md).

* Run on 2026-10-18 on an Apple M-series laptop (MPS)
* The harness and how to rerun or run an experiment: [backend/eval/README.md](../backend/eval/README.md).
* Raw results by stage in `backend/eval/results/{ingestion,retrieval,extraction,e2e}.json`

## Setup

The evaluation was performed for `ingestion` pipeline (including separate for `extraction` phase) and `answering` pipeline (including separate `retrieval` phase). 

### Corpus

Six annual reports, all downloaded from the companies' investor relations sites or public
archives (links in [data/reports/SOURCES.md](../data/reports/SOURCES.md) and
[backend/eval/corpus.yaml](../backend/eval/corpus.yaml)). 

Three companies of different kinds, so the pipeline meets different vocabularies and
layouts: ABN AMRO (a bank), ASML (semiconductor equipment), CM.com (a small software
company). Two fiscal years per company, 2024 and 2025, so questions can compare years and
the system has to tell the years apart: every 2025 report restates the 2024 figures. The
Shell and Heineken sample reports are left out on purpose, so two gold questions about them
test a company that is not in the corpus.

A chunk is a text window or a whole table: page text (tables removed) is cut into windows of
**400** words with **60** words of overlap (15%); each table is one chunk, never cut. Every
chunk is embedded behind a header with company, report year and page.

| Report | Pages | Chunks | Table chunks | Median words per chunk | Longest chunk |
|---|---|---|---|---|---|
| ABN AMRO 2025 | 434 | 928 | 197 | 280 | 710 |
| ABN AMRO 2024 | 508 | 1,134 | 215 | 321 | 674 |
| ASML 2025 | 350 | 838 | 158 | 309 | 549 |
| ASML 2024 | 410 | 960 | 231 | 281 | 670 |
| CM.com 2025 | 159 | 343 | 81 | 209 | 400 |
| CM.com 2024 | 212 | 422 | 69 | 217 | 400 |
| **Total** | 2,073 | 4,625 | 951 | | |

### Gold extraction labels

[backend/eval/gold/extraction.yaml](../backend/eval/gold/extraction.yaml): one label per
report for each single-value datapoint (expected figure, page, quote, acceptable
alternatives) and a list of sustainability goals per report. Values were extracted by hand,
verified by Claude Fable 5.1 against the page text, then proofread by a human once more. A
script (`gold_check`) checks every quote against the ingested page text.

| Datapoint | Labels | Expected figure | Acceptable alternatives (also in the report) | Example |
|---|---|---|---|---|
| Year-end FTE | 6 | Group-wide FTE at year-end | Average FTE, headcount, payroll-only FTE, company-only FTE, a rounded prose figure | ABN AMRO 2025: 23,126 expected; 22,676 (average), 24,632 (headcount), 18,703 (bank only) accepted |
| Revenue | 6 | Income statement total: total net sales, revenue, or total operating income for the bank | The same total rounded to billions or millions | ASML 2024: EUR 28,262.9 million expected; EUR 28.3 billion accepted |
| Net profit | 6 | Profit for the period before minority interests, as the income statement states it | The same figure rounded or rescaled | CM.com 2024: (19,813) EUR thousand expected; -19.8 million accepted |
| Dividend per share | 6 | Total dividend for the year (interims plus final) | None: a final or an interim dividend alone is wrong | ABN AMRO 2025: 1.24 expected; 1.54 (includes a special distribution), 0.70 (final only) are wrong. CM.com pays no dividend: the label is null and the correct extraction is nothing |
| Sustainability goals | 123 (41 `core`) | Every forward-looking, quantified or clearly defined commitment, with target, year, page and quote | Matched by meaning (an LLM judge); goals whose target year is in or before the fiscal year are excluded at scoring, as the registry asks the model to skip them (121 remain) | ABN AMRO 2025: "net-zero emissions by 2050", core |

| Report | Gold goals | Of which core |
|---|---|---|
| ABN AMRO 2025 | 23 | 8 |
| ABN AMRO 2024 | 24 | 7 |
| ASML 2025 | 21 | 8 |
| ASML 2024 | 30 | 8 |
| CM.com 2025 | 11 | 5 |
| CM.com 2024 | 14 | 5 |
| **Total** | 123 | 41 |

### Gold questions

57 questions in [backend/eval/gold/qa.yaml](../backend/eval/gold/qa.yaml), 46 answerable and 11 unanswerable.

One question per case (type, text or table source, single or multi-report, with or without a
trap) was written by hand with its gold answer; Claude Fable 5.1 then multiplied each case
across the other companies and years from the page text. Each answerable question has a
gold answer, key facts as the report writes them, and evidence quotes with pages, checked by
`gold_check`. Questions are phrased away from the report's wording, so keyword overlap is
not rewarded.

| Type | n | What it tests | Example | Gold answer |
|---|---|---|---|---|
| Factoid | 30 (20 text, 10 table) | One figure from one report | *How strong was ABN AMRO's core capital position (CET1) at year-end 2025?* | CET1 ratio of 15.4% at year-end 2025. Trap: the solo-entity ratio of 14.6% on another page |
| Comparison | 8 (6 across years, 2 across companies) | Two figures from two reports, with the direction of change | *Did ABN AMRO pay a higher or lower dividend per share for 2025 than for 2024?* | Lower: EUR 1.24 vs EUR 1.35. Trap: the key-figures table shows 1.54, which includes a special distribution |
| Multi-metric | 5 | Three figures of one report from three different sections in one question | *For 2025, what were ABN AMRO's CET1 ratio, its cost/income ratio and the share of women in senior management?* | 15.4%; 64.4%; 31% (below the 35% target) |
| Follow-up | 3 | A question that only makes sense with the conversation history | *And what was it the year before?* after a question on ABN AMRO's 2025 return on equity | 10.1% in 2024 |
| Unanswerable | 11 | 9 plausible figures the reports do not disclose, 2 about companies outside the corpus | *How many ATMs did ABN AMRO operate in the Netherlands at the end of 2025?*; *How many hectolitres of beer did Heineken sell in 2024?* | Not stated. The labeller searched both reports for ATM, cash machine, geldautomaat, Geldmaat and found no count |

## Extraction

**What is evaluated.** The pre-extraction stage of ingestion. For every registry entry
([registry.yaml](../backend/app/extraction/registry.yaml)) and every report, the eval runs
the same two functions ingestion runs: candidate retrieval (the entry's queries through
hybrid search and reranker, cut to `top_k`) and the model call with quote verification. Each
entry is scored and costed on its own.

| Datapoint (registry key) | Kind | Retrieval queries | Chunks to the model | Model calls per report |
|---|---|---|---|---|
| Employees, `fte` | single value | 3 | 10 | 1 |
| Revenue, `revenue` | single value | 3 | 10 | 1 |
| Net profit, `net_income` | single value | 3 | 10 | 1 |
| Dividend per share, `dividend_per_share` | single value | 3 | 10 | 1 |
| Sustainability goals, `sustainability_goal` | multi value | 6 | 32 | 1 |

**How outcomes are scored.** Two methods, by kind of datapoint:

- **The four single values are scored in code, no model.** The extracted number and the gold
  figure are scaled by their unit word (thousand, million, billion) and compared within
  0.01%, so "28.3 EUR billion" equals "28,262.9 EUR million"; the same check runs against
  each labelled alternative. Deterministic.
- **Sustainability goals are scored by an LLM judge** (`claude-opus-5-5`), because a goal is
  a sentence. It gets the gold goals and the extracted items as two numbered lists and
  returns which extracted item states each gold goal (or none) and whether each extracted
  item is a real forward-looking goal. Recall = gold goals matched; precision = extracted
  items that are real goals with a verified quote. One item can satisfy one gold goal.

Each single-value extraction gets one of six outcomes:

| Outcome | Meaning | Counted as correct |
|---|---|---|
| `exact` | Matches the gold figure | yes |
| `alternative` | Matches an alternative the report also states (average instead of year-end FTE, a rounded figure) | yes, as acceptable |
| `none` | The report states no such figure and nothing was extracted (CM.com dividend) | yes |
| `wrong` | A figure that matches neither | no |
| `missing` | Nothing extracted where the report states a figure | no |
| `spurious` | A figure extracted where the report states none | no |

| Datapoint | Outcomes | Gold page among candidates | Quote verified exactly | Chunks to model | Cost, 6 reports |
|---|---|---|---|---|---|
| FTE | 5 exact, 1 alternative | 4/6 | 6/6 | 7 | $0.10 |
| Revenue | 4 exact, 1 alternative, 1 missing | 4/6 | 5/5 | 10 | $0.11 |
| Net profit | 6 exact | 1/6 | 6/6 | 10 | $0.10 |
| Dividend per share | 4 exact, 2 correctly none | 6/6 | 4/4 | 9 | $0.10 |

Per report, the four single values:

| Report | FTE | Revenue | Net profit | Dividend per share |
|---|---|---|---|---|
| ABN AMRO 2025 | 23,126 exact | 8,716 EUR m exact | 2,252 EUR m exact | 1.24 exact |
| ABN AMRO 2024 | 21,976 exact | 8,861 EUR m exact | 2,403 EUR m exact | 1.35 exact |
| ASML 2025 | 44,209 exact | 32,667.3 EUR m exact | 9,609.4 EUR m exact | 7.50 exact |
| ASML 2024 | 44,027 exact | 28,263 EUR m exact | 7,571.6 EUR m exact | 6.40 exact |
| CM.com 2025 | **646 headcount, alternative** (gold 616 year-end FTE) | **missing** (gold 259,394 EUR k) | (3,786) EUR k exact | none, correct |
| CM.com 2024 | 666 exact | **274.2 EUR m, alternative** (gold 274,249 EUR k) | (19,813) EUR k exact | none, correct |

No figure is wrong and every quote verified. Net profit is 6/6 exact although the gold page
was a candidate only once: the figure is restated on many pages and the model found it
elsewhere. The column measures agreement with the labeller's page, not success.

**Why the alternatives and the miss**

- **CM.com 2025 FTE and revenue**: the exact figures are on the key-figures infographic
  (p. 2) and in the income statement (p. 100), and neither page was retrieved. The
  infographic comes out as numbers without labels, and the statement's table chunk is not
  among the 10 candidates. The model fell back to the headcount (646) and said so; for
  revenue it returned nothing rather than guess.
- **CM.com 2024 revenue**: the income statement was not retrieved either, so the model took
  the rounded "274.2 million" from the key-figures page.

**Sustainability goals**

| Report | Gold goals (core) | Extracted | Recall | Core recall | Precision | Gold pages among candidates | Cost |
|---|---|---|---|---|---|---|---|
| ABN AMRO 2025 | 22 (8) | 18 | 0.55 | 0.63 | 0.89 | 0.59 | $0.11 |
| ABN AMRO 2024 | 24 (7) | 22 | 0.54 | 0.86 | 1.00 | 0.67 | $0.12 |
| ASML 2025 | 21 (8) | 15 | 0.52 | 0.88 | 0.93 | 0.33 | $0.10 |
| ASML 2024 | 30 (8) | 18 | 0.57 | 0.63 | 1.00 | 0.43 | $0.10 |
| CM.com 2025 | 10 (5) | 7 | 0.70 | 0.80 | 1.00 | 1.00 | $0.08 |
| CM.com 2024 | 14 (5) | 7 | 0.50 | 0.60 | 1.00 | 0.93 | $0.07 |
| **All** | 121 (41) | 87 | **0.56** | **0.73** | **0.97** | 0.66 | $0.58 |

Each report ran 6 queries and sent 32 chunks to the model in one call (about 27k input
tokens, 35 to 62 s).

- **Precision 0.97**: almost everything extracted is a real goal with a verified quote.
- **Recall 0.56**, for two reasons. Most missed goals (37 of 54) are on pages that were not
  among the 32 chunks, so retrieval caps recall at 0.69; all ASML misses are of this kind,
  because its goals (energy per wafer, waste, safety, suppliers) are on pages the six
  queries do not describe. The other 17 were in the chunks but skipped: one call over 32
  chunks lists the main goals and skims the rest.

**Improvements**

1. **Full-page expansion for goals.** Use the retrieved chunks only to find the
   sustainability pages, then send those whole pages to the model. The targets next to a
   found chunk get read too.
2. **Read infographics with a vision model.** CM.com's key figures are graphics that come
   out as numbers without labels; a vision call on such pages would fix the two CM.com
   extraction misses and one chat failure.

## Ingestion

**What is evaluated.** The six PDFs are ingested from scratch into an empty database by the
app's own stages, with a timer around each stage and every LLM call's tokens and cost
recorded. Measured: seconds per stage, LLM usage, metadata against the corpus, chunks,
datapoints written, and whether the report reached `ready`. Apple M-series (MPS), models
warm, one report at a time.

| Report | Pages | Chunks (tables) | Total | Parse | Metadata | Chunk + store | Embed | Extract | LLM calls | Tokens in / out | LLM cost | Outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ABN AMRO 2025 | 434 | 928 (197) | 220 s | 37 s | 1 s | 0.1 s | 64 s | 119 s | 6 | 61.5k / 7.0k | $0.18 | ready |
| ABN AMRO 2024 | 508 | 1,134 (215) | 258 s | 45 s | 1 s | 0.1 s | 81 s | 131 s | 6 | 63.5k / 9.1k | $0.21 | ready |
| ASML 2025 | 350 | 838 (158) | 201 s | 43 s | 1 s | 0.1 s | 49 s | 108 s | 6 | 55.9k / 6.5k | $0.16 | ready |
| ASML 2024 | 410 | 960 (231) | 216 s | 47 s | 1 s | 0.1 s | 54 s | 114 s | 6 | 55.4k / 7.0k | $0.17 | ready |
| CM.com 2025 | 159 | 343 (81) | 117 s | 12 s | 1 s | 0.0 s | 17 s | 87 s | 6 | 56.4k / 3.9k | $0.14 | ready |
| CM.com 2024 | 212 | 422 (69) | 128 s | 15 s | 1 s | 0.0 s | 22 s | 91 s | 6 | 58.2k / 4.6k | $0.15 | ready |
| **Total** | 2,073 | 4,625 (951) | **19.0 min** | 199 s (17%) | 6 s (1%) | 0.4 s | 287 s (25%) | 649 s (57%) | 36 | 351k / 38k | **$1.01** | 6 of 6 ready |

Company and fiscal year were detected for 6/6 reports (one Haiku call over the first three
pages). Chunk statistics are in the corpus table above.

Datapoints written by ingestion (what the Extracted data page shows):

| Report | FTE | Revenue | Net profit | Dividend per share | Sustainability goals |
|---|---|---|---|---|---|
| ABN AMRO 2025 | 1 | 1 | 1 | 1 | 18 |
| ABN AMRO 2024 | 1 | 1 | 1 | 1 | 28 |
| ASML 2025 | 1 | 1 | 1 | 1 | 15 |
| ASML 2024 | 1 | 1 | 1 | 1 | 18 |
| CM.com 2025 | 1 | 0 (not found) | 1 | 0 (pays none, correct) | 7 |
| CM.com 2024 | 1 | 1 | 1 | 0 (pays none, correct) | 7 |
| **Total** | 6 | 5 | 6 | 4 | 93 |

**Why the time goes where it goes**

- **Extraction, 57%** (87 to 131 s per report): six model calls run one after another, half
  the time waiting for the model (30 to 53 s), half retrieving candidates for 18 queries.
  It is also 95% of the cost (14 to 21 cents per report).
- **Embedding, 25%** (17 to 81 s): about 14 chunks per second, encoded in slices of 512
  chunks.
- **Parsing, 17%** (12 to 47 s): PyMuPDF text and table detection, slowest on ASML's
  table-dense pages.

**Improvements**

1. **Run the six extraction calls per report in parallel.** They are independent; four at
   a time would cut extraction from about 110 s to 40 s per report and ingestion from 19 to
   about 11 minutes, at the same cost.
2. **Strip lines repeated on 20% or more of a report's pages before chunking.** Removes the
   navigation bar; needs a re-ingest.
3. **Read infographic pages with a vision model** when a page has little text and many
   numbers. Fixes the two CM.com extraction misses and one chat failure.

## Retrieval

**What is evaluated.** The chat's own retrieval, live: the small model rewrites each
question into a standalone question and one sub-query per company, year and topic; each
sub-query runs vector and full-text search over all six reports, fused with RRF, the best
60 reranked, chunks under 0.2 dropped, top 8 kept; the sub-queries' chunks are merged.

Metrics, against the gold evidence pages:

- **hit@k**: share of questions where at least one retrieved chunk is from a gold evidence
  page of the right report.
- **MRR**: position of that first evidence chunk: 1 if first, 1/2 if second, 1/3 if third,
  0 if absent; averaged.
- **facts**: share of questions where every gold figure (e.g. "15.4") appears in the
  retrieved text of the right reports, on any page.
- **report precision**: share of retrieved chunks from the right company and fiscal year.
- **report recall**: for comparisons, share of the gold reports that contributed a chunk.

| Questions | n | hit@k | MRR | facts | report precision | report recall |
|---|---|---|---|---|---|---|
| **All answerable** | 46 | **0.957** | **0.608** | **0.935** | **0.794** | 0.989 |
| Factoid | 30 | 0.933 | 0.607 | 0.900 | 0.744 | 1.000 |
| Comparison | 8 | 1.000 | 0.562 | 1.000 | 0.953 | 1.000 |
| Multi-metric | 5 | 1.000 | 0.587 | 1.000 | 0.722 | 1.000 |
| Follow-up | 3 | 1.000 | 0.778 | 1.000 | 1.000 | 0.833 |
| Text questions | 26 | 0.962 | 0.652 | 0.962 | 0.815 | 1.000 |
| Table questions | 15 | 0.933 | 0.541 | 0.867 | 0.783 | 0.967 |

First gold chunk: rank 1 for 21 questions, rank 2 to 3 for 12, rank 4 to 8 for 8, rank 9 to
12 for 3 (chunks from a second sub-query), 2 misses.

| Sub-queries per question | Questions | Chunks retrieved | Median seconds |
|---|---|---|---|
| 1 | 41 | 8 | 4.6 |
| 2 | 11 | 10 to 16 | 9.1 |
| 3 | 5 | 20 to 24 | 13.5 |

Median 4.8 s, p90 9.7 s; time is linear in sub-queries (one embedding, two searches and 60
reranker pairs each).

**The two misses**

| Question | What was retrieved | Cause |
|---|---|---|
| ASML EUV systems recognised in 2025 (gold: 48, a table on p. 66) | Eight chunks of EUV prose from both years, none from the sales-by-system table | The table row does not say "EUV"; vector search prefers prose and the table chunk never reaches the reranker |
| ABN AMRO CET1 at year-end 2025 (gold: 15.4%) | 2025 capital pages, but three of eight chunks are from 2024 and the headline chunk fell out | Year leakage, see below; `facts` passes because 15.4% is on another retrieved page |

**Year leakage is the main weakness.** 33 of 46 questions got chunks from the other year of
the same company (report precision 0.79), because the year is only a word in the search
text, not a filter, and a 2025 page quoting the 2024 figure reranks as high as the 2024
page. No wrong answer resulted, but on four questions half the context was the wrong year.

**Comparisons work.** All 8 retrieved both sides, one sub-query per year or company.

**Improvements**

1. **Filter each sub-query by its company and fiscal year.** The rewrite already outputs
   both; passing them as a report filter should take report precision from 0.79 to about
   1.0, recover the CET1 miss and free half the context on year-specific questions.
2. **12 chunks per sub-query instead of 8.** Three gold chunks landed at rank 9 to 12.
   Testable with `--top-k 12` on both evals.
3. **Full-text search contributes little.** It was meant for exact terms (FTE, Scope 3) but
   gets whole sentences; build it from the sub-query's key terms, or drop it.

## End to end

**What is evaluated.** The whole chat pipeline (`run_chat`): rewrite and decomposition,
retrieval and reranking per sub-query, the structured answer with the conversation as
earlier turns, quote verification. Each gold question is asked once, one at a time (honest
latency), with its history for follow-ups and no report filter.

**How answers are scored.** Two layers:

| Layer | What it checks | Needs |
|---|---|---|
| Deterministic | Status (answered, partial, not found); every citation's verification (exact, fuzzy, failed); whether a verified citation is on a gold page; whether any retrieved chunk is on a gold page (separates retrieval failures from generation failures); latency per stage; tokens and cost | Code only |
| LLM judge | Correctness against the gold answer and groundedness of the figures in the cited quotes | One `claude-opus-5-5` call per question |

**LLM as a judge for the answers.** `claude-opus-5-5`, a different and stronger model than
the generator, one call per question, not counted in the cost. It gets the question (and
the conversation for a follow-up), the gold answer and the labeller's notes naming the
traps, the system's status and answer, and the cited quotes with their verification status;
not the chunks or the report. It returns `correctness` (correct, partial, incorrect,
abstained), `grounded` and an explanation. Correct allows rounding, unit rewrites and extra
correct detail; incorrect means a wrong figure or fact; for an unanswerable question,
correct means it said the figure is not disclosed. Grounded means every figure in the answer
is in a cited quote. Every verdict other than correct, and every ungrounded one, was read by
hand.

| Metric | Result |
|---|---|
| Correct (answerable, 46) | **91.3%** (42) |
| Correct or partial | 95.7% (44) |
| Incorrect | **0%** |
| False abstention | 4.3% (2) |
| Unanswerable handled correctly (11) | **100%**; 10 "not found", 1 "partial" that clearly says the figure is not disclosed |
| Hallucinated on unanswerable | 0% |
| Retrieval reached a gold page | 95.7% |
| A verified citation on a gold page | 87.0% |
| Citations per answer | 1.86 (106 in all); 9 answers cite nothing, all abstentions |
| Citations verified verbatim / fuzzy / failed | **100%** / 0% / 0% |
| Grounded (every figure in a cited quote) | 84.2% |
| Latency p50 / p90 | 7.5 s / 15.4 s (rewrite 1.1 s, retrieval 3.8 s, generation 2.4 s at p50) |
| Cost | $1.03 for 57 questions, 1.8 cents each (497k input tokens, 22k output; the judge not counted) |

| Type | n | Correct | Correct or partial | Median latency |
|---|---|---|---|---|
| Factoid | 30 | 90% | 93% | 7.4 s |
| Comparison | 8 | **100%** | 100% | 11.4 s |
| Multi-metric | 5 | 80% | 100% | 16.4 s |
| Follow-up | 3 | 100% | 100% | 6.5 s |
| Unanswerable | 11 | 100% | | 6.3 s |
| Text questions | 26 | 96% | 96% | |
| Table questions | 15 | 87% | 93% | |

**What fails**

| Question | Verdict | What happened | Cause |
|---|---|---|---|
| ASML EUV systems recognised in 2025 (gold 48) | Partial | Says the total is not stated; gives the 4 High NA systems | Retrieval: the sales table never retrieved |
| ASML share of 2025 sales from China (gold 29.1%) | Abstained | Gives South Korea's 25.0%, says China is not stated | Retrieval: the gold page is retrieved but the chunk with China is outside the top 8 |
| CM.com total market-based footprint 2024 (gold 4,676 tCO2e) | Abstained | Says the emission tables are not included | Generation: the table chunk with 4,676 was retrieved but not read |
| ASML 2024 R&D, scope 1 and 2 emissions, year-end FTE | Partial | R&D and emissions right; FTE given as "more than 44,000" | Generation: 44,027 was retrieved (a table chunk); the model used the prose figure |

Two retrieval failures, two generation failures; in all four the model said what it could
not find rather than inventing it.

**Groundedness** is 84%, and all 9 ungrounded answers are correct: each gives the right
figure with a verified quote and adds a side figure (a baseline, a one-off) read in a chunk
but not quoted. Quote verification checks quotes, not the prose, so the UI does not show it.

**Improvements**

1. **The two retrieval changes above** (year filter, top 12) address three of the four
   failures, for about 1 s and 0.5 cents per question.
2. **Warn on unsupported figures.** Check in code that every number in the answer is in a
   verified quote (with scale normalisation) and flag the rest as "figure not quoted".

## Limits of this evaluation

- 57 questions over three companies: enough to rank changes with large effects, not to
  separate ones within about 5 points; 8 comparisons and 5 multi-metric questions are thin.
- What counts as "the" answer (internal vs total FTE, ordinary vs total dividend) is one
  labeller's choice, mitigated by the acceptable alternatives.
- One LLM judge, no human agreement study.
- Latency on one machine, models warm, one question at a time.
- Only the Anthropic provider was evaluated.
