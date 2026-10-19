# Evaluation

The evaluation report of the annual reports analysis application described in [DESIGN.md](DESIGN.md).

* Run on 2026-10-19 with the application's default configuration: **OpenAI for the LLM and
  for embeddings, no reranker**. Only PDF parsing runs on the machine (an Apple M-series
  laptop); every model call is hosted.
* The harness and how to rerun it or run an experiment: [backend/eval/README.md](../backend/eval/README.md).
* Raw results by stage in `backend/eval/results/{ingestion,retrieval,extraction,e2e}_openai.json`;
  the retrieval ablations in `retrieval_openai_vector.json` and `retrieval_openai_rerank.json`.
* The optional local retrieval profile (bge-m3 and a cross-encoder reranker in-process,
  with Claude as the LLM) was evaluated on 2026-10-18; its results are summarised in
  [What to expect from the local retrieval profile](#what-to-expect-from-the-local-retrieval-profile)
  and kept in the untagged result files.

## TL;DR

The evaluation covers the `ingestion` pipeline (with the `extraction` phase on its own) and
the `answering` pipeline (with the `retrieval` phase on its own).

| Stage | Result | Improvements |
|---|---|---|
| Extraction | FTE 3/6 exact and 3 acceptable alternatives, revenue 4/6, net profit 6/6, dividend 4/6 exact and 2/2 correctly none; no figure wrong, every extracted quote verified. Goals: recall 0.66 (core 0.68), precision 0.85 | Send whole sustainability pages to the model instead of 32 chunks; read infographics with a vision model |
| Ingestion | Company and year detected 6/6; a report of 345 pages on average becomes 770 chunks in 1.3 min for $0.17 ($0.14 LLM, $0.04 embeddings); 2.4 times faster than with local models at the same cost | Parse pages in parallel (parsing is now 42% of the time); strip repeated page furniture before chunking |
| Retrieval | A gold page retrieved for 71.7% of questions, MRR 0.41, all key facts for 65.2%; 0.4 s per question. With the local reranker added on top of the same embeddings: 91.3%, 0.60, 91.3% | A reranker in the default profile (hosted or LLM-based); filter each sub-query by its company and year |
| Answers | 65.2% correct, 84.8% correct or partial, 6.5% incorrect; 11/11 unanswerable handled, 0 hallucinations; 84/93 citations verified, 9 failed (quotes stitched from table cells), which withheld 2 correct answers; 9.0 s and $0.014 per question | The reranker; verify table quotes cell by cell; check figures and units in the prose |

The gap to the local profile (91.3% correct, see the last section) is mostly the missing
reranker: 9 of the 16 answers that were not correct never had the evidence retrieved, and
the reranker ablation recovers 9 of the 13 retrieval misses.

## Setup

### Configuration under test

| Component | Default (this report) | Local retrieval profile (last section) |
|---|---|---|
| Main model | `gpt-5`, reasoning effort `low` | `claude-sonnet-5-5` |
| Small model (metadata detection, query rewriting) | `gpt-5-mini` | `claude-haiku-4-5` |
| Embeddings | `text-embedding-3-large`, hosted, 3,072 dimensions | `BAAI/bge-m3` in-process, 1,024 dimensions |
| Reranker | None: the fused hybrid order is kept, no score floor | `BAAI/bge-reranker-v2-m3` in-process, floor 0.2 |
| Needs | One OpenAI API key | `torch`, 4.5 GB of models, a GPU for acceptable speed |
| Judge | `claude-opus-5-5` | `claude-opus-5-5` |

Costs are estimated from list prices (gpt-5 $1.25 per million input tokens, $0.125 cached,
$10 output; gpt-5-mini $0.25 / $0.025 / $2; text-embedding-3-large $0.13 per million
tokens). A reasoning model's hidden reasoning tokens are billed as output and included.

### Corpus

- **Six annual reports**, downloaded from the companies' sites or public archives (links in
  [data/reports/SOURCES.md](../data/reports/SOURCES.md) and
  [backend/eval/corpus.yaml](../backend/eval/corpus.yaml)).
- **Three different companies**: ABN AMRO (a bank), ASML (semiconductor equipment), CM.com
  (a small software company), so the pipeline meets different vocabularies and layouts.
- **Two fiscal years each**, 2024 and 2025: questions can compare years, and the system has
  to tell the years apart, since every 2025 report restates the 2024 figures.
- **Shell and Heineken left out** on purpose: two gold questions about them test a company
  that is not in the corpus.
- **A chunk** is a text window of **400 words with 60 words of overlap (15%)** or one whole
  table, never cut; every chunk is embedded behind a header with company, year and page.

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

[backend/eval/gold/extraction.yaml](../backend/eval/gold/extraction.yaml):

- **One label per report and datapoint**: expected figure, page, quote, and the acceptable
  alternatives the report also states; plus **a list of sustainability goals per report**.
- **Made by hand**, verified by Claude Fable 5.1 against the page text, proofread by a human
  once more; `gold_check` verifies every quote against the ingested page text.

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

[backend/eval/gold/qa.yaml](../backend/eval/gold/qa.yaml): **57 questions, 46 answerable
and 11 unanswerable**.

- **One question per case** (type, text or table source, single or multi-report, with or
  without a trap) written by hand with its gold answer; Claude Fable 5.1 multiplied each
  case across the other companies and years from the page text.
- Each answerable question has a **gold answer, key facts** as the report writes them, and
  **evidence quotes with pages**, verified by `gold_check`.
- Questions are **phrased away from the report's wording**, so keyword overlap is not
  rewarded.

| Type | n | What it tests | Example | Gold answer |
|---|---|---|---|---|
| Factoid | 30 (20 text, 10 table) | One figure from one report | *How strong was ABN AMRO's core capital position (CET1) at year-end 2025?* | CET1 ratio of 15.4% at year-end 2025. Trap: the solo-entity ratio of 14.6% on another page |
| Comparison | 8 (6 across years, 2 across companies) | Two figures from two reports, with the direction of change | *Did ABN AMRO pay a higher or lower dividend per share for 2025 than for 2024?* | Lower: EUR 1.24 vs EUR 1.35. Trap: the key-figures table shows 1.54, which includes a special distribution |
| Multi-metric | 5 | Three figures of one report from three different sections in one question | *For 2025, what were ABN AMRO's CET1 ratio, its cost/income ratio and the share of women in senior management?* | 15.4%; 64.4%; 31% (below the 35% target) |
| Follow-up | 3 | A question that only makes sense with the conversation history | *And what was it the year before?* after a question on ABN AMRO's 2025 return on equity | 10.1% in 2024 |
| Unanswerable | 11 | 9 plausible figures the reports do not disclose, 2 about companies outside the corpus | *How many ATMs did ABN AMRO operate in the Netherlands at the end of 2025?*; *How many hectolitres of beer did Heineken sell in 2024?* | Not stated. The labeller searched both reports for ATM, cash machine, geldautomaat, Geldmaat and found no count |

## Extraction

**What is evaluated**: the pre-extraction stage of ingestion.

- For every registry entry ([registry.yaml](../backend/app/extraction/registry.yaml)) and
  every report, the eval runs **the same two functions ingestion runs**: candidate retrieval
  (the entry's queries through hybrid search, restricted to the report, cut to `top_k`) and
  the model call with quote verification.
- Each entry is **scored and costed on its own**; nothing is written to the database.

| Datapoint (registry key) | Kind | Retrieval queries | Chunks to the model | Model calls per report |
|---|---|---|---|---|
| Employees, `fte` | single value | 3 | 10 | 1 |
| Revenue, `revenue` | single value | 3 | 10 | 1 |
| Net profit, `net_income` | single value | 3 | 10 | 1 |
| Dividend per share, `dividend_per_share` | single value | 3 | 10 | 1 |
| Sustainability goals, `sustainability_goal` | multi value | 6 | 32 | 1 |

**How outcomes are scored**: two methods, by kind of datapoint.

- **Single values: in code, no model.** Extracted and gold numbers are scaled by their unit
  word (thousand, million, billion) and compared within 0.01%, so "28.3 EUR billion" equals
  "28,262.9 EUR million"; the same check runs against each labelled alternative.
- **Sustainability goals: an LLM judge** (`claude-opus-5-5`), because a goal is a sentence.
  It gets the gold goals and the extracted items as two numbered lists and returns which
  item states each gold goal (or none) and whether each item is a real forward-looking
  goal. **Recall** = gold goals matched; **precision** = items that are real goals with a
  verified quote; one item can satisfy one gold goal.

Each single-value extraction gets one of six outcomes:

| Outcome | Meaning | Counted as correct |
|---|---|---|
| `exact` | Matches the gold figure | yes |
| `alternative` | Matches an alternative the report also states (average instead of year-end FTE, a rounded figure) | yes, as acceptable |
| `none` | The report states no such figure and nothing was extracted (CM.com dividend) | yes |
| `wrong` | A figure that matches neither | no |
| `missing` | Nothing extracted where the report states a figure | no |
| `spurious` | A figure extracted where the report states none | no |

| Datapoint | Outcomes | Gold page among candidates | Quote verified exactly | Chunks to model | Seconds per call | Cost, 6 reports |
|---|---|---|---|---|---|---|
| FTE | 3 exact, 3 alternative | 2/6 | 6/6 | 7 | 13 | $0.09 |
| Revenue | 4 exact, 1 alternative, 1 missing | 4/6 | 5/5 | 10 | 9 | $0.08 |
| Net profit | 6 exact | 6/6 | 6/6 | 10 | 11 | $0.08 |
| Dividend per share | 4 exact, 2 correctly none | 5/6 | 4/4 | 9 | 9 | $0.06 |

- **No figure is wrong and every extracted quote verified**: 17 exact, 4 acceptable
  alternatives, 2 correctly empty, 1 missing.
- The **gold page among the candidates** column measures agreement with the labeller's page,
  not success: net profit is 6/6 exact with the income statement among the candidates every
  time, FTE is right or acceptable with the labeller's page found only twice.

**Why the alternatives and the miss**

- **ABN AMRO FTE, both years**: the FTE table (p. 46 and 47) was not among the candidates
  without a reranker, so the model took the **headcount** from the workforce pages (23,299
  and 24,632) and said so. With the local profile the same table was found and the figure
  exact.
- **CM.com 2025 FTE and revenue**: the exact figures sit on the key-figures infographic
  (p. 2) and the income statement (p. 100), neither retrieved; the model used the
  workforce page's 648 FTE and the rounded **EUR 259.4 million** from p. 73.
- **CM.com 2024 revenue**: the income statement (p. 153) was not retrieved and the model
  **returned nothing rather than guess** (with the local profile it took the rounded
  274.2 million from the key-figures page).

**Sustainability goals**

| Report | Gold goals (core) | Extracted | Recall | Core recall | Precision | Gold pages among candidates | Cost |
|---|---|---|---|---|---|---|---|
| ABN AMRO 2025 | 22 (8) | 22 | 0.77 | 0.63 | 0.50 | 0.77 | $0.04 |
| ABN AMRO 2024 | 24 (7) | 24 | 0.79 | 0.71 | 0.96 | 0.54 | $0.07 |
| ASML 2025 | 21 (8) | 14 | 0.48 | 0.88 | 0.93 | 0.43 | $0.06 |
| ASML 2024 | 30 (8) | 28 | 0.80 | 0.88 | 0.96 | 0.47 | $0.06 |
| CM.com 2025 | 10 (5) | 8 | 0.70 | 0.60 | 1.00 | 1.00 | $0.03 |
| CM.com 2024 | 14 (5) | 8 | 0.43 | 0.40 | 0.75 | 0.93 | $0.04 |
| **All** | 121 (41) | 104 | **0.66** | **0.68** | **0.85** | 0.69 | $0.30 |

Each report: **6 queries, 32 chunks, one model call** (about 17k input and 4.5k output
tokens, 24 to 42 s).

- **Recall 0.66**, up from 0.56 with Claude: gpt-5 lists more of what is in front of it
  (104 items for 121 gold goals, against 87), from the same chunks (gold pages among the
  candidates 0.69, about the same as before). **Retrieval still caps recall**: 31% of the
  gold goals were on no candidate page, and all of ASML's misses are of this kind.
- **Precision 0.85**, down from 0.97, for two reasons:
  - **ABN AMRO 2025, 0.50**: the judge accepted all 22 items as goals, so 11 quotes
    failed verification in that call. The ingestion run of the same report, minutes
    earlier, wrote 23 goals with every quote verified; **verbatim quoting varies from call
    to call** with gpt-5 (see the end-to-end section for what the failed quotes look like).
  - **CM.com 2024, 0.75**: two items the judge did not accept as goals ("reduce the
    proportion of flights in business travel", "energy efficiency actions"), policy
    statements without a target.
- **Cost $0.61 for six reports** ($0.10 per report), against $0.99 with Claude.

**Improvements**

1. **Full-page expansion for goals**: use the retrieved chunks only to find the
   sustainability pages, then send those whole pages to the model, so the targets next to a
   found chunk are read too.
2. **A reranker for candidate retrieval** (see Retrieval): the FTE table and the income
   statement are the kind of chunk the fused order misses and a cross-encoder finds.
3. **Read infographics with a vision model**: CM.com's key figures come out as numbers
   without labels; a vision call on such pages fixes the CM.com extraction misses and one
   chat failure.

## Ingestion

**What is evaluated**: ingestion from scratch.

- The six PDFs go into an **empty database through the app's own stages**, a timer around
  each stage, every LLM call's and every embedding request's tokens and cost recorded.
- Measured: **seconds per stage, LLM and embedding usage, metadata against the corpus,
  chunks, datapoints written, and whether the report reached `ready`**.
- One report at a time; parsing on the laptop, everything else hosted.

| Report | Pages | Chunks (tables) | Total | Parse | Metadata | Chunk + store | Embed | Extract | LLM tokens in / out | LLM cost | Embedding tokens | Embedding cost | Outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ABN AMRO 2025 | 434 | 928 (197) | 91 s | 36 s | 2.6 s | 0.1 s | 6.7 s | 46 s | 40.2k / 11.3k | $0.16 | 361k | $0.05 | ready |
| ABN AMRO 2024 | 508 | 1,134 (215) | 96 s | 44 s | 1.7 s | 0.1 s | 7.1 s | 44 s | 38.6k / 10.8k | $0.16 | 450k | $0.06 | ready |
| ASML 2025 | 350 | 838 (158) | 79 s | 42 s | 1.5 s | 0.1 s | 5.0 s | 30 s | 38.4k / 8.3k | $0.13 | 307k | $0.04 | ready |
| ASML 2024 | 410 | 960 (231) | 94 s | 45 s | 2.4 s | 0.1 s | 6.0 s | 41 s | 35.9k / 9.1k | $0.13 | 330k | $0.04 | ready |
| CM.com 2025 | 159 | 343 (81) | 47 s | 13 s | 1.8 s | 0.0 s | 2.5 s | 30 s | 34.6k / 8.0k | $0.12 | 114k | $0.01 | ready |
| CM.com 2024 | 212 | 422 (69) | 59 s | 15 s | 1.5 s | 0.0 s | 3.4 s | 40 s | 33.2k / 7.9k | $0.12 | 142k | $0.02 | ready |
| **Total** | 2,073 | 4,625 (951) | **7.8 min** | 194 s (42%) | 12 s (2%) | 0.4 s | 31 s (7%) | 230 s (49%) | 221k / 55k | $0.82 | 1.70M | $0.22 | 6 of 6 ready |

- **Company and fiscal year detected for 6/6 reports** (one gpt-5-mini call over the first
  three pages).
- **Chunks are identical** to the local run (same parser and chunker); only the vectors
  differ.
- Datapoints written (what the Extracted data page shows):

| Report | FTE | Revenue | Net profit | Dividend per share | Sustainability goals |
|---|---|---|---|---|---|
| ABN AMRO 2025 | 1 | 1 | 1 | 1 | 23 |
| ABN AMRO 2024 | 1 | 1 | 1 | 1 | 24 |
| ASML 2025 | 1 | 1 | 1 | 1 | 17 |
| ASML 2024 | 1 | 1 | 1 | 1 | 28 |
| CM.com 2025 | 1 | 1 | 1 | 0 (pays none, correct) | 7 |
| CM.com 2024 | 1 | 0 (not found) | 1 | 0 (pays none, correct) | 11 |
| **Total** | 6 | 5 | 6 | 4 | 110 |

**Where the time goes**

- **Extraction, 49%** (30 to 46 s per report): almost all of it waiting for gpt-5. The six
  calls run concurrently and add up to 60 to 94 s of model time per report; candidate
  retrieval is now a few seconds (18 queries, no reranker pairs). Also **79% of the cost**
  ($0.12 to $0.16 per report).
- **Parsing, 42%** (13 to 45 s): PyMuPDF text and table detection, unchanged in absolute
  terms and now the largest local cost.
- **Embedding, 7%** (2.5 to 7 s): 24 to 36 requests of 64 chunks, four in flight, about
  150 chunks per second against 14 with the local model; $0.01 to $0.06 per report.

**Against the local profile**: 7.8 minutes instead of 19.0 for the same six reports, at
the same total cost ($1.04 against $1.01: the embeddings cost $0.22 and gpt-5 is cheaper
than Sonnet, although it writes more output tokens, reasoning included).

**Improvements**

1. **Parse pages in parallel**: parsing is now the largest fixed cost; PyMuPDF releases
   the GIL poorly, so a process pool over page ranges, or skipping table detection on
   pages without ruled lines, is the fix.
2. **Strip lines repeated on 20% or more of a report's pages before chunking**: removes
   CM.com's navigation bar from half its chunks; needs a re-ingest.
3. **Read infographic pages with a vision model** when a page has little text and many
   numbers: fixes the CM.com extraction misses and one chat failure.

## Retrieval

**What is evaluated**: the chat's own retrieval, live.

- The small model (`gpt-5-mini`) rewrites each question into **a standalone question and
  one sub-query per company, year and topic**.
- Each sub-query runs **vector + full-text search over all six reports**, **60 candidates
  each**, fused with Reciprocal Rank Fusion; **the top 8 of the fused order are kept**, no
  reranker and no score floor; the sub-queries' chunks are merged.

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
| **All answerable** | 46 | **0.717** | **0.407** | **0.652** | **0.609** | 1.000 |
| Factoid | 30 | 0.633 | 0.395 | 0.667 | 0.533 | 1.000 |
| Comparison | 8 | 1.000 | 0.473 | 0.875 | 0.856 | 1.000 |
| Multi-metric | 5 | 1.000 | 0.590 | 0.200 | 0.509 | 1.000 |
| Follow-up | 3 | 0.333 | 0.042 | 0.667 | 0.875 | 1.000 |
| Text questions | 26 | 0.731 | 0.396 | 0.846 | 0.645 | 1.000 |
| Table questions | 15 | 0.600 | 0.363 | 0.467 | 0.579 | 1.000 |

- **First gold chunk at rank 1 for 12 questions**, rank 2 to 3 for 9, rank 4 to 8 for 12;
  **13 misses**.

| Sub-queries per question | Questions | Chunks retrieved | Median seconds |
|---|---|---|---|
| 1 | 42 | 8 | 0.40 |
| 2 | 10 | 9 to 16 | 0.71 |
| 3 | 4 | 18 to 23 | 1.18 |
| 4 | 1 | 29 | 1.53 |

- **Median 0.42 s, p90 0.88 s**: one hosted embedding call and two SQL searches per
  sub-query, against 4.8 s with the reranker on a laptop GPU.

**Why the fused order is weak without a reranker**

- With RRF (k = 60) a chunk at rank r in one search scores 1/(60 + r) and a chunk in both
  searches about twice that. The two searches rarely agree, so most top-8 lists are chunks
  from one search only, in strict alternation: **ranks 1, 3, 5 and 7 go to the vector
  ranking, ranks 2, 4, 6 and 8 to the full-text ranking**. Full-text search matches any
  word of a whole sentence, ranked by `ts_rank_cd`, and its top four rarely hold the
  answer. The reranker used to read all 60 and put the right chunk first.
- **No score floor**: every question, answerable or not, gets 8 chunks (the abstention gate
  in the pipeline never fires); the model has to recognise an unanswerable question from
  the chunks alone, which it did for all 11.
- **Year leakage is worse**: 41 of 46 questions got chunks from the other year of the same
  company (report precision 0.61 against 0.79 with the reranker, which demoted them).

**Comparisons still work**: all 8 retrieved both sides, one sub-query per year or company.

**Ablations** (same database, same cached rewrites):

| Run | hit@k | MRR | facts | report precision | Median seconds |
|---|---|---|---|---|---|
| **Default: hybrid, no reranker** | 0.717 | 0.407 | 0.652 | 0.609 | 0.42 |
| Vector search alone, no reranker | 0.739 | 0.426 | 0.761 | 0.641 | 0.32 |
| Hybrid + the local reranker (`bge-reranker-v2-m3`) on the same hosted embeddings | **0.913** | **0.597** | **0.913** | 0.778 | 4.39 |
| Local profile, 2026-10-18 (bge-m3 + reranker, Haiku rewrites) | 0.957 | 0.608 | 0.935 | 0.794 | 4.81 |

- **The embedding model is not the problem**: hosted embeddings with the reranker land
  within two questions of the local profile. **The missing reranker is**: it recovers 9 of
  the 13 misses, and 22 questions get the gold chunk at rank 1 instead of 12.
- **Dropping full-text search** gains one question and 11 points of facts, not enough on
  its own.
- The four misses that remain with the reranker: ASML's EUV system count (a sales table
  whose row does not say "EUV"), ASML's backlog (the 2024 report's backlog page outranks
  the 2025 one) and China share (the segment table on p. 67 is never a candidate), ABN
  AMRO's internal FTE 2024 (the workforce pages next to the labeller's, from both years,
  fill the eight slots).

**Improvements**

1. **A reranker in the default profile.** The measured gain is 20 points of hit@k and 26
   points of facts. Options, in order of effort: a hosted rerank API (no local model, a
   second key); an LLM rerank with `gpt-5-mini` over the 60 candidates (one cheap call per
   sub-query); the local cross-encoder whenever the `local-models` group is installed
   (what the ablation ran, 4 s per question on a laptop GPU).
2. **Until then, tune the fused order**: vector search alone or a lower weight for
   full-text in RRF, and **12 to 16 chunks per sub-query**, which is cheap without a
   reranker and gives the model the rank 9 to 16 chunks.
3. **Filter each sub-query by its company and fiscal year**: the rewrite already outputs
   both; a report filter should take report precision from 0.61 to about 1.0 and free 40%
   of the context on year-specific questions.

## End to end

**What is evaluated**: the whole chat pipeline (`run_chat`).

- Rewrite and decomposition, retrieval per sub-query, the structured answer with the
  conversation as earlier turns, quote verification.
- Each gold question asked **once, one at a time** (honest latency), with its history for
  follow-ups and no report filter.

**How answers are scored.** Two layers:

| Layer | What it checks | Needs |
|---|---|---|
| Deterministic | Status (answered, partial, not found); every citation's verification (exact, fuzzy, failed); whether a verified citation is on a gold page; whether any retrieved chunk is on a gold page (separates retrieval failures from generation failures); latency per stage; tokens and cost | Code only |
| LLM judge | Correctness against the gold answer and groundedness of the figures in the cited quotes | One `claude-opus-5-5` call per question |

**LLM as a judge**: `claude-opus-5-5`, a different and stronger model than the generator,
one call per question, not counted in the cost.

- **Sees**: the question (and the conversation for a follow-up), the gold answer and the
  labeller's notes naming the traps, the system's status and answer, the cited quotes with
  their verification status. **Not** the chunks or the report.
- **Returns**: `correctness` (correct, partial, incorrect, abstained), `grounded`, an
  explanation.
- **Rubric**: correct allows rounding, unit rewrites and extra correct detail; incorrect is
  a wrong figure or fact; for an unanswerable question, correct means it said the figure is
  not disclosed. Grounded means every figure in the answer is in a cited quote.
- Every verdict other than correct, and every ungrounded one, **was read by hand**.

| Metric | Result |
|---|---|
| Correct (answerable, 46) | **65.2%** (30) |
| Correct or partial | 84.8% (39) |
| Incorrect | **6.5%** (3) |
| Abstained (judge) | 8.7% (4), of which 2 with status "not found" |
| Unanswerable handled correctly (11) | **100%**; 6 "not found", 5 "partial" that clearly say the figure is not disclosed |
| Hallucinated on unanswerable | 0% |
| Retrieval reached a gold page | 69.6% |
| A verified citation on a gold page | 52.2% |
| Citations per answer | 1.63 (93 in all); 6 answers cite nothing, all abstentions |
| Citations verified verbatim / fuzzy / failed | **90.3%** / 0% / **9.7%** (84 / 0 / 9) |
| Grounded (every figure in a cited quote) | 87.7% |
| Latency p50 / p90 | 9.0 s / 17.3 s (rewrite 2.5 s, retrieval 0.4 s, generation 5.6 s at p50) |
| Cost | $0.81 for 57 questions, $0.014 each (336k input tokens, 51k output; the judge not counted) |

| Type | n | Correct | Correct or partial | Median latency |
|---|---|---|---|---|
| Factoid | 30 | 67% | 80% | 8.1 s |
| Comparison | 8 | 88% | 100% | 10.8 s |
| Multi-metric | 5 | 40% | 100% | 18.6 s |
| Follow-up | 3 | 33% | 67% | 10.7 s |
| Unanswerable | 11 | 100% | | 6.1 s |
| Text questions | 26 | 81% | 88% | |
| Table questions | 15 | 47% | 73% | |

**What fails**: 16 answerable questions were not judged correct.

| Cause | Questions | What happened |
|---|---|---|
| **Retrieval**: no gold page among the chunks | 9 | ABN AMRO's renewable financing, women in senior management and internal FTE, ASML's EUV count, backlog and China share, CM.com's adjusted EBITDA, offices and net result follow-up. In 3 the model said what it could not find; in 3 it answered from another page and came close (China's share computed from the segment table, the net result from the income statement without its unit, the EUV split without the total); in 3 it answered wrongly (below) |
| **Verification**: the right figure, a quote that failed | 2 | ASML's 2024 FTE (44,027) and 2024 lithography units (418) were retrieved and answered, but the quotes were stitched from table cells, failed verification, and the pipeline withheld the answers as "not stated" |
| **Generation**: gold page retrieved, figure missed | 5 | CM.com's market-based emissions (the table chunk was in the context, the model abstained); three multi-metric questions each missing one or two of three figures (the EBITDA trap, the EUV total, remaining performance obligations given as backlog); headcount instead of FTE in the cross-company comparison |

| Question | Verdict | What happened | Cause |
|---|---|---|---|
| ABN AMRO internal FTE at end-2024 (gold 21,976) | Incorrect | 19,916, the 2025 report's restated 2024 figure excluding an acquisition | Year leakage: only 2025 chunks were retrieved; the quote also failed verification |
| CM.com adjusted EBITDA 2025 (gold EUR 19.8 million) | Incorrect | Says adjusted EBITDA is not stated and gives reported EBITDA, EUR 18.4 million | Retrieval: the page with the adjusted figure was not retrieved; the labeller's trap |
| CM.com offices and countries 2024 (gold 23 offices in 15 countries) | Incorrect | "15 offices in 24 countries" | The key-figures infographic parses to numbers without labels ("15 1 1 24 1"); the model guessed the labels |
| CM.com net result 2024, follow-up (gold EUR -19.8 million) | Partial | "a net loss of EUR 19,813": the right figure without its "thousand" | Unit dropped from a table in thousands |

- **The failed quotes** (9 of 93, all from tables or key-figure pages) are not invented:
  gpt-5 **assembles a label and a figure from non-adjacent cells**, joined with line breaks
  or "...", for example `Year ended December 31 (in FTE) 2022 2023 2024 ... Total 39,086
  42,416 44,027`, where the page has the heading and the row far apart. Claude copied
  contiguous rows (0 failures of 106). The verifier is right to reject them; the cost is
  two correct answers withheld and three correct answers shown with a failed badge.
- **No hallucination**: the 11 unanswerable questions were handled, and in 6 of the 9
  retrieval failures the model either said what it could not find or answered from
  another page of the right report.
- **Groundedness 87.7%**: 7 ungrounded answers, 3 of them correct (a side figure read in a
  chunk but not quoted).
- **Latency 9.0 s**, 1.5 s more than the local profile although retrieval is 3.4 s faster:
  gpt-5-mini's rewrite (2.5 s) and gpt-5's generation (5.6 s) are slower than Haiku's and
  Sonnet's. **Cost $0.014 per question**, 20% less.

**Improvements**

1. **The reranker** (Retrieval above): 9 of the 16 failures are retrieval failures, and the
   ablation recovers 9 of the 13 retrieval misses, so it is the one change that moves
   correctness towards the local profile's 91%.
2. **Verify table quotes cell by cell**: when a quote fails, split it on line breaks and
   "..." and verify each segment with a figure on the page; accept as `fuzzy` when all
   segments are found. Saves the two withheld answers. The prompt should also ask for one
   contiguous span, a whole table row rather than a heading plus a cell.
3. **Check figures and units in the prose**: every number in the answer should be in a
   verified quote with its scale word (19,813 against "in thousands"); flag the rest as
   "figure not quoted".
4. **Filter each sub-query by its company and year**: the one incorrect answer that came
   from the wrong year would not have happened.

## What to expect from the local retrieval profile

`RETRIEVAL_MODELS=local` runs `bge-m3` and the `bge-reranker-v2-m3` cross-encoder in the
backend process instead of calling a hosted embedding endpoint. It was evaluated on
2026-10-18 with Claude as the LLM (`claude-sonnet-5-5`, `claude-haiku-4-5`), on the same
corpus and gold set; the full results are in the untagged files under
`backend/eval/results/` and the earlier version of this report is in the git history.

| | Default: gpt-5, hosted embeddings, no reranker | Local profile: claude-sonnet-5-5, bge-m3, bge reranker |
|---|---|---|
| Ingestion, six reports | 7.8 min, $1.04 | 19.0 min, $1.01 (embeddings free, on a laptop GPU; 10 to 15 min per report on CPU in Docker) |
| Extraction, 24 single values | 17 exact, 4 alternative, 2 correctly none, 1 missing, 0 wrong | 19 exact, 2 alternative, 2 correctly none, 1 missing, 0 wrong |
| Goals recall / core recall / precision | 0.66 / 0.68 / 0.85 | 0.56 / 0.73 / 0.97 |
| Retrieval hit@k / MRR / facts | 0.72 / 0.41 / 0.65 in 0.4 s | 0.96 / 0.61 / 0.94 in 4.8 s |
| Answers correct / correct or partial / incorrect | 65% / 85% / 6.5% | 91% / 96% / 0% |
| Unanswerable handled, hallucinations | 11/11, 0 | 11/11, 0 |
| Citations verified | 90% (9 of 93 failed) | 100% (106 of 106) |
| Latency / cost per question | 9.0 s / $0.014 | 7.5 s / $0.018 |

What the local profile buys is **the reranker**: the cross-encoder reads the 60 fused
candidates and puts the right chunk first, its probability gives a calibrated **abstention
floor** (answerable questions score 0.96 or more, out-of-corpus questions below 0.2), and
year leakage is lower. The ablation above shows the same gain with the hosted embeddings,
so the embedding model itself is a wash. What it costs is `torch`, 4.5 GB of models, and a
GPU: on CPU every query waits for 60 reranker pairs and a report takes 10 to 15 minutes to
ingest. The rest of the answer gap is the LLM's handling of tables: Claude copied table
rows verbatim and read them more carefully, where gpt-5 stitched cells and dropped a unit;
on the other hand gpt-5 extracted more goals and cost less.

## Limits of this evaluation

- **57 questions over three companies**: enough to rank changes with large effects, not to
  separate ones within about 5 points; 8 comparisons and 5 multi-metric questions are thin.
- **The two profiles differ in more than one component** (LLM, embeddings, reranker,
  rewrite model, and a day between the runs). Only the reranker ablation isolates one
  component; the rest of the comparison is between whole stacks.
- **One labeller's choice** of "the" answer (internal vs total FTE, ordinary vs total
  dividend), mitigated by the acceptable alternatives.
- **One LLM judge**, no human agreement study.
- **Latency on one machine and one evening**, hosted endpoints, one question at a time.
