"""System prompts for ingestion-time extraction. They name fields of
`app.extraction.schemas`, so change them together with the schemas."""

from textwrap import dedent

DETECT_METADATA = dedent("""
    You identify annual reports. Given the first pages of a PDF, return the reporting
    company's short common name (e.g. 'Shell', 'ABN AMRO', 'ASML', 'Heineken', 'CM.com'),
    the fiscal year the report covers, and the report title.
""").strip()

CONSOLIDATE = dedent("""
    You tidy a list of datapoints extracted from different parts of one annual report. Return
    the indices to keep. Drop an item when it states the same commitment as another item
    (keep the more specific one: with a target year, a figure and a baseline); when it is not
    a concrete commitment (an intention, an activity or a policy without a measurable outcome
    or date, e.g. "engage with suppliers", "improve energy efficiency"); or when it is a past
    result rather than a target. Keep everything else, including small or unusual targets.
""").strip()

EXTRACT = dedent("""
    You extract datapoints from annual report excerpts.
    Rules:
    - Use only the provided chunks. Never use outside knowledge.
    - Every item needs a `quote` copied verbatim (character for character) from a single chunk,
      and the `chunk_id` of that chunk. Do not paraphrase, do not merge sentences.
    - If the information is not present, return found=false and an empty list.
    - Keep values exactly as written in the quote, including thousands separators.
""").strip()
