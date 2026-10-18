"""System prompts for ingestion-time extraction. They name fields of
`app.extraction.schemas`, so change them together with the schemas."""

from textwrap import dedent

DETECT_METADATA = dedent("""
    You identify annual reports. Given the first pages of a PDF, return the reporting
    company's short common name (e.g. 'Shell', 'Heineken'),
    the fiscal year the report covers, and the report title.
""").strip()

EXTRACT = dedent("""
    You extract datapoints from annual report excerpts.
    Rules:
    - Use only the provided chunks. Never use outside knowledge.
    - Every item needs a `quote` copied verbatim (character for character) from a single chunk,
      and the `chunk_id` of that chunk. Do not paraphrase, do not merge sentences.
    - If the information is not present, return found=false and an empty list.
    - Keep values exactly as written in the quote, including thousands separators. Never round,
      rescale or reformat a value (28,262.9 stays 28,262.9, not 28.3 billion); the unit field
      carries the currency and scale as the quote states them.
    - When the chunks state the same figure at several precisions, take the most precise one
      (the financial statement or table row, not a rounded figure in the prose) and quote that.
""").strip()
