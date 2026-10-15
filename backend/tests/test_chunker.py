from app.ingestion.chunker import chunk_document
from app.ingestion.parser import ParsedDocument, Table


def words(n: int, token: str = "w") -> str:
    return " ".join(f"{token}{i}" for i in range(n))


def document(pages: list[str], tables: list[Table] | None = None) -> ParsedDocument:
    return ParsedDocument(pages=pages, bodies=pages, tables=tables or [])


def test_page_is_cut_into_overlapping_windows():
    doc = document([words(1000)])
    chunks = chunk_document(doc, max_words=400, overlap_words=100, min_words=5)
    assert [c.word_count for c in chunks] == [400, 400, 400]
    assert chunks[0].content.split()[-100:] == chunks[1].content.split()[:100]  # overlap
    assert chunks[-1].content.endswith("w999")
    assert all(c.page == 1 for c in chunks)


def test_short_page_is_one_chunk_and_tiny_pages_are_dropped():
    doc = document([words(50), "a b", words(10, "x")])
    chunks = chunk_document(doc, max_words=400, overlap_words=60, min_words=5)
    assert [(c.page, c.word_count) for c in chunks] == [(1, 50), (3, 10)]


def test_tables_are_separate_chunks_on_their_page():
    table = "| a | b |\n| --- | --- |\n" + "\n".join(f"| {i} | {i * 2} |" for i in range(30))
    doc = document([words(50), words(50)], [Table(page=2, text=table)])
    chunks = chunk_document(doc, max_words=400, overlap_words=60, min_words=5)
    assert [(c.page, c.kind) for c in chunks] == [(1, "text"), (2, "text"), (2, "table")]
    assert chunks[2].content == table
