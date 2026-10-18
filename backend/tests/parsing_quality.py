"""Score the PDF parser against the hand-labelled fixture pages in tests/fixtures/parsing.yaml.

Shared by the pytest test (thresholds) and the `python -m tests.parsing_quality` scorecard.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from app.ingestion.parser import ParsedDocument, parse_pdf
from app.textutil import normalize

FIXTURES = Path(__file__).parent / "fixtures"


@dataclass
class Check:
    file: str
    page: int
    category: str
    detail: str
    ok: bool


@dataclass
class Scorecard:
    checks: list[Check] = field(default_factory=list)

    def rate(self, category: str) -> float:
        rows = [c for c in self.checks if c.category == category]
        return sum(c.ok for c in rows) / len(rows) if rows else 1.0

    def failures(self, category: str | None = None) -> list[Check]:
        return [c for c in self.checks if not c.ok and (category is None or c.category == category)]

    @property
    def categories(self) -> list[str]:
        return sorted({c.category for c in self.checks})


def page_text_outside_tables(page_text: str, tables: list[str]) -> str:
    """Page text with each table's lines removed, to tell prose mentions from table rows."""
    table_lines = {line.strip() for t in tables for line in t.split("\n") if line.strip()}
    return "\n".join(line for line in page_text.split("\n") if line.strip() not in table_lines)


def _contains(haystack: str, needle: str) -> bool:
    return normalize(needle) in normalize(haystack)


def score_document(spec: dict, doc: ParsedDocument, card: Scorecard) -> None:
    f = spec["file"]
    for p in spec["pages"]:
        n = p["page"]
        page_text = doc.pages[n - 1]
        tables = [t.text for t in doc.tables if t.page == n]

        def add(category: str, detail: str, ok: bool, page_no: int = n) -> None:
            card.checks.append(Check(f, page_no, category, detail, ok))

        for s in p.get("text_contains", []):
            add("text", s, _contains(page_text, s))
        body = doc.bodies[n - 1]  # what text windows are built from
        outside_tables = page_text_outside_tables(page_text, tables)
        for s in p.get("table_cells", []):
            add("table_cells", s, any(_contains(t, s) for t in tables))
            # The body must not repeat table content, unless the prose also mentions it.
            add("table_not_in_body", s, not _contains(body, s) or _contains(outside_tables, s))
        if p.get("no_tables"):
            add("no_false_tables", f"{len(tables)} tables", not tables)
        for earlier, later in p.get("order", []):
            text = normalize(page_text)
            a, b = text.find(normalize(earlier)), text.find(normalize(later))
            add("reading_order", f"{earlier!r} before {later!r}", a >= 0 and b >= 0 and a < b)


def run() -> Scorecard:
    card = Scorecard()
    for spec in yaml.safe_load((FIXTURES / "parsing.yaml").read_text()):
        doc = parse_pdf(FIXTURES / "pdf" / spec["file"])
        score_document(spec, doc, card)
    return card


def print_scorecard(card: Scorecard) -> None:
    print(f"{'category':<20} {'pass':>5} {'total':>6} {'rate':>6}")
    for cat in card.categories:
        rows = [c for c in card.checks if c.category == cat]
        print(f"{cat:<20} {sum(c.ok for c in rows):>5} {len(rows):>6} {card.rate(cat):>6.2f}")
    if card.failures():
        print("\nfailures:")
        for c in card.failures():
            print(f"  {c.file:<22} p{c.page:<3} {c.category:<18} {c.detail}")


if __name__ == "__main__":
    print_scorecard(run())
