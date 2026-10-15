import pytest

from app.verify import verify_quote

PAGE = (
    "At year-end 2025 the Group employed 23,500 FTEs (2024: 22,900), of which 60% in the "
    "Netherlands.\nOur ambition is to reach net-zero emissions by 2050 – across all scopes."
)


@pytest.mark.parametrize(
    ("quote", "status"),
    [
        ("the Group employed 23,500 FTEs (2024: 22,900)", "verified"),
        ("net-zero  emissions by 2050 — across all scopes", "verified"),  # whitespace, dashes
        ("the Group employed 23,500 FTE (2024: 22,900), of which 60% in the", "fuzzy"),
        ("The Group spent EUR 5 billion on adaptation.", "failed"),
    ],
)
def test_verify_quote(quote: str, status: str):
    assert verify_quote(quote, [PAGE]).status == status


def test_any_source_may_hold_the_quote():
    # A table chunk is text clipped to the table region, so a row may read differently from
    # the page's block text; verification accepts either source.
    table_chunk = "FTEs year-end | 23,500 | 22,900"
    assert verify_quote("23,500 | 22,900", [PAGE, table_chunk]).status == "verified"
    assert verify_quote("invented quote about adaptation", [PAGE, table_chunk]).status == "failed"


def test_too_short_quotes_fail():
    assert verify_quote("a", [PAGE]).status == "failed"
