import pytest

from app.extraction.numbers import parse_number


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("23,500", 23500),  # thousands comma
        ("23.500", 23500),  # thousands dot (Dutch reports)
        ("3,4", 3.4),  # decimal comma
        ("1,234.56", 1234.56),  # grouped with decimals
        ("1.2 million", 1_200_000),
        ("€ 3.4bn", 3_400_000_000),
        ("99.5%", 99.5),
        ("−12.3", -12.3),  # unicode minus
        ("net zero", None),
        (None, None),
    ],
)
def test_parse_number(text, expected):
    assert parse_number(text) == expected
