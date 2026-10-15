"""Parser quality against hand-labelled pages from the five real annual reports.

Text present and labelled table cells recovered must hold. False tables and reading order
are held at a floor so regressions fail; raise the floors when the parser improves. Run
`uv run python -m tests.parsing_quality` for the full scorecard.
"""

import pytest

from tests.parsing_quality import Scorecard, print_scorecard, run

# Floors for the heuristic categories, set from the measured baseline.
FLOORS = {
    "text": 1.0,
    "table_cells": 1.0,
    "no_false_tables": 1.0,
    "reading_order": 1.0,
    "table_not_in_body": 0.9,
}


@pytest.fixture(scope="module")
def card() -> Scorecard:
    card = run()
    print_scorecard(card)
    return card


@pytest.mark.parametrize("category", sorted(FLOORS))
def test_parser_quality_floor(card: Scorecard, category: str):
    rate = card.rate(category)
    failures = "\n".join(f"  {c.file} p{c.page}: {c.detail}" for c in card.failures(category))
    assert rate >= FLOORS[category], f"{category} {rate:.2f} < {FLOORS[category]}\n{failures}"
