"""Parse numbers as they appear in annual reports ("23,500", "1.2 million", "€ 3.4bn")."""

from __future__ import annotations

import re

_MULTIPLIERS = {
    "thousand": 1e3,
    "k": 1e3,
    "million": 1e6,
    "mln": 1e6,
    "mn": 1e6,
    "m": 1e6,
    "billion": 1e9,
    "bln": 1e9,
    "bn": 1e9,
    "b": 1e9,
    "trillion": 1e12,
    "tn": 1e12,
}

# Digit groups of three separated by ",", "." or a space, e.g. "23,500" or "1 234 567".
_GROUPED = r"\d{1,3}(?:[,.\s]\d{3})+"

_NUMBER_RE = re.compile(
    rf"(?P<sign>[-−])?\s*(?P<num>{_GROUPED}(?:[.,]\d+)?|\d+(?:[.,]\d+)?)"
    r"\s*(?P<mult>thousand|million|mln|mn|billion|bln|bn|trillion|tn|[kKmMbB])?\b",
    re.IGNORECASE,
)
_GROUPED_RE = re.compile(_GROUPED)
_GROUPED_WITH_DECIMALS_RE = re.compile(rf"{_GROUPED}[.,]\d+")
_SEPARATORS_RE = re.compile(r"[,.\s]")


def parse_number(text: str | None) -> float | None:
    """Return the first number in `text` as a float, honouring thousands separators,
    magnitude words and the accounting notation for negatives, "(3,786)". Returns None when
    nothing numeric is found."""
    if not text:
        return None
    match = _NUMBER_RE.search(text)
    if not match:
        return None
    number = _to_float(match.group("num"))
    if match.group("mult"):
        number *= _MULTIPLIERS[match.group("mult").lower()]
    if match.group("sign") or _in_parentheses(text, match):
        number = -number
    return number


def _in_parentheses(text: str, match: re.Match[str]) -> bool:
    """Whether the matched number is wrapped like "(3,786)", a loss in financial tables."""
    before = text[: match.start()].rstrip()
    after = text[match.end() :].lstrip()
    return before.endswith("(") and after.startswith(")")


def _to_float(raw: str) -> float:
    """Decide which separators are thousands groups and which one is the decimal mark:
    "23.500" and "1,234,567" are grouped, "3,4" is a decimal comma, "1,234.56" is both."""
    if _GROUPED_RE.fullmatch(raw):
        return float(_SEPARATORS_RE.sub("", raw))
    if _GROUPED_WITH_DECIMALS_RE.fullmatch(raw):
        head, decimals = re.split(r"[.,](?=\d+$)", raw)
        return float(_SEPARATORS_RE.sub("", head) + "." + decimals)
    return float(raw.replace(",", "."))
