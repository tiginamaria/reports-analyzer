"""Text normalisation shared by the parser, the verifier and retrieval."""

import re
import unicodedata

# Typographic characters that PDFs use and that models do not reproduce verbatim.
_TYPOGRAPHY = {
    "‘": "'",  # left single quote
    "’": "'",  # right single quote
    "‚": "'",  # low single quote
    "“": '"',  # left double quote
    "”": '"',  # right double quote
    "„": '"',  # low double quote
    "–": "-",  # en dash
    "—": "-",  # em dash
    "−": "-",  # minus sign
    " ": " ",  # non-breaking space
    " ": " ",  # thin space
    " ": " ",  # narrow non-breaking space
    "ﬁ": "fi",  # ligature
    "ﬂ": "fl",  # ligature
    "­": "",  # soft hyphen
}


def normalize(text: str) -> str:
    """Lower-case, unify quotes and dashes, collapse whitespace. Used for verbatim matching."""
    text = unicodedata.normalize("NFKC", text)
    for src, dst in _TYPOGRAPHY.items():
        text = text.replace(src, dst)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def clean_page_text(text: str) -> str:
    """Light cleanup of raw PDF text: join words hyphenated across lines, collapse blank lines."""
    text = text.replace("\r", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(text.split())
