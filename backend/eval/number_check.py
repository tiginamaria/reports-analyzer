"""Prototype of a deterministic groundedness check: does every number in an answer appear in
one of its verified quotes? Scored against the LLM judge's `grounded` verdict on saved e2e
results, to see whether a cheap check in code could flag unsupported figures at answer time.

    uv run python -m eval.number_check baseline tuned
"""

import argparse
import json
import re

from eval.common import RESULTS_DIR, write_json

# 1,234.5 / 15.4% / 9,609.4m; a number preceded by a letter (Q4, CO2e, Scope3) is skipped.
# A rescaled figure (2,519 million written as 2.5 billion) is flagged: strict on purpose.
NUMBER = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)*")
# years and small ordinals are context, not figures that need support
IGNORED = re.compile(r"^(19|20)\d\d$|^\d$")


def canonical(number: str) -> str:
    """'2,519.0' and '2519' compare equal; trailing '.0' and separators are dropped."""
    digits = number.replace(",", "")
    if "." in digits:
        digits = digits.rstrip("0").rstrip(".")
    return digits


def figures(text: str) -> set[str]:
    return {canonical(n) for n in NUMBER.findall(text) if not IGNORED.match(canonical(n))}


def unsupported(answer: str, quotes: list[str]) -> set[str]:
    supported = set().union(*(figures(q) for q in quotes)) if quotes else set()
    return figures(answer) - supported


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    args = parser.parse_args()
    out = {}
    for name in args.configs:
        rows = json.loads((RESULTS_DIR / f"e2e_{name}.json").read_text())["rows"]
        rows = [row for row in rows if "cited" in row]
        flagged, agree, judge_ungrounded, caught = [], 0, 0, 0
        for row in rows:
            quotes = [
                c["quote"] for c in row["cited"] if c["verification"] in ("verified", "fuzzy")
            ]
            missing = unsupported(row["answer"] or "", quotes)
            grounded = row["judge"]["grounded"]
            agree += (not missing) == grounded
            judge_ungrounded += not grounded
            caught += (not grounded) and bool(missing)
            if missing:
                flagged.append({"id": row["id"], "unsupported": sorted(missing)})
        out[name] = {
            "answers": len(rows),
            "flagged": len(flagged),
            "agreement_with_judge": round(agree / max(len(rows), 1), 3),
            "judge_ungrounded": judge_ungrounded,
            "judge_ungrounded_caught": caught,
            "examples": flagged[:10],
        }
        print(name, {k: v for k, v in out[name].items() if k != "examples"})
    print(write_json("number_check.json", out))


if __name__ == "__main__":
    main()
