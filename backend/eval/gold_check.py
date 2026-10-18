"""Validate the gold files against the ingested page text.

Every evidence quote must be found on its page in `pages.text` (the app's source of truth),
after the same normalisation the verifier uses. Labels were drafted with LLM help, so this
is what keeps an invented quote or a wrong page number out of the gold set.

    uv run python -m eval.gold_check
"""

from __future__ import annotations

import sys
from collections import Counter

from app.db import SessionLocal
from app.models import Page
from app.textutil import normalize
from eval.common import load_corpus, load_yaml, report_keys

QA_TYPES = {"factoid", "comparison", "multimetric", "followup", "unanswerable"}


def main() -> int:
    qa = load_yaml("qa.yaml")["qa"]
    extraction = load_yaml("extraction.yaml")
    keys = {doc.key for doc in load_corpus()}
    errors: list[str] = []
    with SessionLocal() as db:
        ids = report_keys(db)
        pages = {
            (page.report_id, page.page_no): normalize(page.text) for page in db.query(Page).all()
        }

    def check(where: str, report: str, page: int, quote: str) -> None:
        if report not in ids:
            errors.append(f"{where}: unknown or not ingested report {report!r}")
        elif normalize(quote) not in pages.get((ids[report], page), ""):
            found = [
                p for (r, p), text in pages.items() if r == ids[report] and normalize(quote) in text
            ]
            errors.append(
                f"{where}: quote not on {report} p{page} (found on {found}): {quote[:70]!r}"
            )

    for item_id, n in Counter(item["id"] for item in qa).items():
        if n > 1:
            errors.append(f"duplicate qa id {item_id}")
    for item in qa:
        if item["type"] not in QA_TYPES:
            errors.append(f"{item['id']}: unknown type {item['type']}")
        if not set(item["reports"]) <= keys:
            errors.append(f"{item['id']}: unknown report in {item['reports']}")
        if item["type"] != "unanswerable" and not item["evidence"]:
            errors.append(f"{item['id']}: answerable without evidence")
        for e in item["evidence"]:
            check(item["id"], e["report"], e["page"], e["quote"])
        context = " ".join(normalize(e["quote"]) for e in item["evidence"])
        for fact in item.get("key_facts", []):
            if item["evidence"] and normalize(fact) not in context:
                errors.append(f"{item['id']}: key fact {fact!r} not in any evidence quote")
    for fte in extraction["fte"]:
        check(f"fte {fte['report']}", fte["report"], fte["page"], fte["quote"])
    for entry in extraction["sustainability_goals"]:
        for goal in entry["goals"]:
            check(
                f"goal {entry['report']} {goal['label']}",
                entry["report"],
                goal["page"],
                goal["quote"],
            )

    by_type = Counter(item["type"] for item in qa)
    goals = sum(len(e["goals"]) for e in extraction["sustainability_goals"])
    print(f"{len(qa)} questions {dict(by_type)}, {len(extraction['fte'])} FTE, {goals} goals")
    for error in errors:
        print("ERROR", error)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
