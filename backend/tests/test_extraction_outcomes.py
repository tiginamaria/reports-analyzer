"""Extraction outcomes: a failed extraction is recorded as failed, never as "nothing found";
the report stays searchable; a retry fills the entry without re-ingesting the PDF.

Needs Postgres (migrated to head) and the embedding model, but no API key: the LLM is
scripted so that chosen registry entries fail on demand.
"""

import os
import re
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel
from sqlalchemy import text

from app.extraction.schemas import ExtractedItem, ExtractionResult, ReportMetadata
from app.llm.base import ChatTurn, LLMError, LLMProvider
from tests.helpers import require_postgres, sync_engine

pytestmark = pytest.mark.integration

FTE_SENTENCE = "At year-end 2025 Test Company employed 1,234 FTEs across all regions."
REGISTRY_KEYS = {"fte", "revenue", "net_income", "dividend_per_share", "sustainability_goal"}


class ScriptedProvider(LLMProvider):
    """Detects the fixture's metadata; for extraction, raises for the entries in `failing`,
    quotes the FTE sentence for `fte` and finds nothing for the other entries."""

    name = "scripted"

    def __init__(self) -> None:
        self.failing: set[str] = set()

    def model_name(self, small: bool = False) -> str:
        return "scripted"

    def _raw_json(
        self,
        *,
        system: str,
        turns: list[ChatTurn],
        schema: type[BaseModel],
        small: bool,
        max_tokens: int,
    ) -> str:
        prompt = turns[-1].content
        if schema is ReportMetadata:
            meta = ReportMetadata(company="Test Company", fiscal_year=2025, title=None)
            return meta.model_dump_json()
        key = _entry_key(prompt)
        if key in self.failing:
            raise LLMError(f"scripted failure for {key}")
        if key != "fte":
            return ExtractionResult(found=False, items=[]).model_dump_json()
        chunk_id = re.search(r"\[chunk ([^\]]+)\]", prompt).group(1)
        item = ExtractedItem(
            label="FTE",
            value="1,234",
            unit="FTE",
            period="2025",
            qualifiers=[],
            quote=FTE_SENTENCE,
            chunk_id=chunk_id,
        )
        return ExtractionResult(found=True, items=[item]).model_dump_json()


def _entry_key(prompt: str) -> str:
    """Which registry entry an extraction prompt is for: the one whose instructions it quotes."""
    from app.extraction.extractor import load_registry

    return next(spec.key for spec in load_registry() if spec.instructions.strip() in prompt)


PROVIDER = ScriptedProvider()


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["REPORTS_DIR"] = str(tmp_path_factory.mktemp("reports"))
    os.environ["RERANKER_PROVIDER"] = "none"
    from app.config import get_settings

    get_settings.cache_clear()
    require_postgres("select 1 from extractions limit 1")
    from app.ingestion import pipeline

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pipeline, "get_provider", lambda: PROVIDER)
        from app.main import app

        with TestClient(app) as c:
            yield c


def _make_pdf(path: Path, text_: str) -> None:
    import pymupdf

    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), text_, fontsize=11)
    doc.save(path)


def _upload(client: TestClient, pdf: Path) -> str:
    with pdf.open("rb") as f:
        r = client.post("/api/reports", files=[("files", (pdf.name, f, "application/pdf"))])
    assert r.status_code == 201, r.text
    return r.json()["started"][0]["id"]


def _wait_until_settled(client: TestClient, report_id: str) -> dict:
    for _ in range(180):
        report = client.get(f"/api/reports/{report_id}").json()
        if report["status"] in ("ready", "failed"):
            return report
        time.sleep(1)
    pytest.fail(f"report did not settle: {report}")


def _outcomes(report: dict) -> dict[str, dict]:
    return {e["key"]: e for e in report["extractions"]}


def test_failed_entries_are_recorded_and_retried_without_reingesting(client, tmp_path):
    pdf = tmp_path / "test-company-annual-report-2025.pdf"
    _make_pdf(pdf, f"Test Company Annual Report 2025\n\n{FTE_SENTENCE}\n")
    PROVIDER.failing = {"fte", "sustainability_goal"}
    report_id = _upload(client, pdf)
    report = _wait_until_settled(client, report_id)

    # The index is complete, so the report is ready and searchable...
    assert report["status"] == "ready", report
    assert report["error"] is None
    assert report["chunk_count"] >= 1
    from app.db import SessionLocal
    from app.retrieval.search import retrieve

    async def search() -> list:
        async with SessionLocal() as db:
            return await retrieve(db, "number of FTEs", report_ids=[report_id], top_k=3)

    assert client.portal.call(search)  # on the app's loop, where the async pool lives

    # ...and the entries whose call failed say so instead of looking like "nothing found".
    outcomes = _outcomes(report)
    assert set(outcomes) == REGISTRY_KEYS
    assert outcomes["fte"]["status"] == "error"
    assert "scripted failure for fte" in outcomes["fte"]["message"]
    assert outcomes["sustainability_goal"]["status"] == "error"
    assert {k: o["status"] for k, o in outcomes.items() if k not in PROVIDER.failing} == {
        "revenue": "not_found",
        "net_income": "not_found",
        "dividend_per_share": "not_found",
    }
    assert outcomes["revenue"]["message"] is None
    assert client.get("/api/datapoints", params={"report_id": report_id}).json() == []

    # The retry runs the failed entries only, on the existing index.
    chunks_before = report["chunk_count"]
    PROVIDER.failing = set()
    r = client.post(f"/api/reports/{report_id}/extract", json={})
    assert r.status_code == 202, r.text
    pending = {e["key"] for e in r.json()["extractions"] if e["status"] == "pending"}
    assert pending == {"fte", "sustainability_goal"}
    report = _wait_until_settled(client, report_id)
    assert report["status"] == "ready"
    assert report["chunk_count"] == chunks_before, "a retry must not re-embed the report"
    outcomes = _outcomes(report)
    assert outcomes["fte"]["status"] == "extracted"
    assert outcomes["sustainability_goal"]["status"] == "not_found"
    dps = client.get("/api/datapoints", params={"report_id": report_id}).json()
    assert [(d["key"], d["value"]) for d in dps] == [("fte", "1,234")]

    # Extracting an entry again replaces its datapoints instead of adding to them.
    r = client.post(f"/api/reports/{report_id}/extract", json={"keys": ["fte"]})
    assert r.status_code == 202, r.text
    _wait_until_settled(client, report_id)
    dps = client.get("/api/datapoints", params={"report_id": report_id}).json()
    assert [(d["key"], d["value"]) for d in dps] == [("fte", "1,234")]

    # Nothing failed any more; unknown keys are rejected.
    assert client.post(f"/api/reports/{report_id}/extract", json={}).status_code == 409
    r = client.post(f"/api/reports/{report_id}/extract", json={"keys": ["nope"]})
    assert r.status_code == 422

    # A restart during extraction keeps the index and marks the interrupted entry failed.
    from app.services.maintenance import recover_interrupted_ingestions

    with sync_engine().begin() as conn:
        conn.execute(
            text("update reports set status = 'extracting' where id = :id"), {"id": report_id}
        )
        conn.execute(
            text("update extractions set status = 'pending' where report_id = :id and key = 'fte'"),
            {"id": report_id},
        )
    client.portal.call(recover_interrupted_ingestions)
    report = client.get(f"/api/reports/{report_id}").json()
    assert report["status"] == "ready" and report["error"] is None
    assert _outcomes(report)["fte"]["status"] == "error"
    assert "restart" in _outcomes(report)["fte"]["message"]

    assert client.delete(f"/api/reports/{report_id}").status_code == 204


def test_no_provider_marks_every_entry_failed_and_keeps_the_report_ready(
    client, tmp_path, monkeypatch
):
    """A failure before any entry runs (no provider configured, say) is recorded on every
    entry rather than failing the report or leaving the entries without an outcome."""
    from app.ingestion import pipeline

    def no_provider():
        raise RuntimeError("no LLM provider configured")

    monkeypatch.setattr(pipeline, "get_provider", no_provider)
    pdf = tmp_path / "other-company-2024.pdf"
    _make_pdf(pdf, "Other Company Annual Report 2024\n\nWe employed 99 people.\n")
    report_id = _upload(client, pdf)
    report = _wait_until_settled(client, report_id)

    assert report["status"] == "ready", report
    outcomes = _outcomes(report)
    assert set(outcomes) == REGISTRY_KEYS
    assert {o["status"] for o in outcomes.values()} == {"error"}
    assert all("no LLM provider configured" in o["message"] for o in outcomes.values())

    assert client.delete(f"/api/reports/{report_id}").status_code == 204
