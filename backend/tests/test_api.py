"""API tests against a real Postgres (skipped when DATABASE_URL is unreachable).

Uses a tiny generated PDF, the configured LLM provider and the configured embedding
provider; skipped when no API key is set.
"""

import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from tests.helpers import require_api_key

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    reports_dir = tmp_path_factory.mktemp("reports")
    os.environ["REPORTS_DIR"] = str(reports_dir)
    from app.config import get_settings

    get_settings.cache_clear()
    require_api_key(get_settings())
    from app.db import engine

    try:
        with engine.connect() as conn:
            conn.execute(text("select 1"))
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Postgres not available: {exc}")
    from app.main import app

    with TestClient(app) as c:
        yield c


def _make_pdf(path: Path) -> None:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Test Company Annual Report 2025\n\n"
        "At year-end 2025 Test Company employed 1,234 FTEs across all regions.\n"
        "Our goal is to reach net zero emissions by 2040.",
        fontsize=11,
    )
    doc.save(path)


def test_upload_ingest_and_chat(client: TestClient, tmp_path: Path):
    pdf = tmp_path / "test-company-annual-report-2025.pdf"
    _make_pdf(pdf)
    with pdf.open("rb") as f:
        r = client.post("/api/reports", files=[("files", (pdf.name, f, "application/pdf"))])
    assert r.status_code == 201, r.text
    assert r.json()["rejected"] == []
    report_id = r.json()["started"][0]["id"]

    # a duplicate and a non-PDF are rejected per file; nothing started, so 200 not 201
    not_pdf = tmp_path / "notes.pdf"
    not_pdf.write_text("not a pdf")
    with pdf.open("rb") as f, not_pdf.open("rb") as g:
        dup = client.post(
            "/api/reports",
            files=[
                ("files", (pdf.name, f, "application/pdf")),
                ("files", (not_pdf.name, g, "application/pdf")),
            ],
        )
    assert dup.status_code == 200
    assert dup.json()["started"] == []
    reasons = {r["filename"]: r["reason"] for r in dup.json()["rejected"]}
    assert "Already ingested" in reasons[pdf.name]
    assert reasons[not_pdf.name] == "Not a PDF file"
    assert not list(Path(os.environ["REPORTS_DIR"]).glob("*.upload")), "temp files must not leak"

    for _ in range(120):
        status = client.get(f"/api/reports/{report_id}").json()
        if status["status"] in ("ready", "failed"):
            break
        time.sleep(1)
    assert status["status"] == "ready", status
    assert status["company"]  # detected by the LLM from the PDF, not supplied by the client
    assert status["fiscal_year"] == 2025
    assert status["page_count"] == 1
    assert status["chunk_count"] >= 1

    dps = client.get("/api/datapoints", params={"report_id": report_id}).json()
    assert dps, "the FTE sentence in the fixture should yield a datapoint"
    assert all(d["verification"] == "verified" for d in dps)

    with client.stream(
        "POST", "/api/chat", json={"question": "How many FTEs in 2025?", "report_ids": [report_id]}
    ) as resp:
        body = resp.read().decode()
    assert "event: answer" in body
    assert '"verification": "verified"' in body or '"verification":"verified"' in body

    convos = client.get("/api/conversations").json()
    first_id = convos[0]["id"]  # newest first; the database may hold older conversations
    detail = client.get(f"/api/conversations/{first_id}").json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    assert detail["updated_at"] > detail["created_at"], "messages must bump updated_at"

    # a filter on a report that is not indexed abstains instead of searching everything
    with client.stream(
        "POST", "/api/chat", json={"question": "How many FTEs?", "report_ids": ["no-such-id"]}
    ) as resp:
        body = resp.read().decode()
    assert "event: answer" in body
    assert "not_found" in body
    assert '"citations": []' in body or '"citations":[]' in body

    # the second conversation is newer, so it moves ahead of the first
    convos = client.get("/api/conversations").json()
    second_id = convos[0]["id"]
    assert second_id != first_id and convos[1]["id"] == first_id

    pdf_resp = client.get(f"/api/reports/{report_id}/pdf")
    assert pdf_resp.status_code == 200

    # a cited page is served as an image, so the UI never has to load the whole PDF
    page_img = client.get(f"/api/reports/{report_id}/pages/1")
    assert page_img.status_code == 200
    assert page_img.headers["content-type"] == "image/png"
    assert page_img.content.startswith(b"\x89PNG")
    assert client.get(f"/api/reports/{report_id}/pages/2").status_code == 404

    assert client.delete(f"/api/reports/{report_id}").status_code == 204
    assert client.get(f"/api/reports/{report_id}").status_code == 404
    assert not list(Path(os.environ["REPORTS_DIR"]).glob("*.pdf")), "delete removes the PDF"
    for convo_id in (first_id, second_id):
        assert client.delete(f"/api/conversations/{convo_id}").status_code == 204
