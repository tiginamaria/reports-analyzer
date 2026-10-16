"""Offline evaluation of ingestion, retrieval, extraction and answers. See README.md.

Importing any `eval` module points the app at the eval database (`reports_eval` unless
EVAL_DATABASE_URL says otherwise) before `app.db` creates its engine, so an eval run never
touches the reports ingested through the UI.
"""

import os

EVAL_DATABASE_URL = os.environ.get(
    "EVAL_DATABASE_URL", "postgresql+psycopg://reports:reports@localhost:5432/reports_eval"
)
os.environ["DATABASE_URL"] = EVAL_DATABASE_URL
