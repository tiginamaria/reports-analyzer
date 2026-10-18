---
name: dev
description: Start the app for local development (Postgres in Docker, backend and frontend on the host) and load sample reports. Use when asked to run, start, try or demo the app.
---

# Run the app locally

Secrets live in the repo-root `.env` (gitignored, copied from `.env.example`); a `backend/.env`
overrides it. Never print either file.
/
```bash
docker compose up -d db                       # Postgres + pgvector on :5432
cd backend && uv run alembic upgrade head && uv run uvicorn app.main:app --reload --port 8000
cd frontend && npm run dev                    # http://localhost:5173, proxies /api to :8000
```

Check: `curl localhost:8000/api/status` shows the LLM provider, the embedding signature
(`openai/text-embedding-3-large` by default) and the reranker (`none` by default). Backend logs
go to the uvicorn terminal. `RETRIEVAL_MODELS=local` in `.env` switches to bge-m3 embeddings
plus the bge reranker in-process (first start downloads about 4.5 GB).

Reports are searched only when their `embedding_model` equals the configured signature; after
changing the embedding settings run `uv run python -m app.reembed` (or delete and re-upload).

Load the sample reports (ingestion runs in the background; with hosted embeddings a report takes
about 1.5 to 3 minutes, most of it the extraction LLM calls):

```bash
for f in data/reports/*.pdf; do curl -s -F "files=@$f" localhost:8000/api/reports; done
curl -s localhost:8000/api/reports | jq '.[] | {company, status, progress}'
```

Everything in Docker instead: `docker compose up --build -d`, UI on http://localhost:8080.
