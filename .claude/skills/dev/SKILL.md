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

Check: `curl localhost:8000/api/status` shows provider and models. Backend logs go to the
uvicorn terminal; the first start downloads bge-m3 and the reranker (about 4.5 GB).

Load the sample reports (ingestion runs in the background, about 2 minutes per report on a Mac,
10 to 15 minutes on CPU):

```bash
for f in data/reports/*.pdf; do curl -s -F "files=@$f" localhost:8000/api/reports; done
curl -s localhost:8000/api/reports | jq '.[] | {company, status, progress}'
```

Everything in Docker instead: `docker compose up --build -d`, UI on http://localhost:8080.
Run the backend on the host on Apple silicon: Metal makes embedding several times faster.
