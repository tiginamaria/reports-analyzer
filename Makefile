.PHONY: up down logs dev-db dev-backend dev-frontend test lint

up:            ## build and start db, backend and frontend
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f backend

dev-db:        ## only the database, for running backend/frontend on the host
	docker compose up -d db

dev-backend:
	cd backend && uv run alembic upgrade head && uv run uvicorn app.main:app --reload --port 8000

dev-frontend:
	cd frontend && npm run dev

test:          ## unit tests (offline) and integration tests (need db + embedding model)
	cd backend && uv run pytest -q -m "not integration" && uv run pytest -q -m integration

lint:
	cd backend && uv run ruff check . && uv run ruff format --check . && uv run mypy app
	cd frontend && npm run lint && npm run typecheck
