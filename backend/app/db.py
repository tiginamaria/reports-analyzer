"""Async database engine and session factory.

The whole application talks to Postgres through `AsyncSession` (psycopg 3 in async mode):
request handlers, background ingestion, the chat pipeline and the evaluation scripts alike.
Alembic keeps its own synchronous engine built from the same URL."""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings

engine = create_async_engine(get_settings().database_url, pool_pre_ping=True)
SessionLocal = async_sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

__all__ = ["AsyncSession", "SessionLocal", "engine"]
