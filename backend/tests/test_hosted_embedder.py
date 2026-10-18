"""The hosted embedder: batched requests, bounded concurrency, cancellation between windows,
input limits, the SDK's retry and timeout settings, and provider errors as `EmbeddingError`."""

import threading
import time
from types import SimpleNamespace

import httpx
import openai
import pytest

from app.config import Settings
from app.retrieval.embedder import EMBEDDING_MAX_CHARS, EmbeddingError, OpenAIEmbedder


class FakeEmbeddings:
    """`embeddings.create` returning one-float vectors (the text length), out of order to
    check that the embedder sorts by index; tracks how many requests run at once."""

    def __init__(self, delay: float = 0.0, error: Exception | None = None) -> None:
        self.requests: list[list[str]] = []
        self.delay = delay
        self.error = error
        self.in_flight = 0
        self.max_in_flight = 0
        self._lock = threading.Lock()

    def create(self, *, model, input, **kwargs):
        with self._lock:
            self.requests.append(list(input))
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.error is not None:
                raise self.error
            time.sleep(self.delay)
            data = [
                SimpleNamespace(index=i, embedding=[float(len(t))]) for i, t in enumerate(input)
            ]
            return SimpleNamespace(data=list(reversed(data)))
        finally:
            with self._lock:
                self.in_flight -= 1


def embedder(
    batch_size=4, concurrency=2, delay=0.0, error=None
) -> tuple[OpenAIEmbedder, FakeEmbeddings]:
    fake = FakeEmbeddings(delay, error)
    e = OpenAIEmbedder.__new__(OpenAIEmbedder)
    e._client = SimpleNamespace(embeddings=fake)
    e._model = "test-model"
    e._dimensions = None
    e.batch_size = batch_size
    e.concurrency = concurrency
    e.slice_size = batch_size * concurrency
    e.signature = "openai/test-model"
    return e, fake


def test_documents_go_in_batches_and_come_back_in_order():
    e, fake = embedder(batch_size=4)
    texts = ["x" * n for n in range(1, 11)]
    vectors = e.embed_documents(texts)
    assert [len(r) for r in fake.requests] == [4, 4, 2]
    assert vectors == [[float(n)] for n in range(1, 11)]


def test_at_most_concurrency_requests_are_in_flight():
    e, fake = embedder(batch_size=1, concurrency=2, delay=0.02)
    e.embed_documents([str(i) for i in range(6)])
    assert len(fake.requests) == 6
    assert fake.max_in_flight == 2


def test_cancellation_is_seen_between_windows_and_stops_the_rest():
    e, fake = embedder(batch_size=1, concurrency=2)
    windows = {"started": 0}

    def check():
        windows["started"] += 1
        if windows["started"] == 2:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        e.embed_documents([str(i) for i in range(6)], check=check)
    assert len(fake.requests) == 2, "only the first window ran"


def test_empty_and_over_long_inputs_are_made_acceptable():
    e, fake = embedder()
    e.embed_documents(["", "y" * (EMBEDDING_MAX_CHARS + 5)])
    assert fake.requests[0][0] == "-"
    assert len(fake.requests[0][1]) == EMBEDDING_MAX_CHARS


def test_query_embedding_is_a_single_request():
    e, fake = embedder()
    assert e.embed_query("hello") == [5.0]
    assert fake.requests == [["hello"]]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (openai.RateLimitError, "rate limit hit after retries"),
        (openai.AuthenticationError, "rejected the API key"),
        (openai.InternalServerError, "error 500"),
    ],
)
def test_provider_errors_become_embedding_errors(error, expected):
    status = {openai.RateLimitError: 429, openai.AuthenticationError: 401}.get(error, 500)
    response = httpx.Response(status, request=httpx.Request("POST", "https://example.test"))
    e, _ = embedder(error=error("boom", response=response, body=None))
    with pytest.raises(EmbeddingError, match=expected):
        e.embed_query("q")


def test_timeout_and_retries_come_from_the_settings():
    settings = Settings(
        _env_file=None,
        embedding_api_key="test-key",
        embedding_timeout_seconds=7,
        embedding_max_retries=2,
        embedding_batch_size=16,
        embedding_concurrency=3,
    )
    e = OpenAIEmbedder(settings)
    assert e._client.timeout == 7
    assert e._client.max_retries == 2
    assert e.slice_size == 48
    assert e.signature == "openai/text-embedding-3-large"


def test_a_missing_key_is_a_clear_error():
    with pytest.raises(EmbeddingError, match="EMBEDDING_API_KEY"):
        OpenAIEmbedder(Settings(_env_file=None, openai_api_key=None, embedding_api_key=None))


def test_the_signature_names_provider_model_and_dimensions():
    assert Settings(_env_file=None).embedding_signature == "openai/text-embedding-3-large"
    assert (
        Settings(_env_file=None, embedding_dimensions=1024).embedding_signature
        == "openai/text-embedding-3-large@1024"
    )
    local = Settings(_env_file=None, embedding_provider="local", embedding_model="BAAI/bge-m3")
    assert local.embedding_signature == "local/BAAI/bge-m3"
    assert local.reranker_name is None
    assert (
        Settings(_env_file=None, reranker_provider="local").reranker_name
        == "BAAI/bge-reranker-v2-m3"
    )


# --- the retrieval switch ----------------------------------------------------------------


def test_retrieval_models_local_selects_bge_embeddings_and_the_reranker(monkeypatch):
    monkeypatch.delenv("RERANKER_PROVIDER", raising=False)  # conftest pins it off for unit tests
    s = Settings(_env_file=None, retrieval_models="local")
    assert (s.embedding_provider, s.embedding_model, s.reranker_provider) == (
        "local",
        "BAAI/bge-m3",
        "local",
    )
    assert s.embedding_signature == "local/BAAI/bge-m3"
    assert s.reranker_name == "BAAI/bge-reranker-v2-m3"


def test_retrieval_models_openai_is_the_default_and_runs_no_reranker():
    s = Settings(_env_file=None)
    assert s.retrieval_models == "openai"
    assert (s.embedding_provider, s.embedding_model, s.reranker_provider) == (
        "openai",
        "text-embedding-3-large",
        "none",
    )


def test_explicit_settings_win_over_the_profile():
    s = Settings(_env_file=None, retrieval_models="local", reranker_provider="none")
    assert s.embedding_provider == "local" and s.reranker_provider == "none"
    s = Settings(
        _env_file=None, retrieval_models="openai", embedding_model="text-embedding-3-small"
    )
    assert s.embedding_signature == "openai/text-embedding-3-small"


def test_the_switch_is_read_from_the_environment_and_empty_overrides_are_ignored(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_MODELS", "LOCAL")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "")
    monkeypatch.setenv("RERANKER_PROVIDER", "")
    s = Settings(_env_file=None)
    assert s.retrieval_models == "local"
    assert s.embedding_provider == "local" and s.reranker_provider == "local"


def test_an_unknown_switch_value_is_rejected():
    with pytest.raises(ValueError):
        Settings(_env_file=None, retrieval_models="cohere")
