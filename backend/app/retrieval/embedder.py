"""Embeddings and reranking behind one interface: a hosted embedding endpoint (default) or
local sentence-transformers models, loaded lazily, once.

Local models share `model_lock`, a GPU-safety lock: loading a model onto the GPU while
another model is encoding crashes Metal (MPS), so construction and inference take the same
lock. The lock is held per model batch, never per report, and interactive callers (a chat
query, a rerank) go ahead of background work (document embedding) between batches.
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from app.config import Settings, get_settings

log = logging.getLogger(__name__)

RERANK_BATCH_SIZE = 8
RERANK_MAX_TOKENS = 1024
# Hosted models reject inputs over their token limit (8,192 for OpenAI's); a chunk is at most
# a few hundred words, but a table chunk can be long. Cut by characters, a safe margin below.
EMBEDDING_MAX_CHARS = 24_000

# Called between inference batches; raises to stop the work (ingestion passes its
# cancellation check, so a deleted report stops before its next batch starts).
Check = Callable[[], None]


class EmbeddingError(RuntimeError):
    """The embedding endpoint could not be used; the message is shown to the user."""


class ModelLock:
    """A mutex with two priorities. Background work (embedding a report) holds it for one
    model batch at a time and, on release, lets any waiting interactive caller (a chat
    query, a rerank) in before it takes the lock again, so a long ingestion never
    monopolises the model. Only one thread ever runs a local model, which is what keeps
    Metal from crashing."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._held = False
        self._interactive_waiting = 0

    def acquire(self, interactive: bool = False) -> None:
        with self._cond:
            if interactive:
                self._interactive_waiting += 1
                try:
                    while self._held:
                        self._cond.wait()
                finally:
                    self._interactive_waiting -= 1
            else:
                while self._held or self._interactive_waiting:
                    self._cond.wait()
            self._held = True

    def release(self) -> None:
        with self._cond:
            self._held = False
            self._cond.notify_all()

    @contextmanager
    def held(self, interactive: bool = False) -> Iterator[None]:
        self.acquire(interactive)
        try:
            yield
        finally:
            self.release()

    def locked(self) -> bool:
        return self._held


model_lock = ModelLock()


def _device() -> str:
    settings = get_settings()
    if settings.model_device:
        return settings.model_device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _batches(texts: list[str], size: int) -> list[list[str]]:
    return [texts[i : i + size] for i in range(0, len(texts), size)]


class Embedder(ABC):
    """`signature` names the vectors (see `Settings.embedding_signature`); `slice_size` is
    how many documents one `embed_documents` call should get to be efficient while staying
    short, so the caller can persist, report progress and check cancellation between calls."""

    signature: str
    slice_size: int

    @abstractmethod
    def embed_documents(self, texts: list[str], check: Check | None = None) -> list[list[float]]:
        """Vectors for `texts`, in order. `check` is called before every batch."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...


class LocalEmbedder(Embedder):
    """A sentence-transformers model in this process. Each batch of `batch_size` texts is
    encoded under the model lock on its own, so a chat query waits for one batch at most."""

    def __init__(self, settings: Settings) -> None:
        from sentence_transformers import SentenceTransformer

        log.info("Loading embedding model %s", settings.embedding_model)
        self.model = SentenceTransformer(settings.embedding_model, device=_device())
        self.batch_size = settings.embedding_batch_size
        self.slice_size = settings.embedding_batch_size
        self.signature = settings.embedding_signature

    def embed_documents(self, texts: list[str], check: Check | None = None) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in _batches(texts, self.batch_size):
            with model_lock.held():
                if check is not None:  # the wait may have been long; stop before encoding
                    check()
                vectors.extend(self._encode(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        with model_lock.held(interactive=True):
            return self._encode([text])[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        encoded = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in encoded]


class OpenAIEmbedder(Embedder):
    """A hosted embedding endpoint with the OpenAI API (OpenAI itself or any compatible
    server through `embedding_base_url`). Documents go in batches of `batch_size` texts,
    `concurrency` requests at a time; the SDK retries each request on timeouts, rate limits
    and server errors with backoff, `max_retries` times. No lock: the service is the
    limit, and a chat query never waits for a report."""

    # Called after every request with the tokens billed and the seconds it took; the eval
    # harness records the embedding cost through it.
    on_response: Callable[[int, float], None] | None = None

    def __init__(self, settings: Settings) -> None:
        import openai

        api_key = settings.embedding_api_key or settings.openai_api_key
        if not api_key:
            raise EmbeddingError(
                "No API key for the embedding provider: set EMBEDDING_API_KEY (or OPENAI_API_KEY)"
            )
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=settings.embedding_base_url,
            timeout=settings.embedding_timeout_seconds,
            max_retries=settings.embedding_max_retries,
        )
        self._model = settings.embedding_model
        self._dimensions = settings.embedding_dimensions
        self.batch_size = settings.embedding_batch_size
        self.concurrency = max(1, settings.embedding_concurrency)
        self.slice_size = self.batch_size * self.concurrency
        self.signature = settings.embedding_signature

    def embed_documents(self, texts: list[str], check: Check | None = None) -> list[list[float]]:
        batches = _batches(texts, self.batch_size)
        vectors: list[list[float]] = []
        with ThreadPoolExecutor(self.concurrency) as pool:
            # one window of `concurrency` requests at a time, so a cancellation is seen
            # before the next window and at most one window is ever in flight
            for window in _batches(batches, self.concurrency):  # type: ignore[arg-type]
                if check is not None:
                    check()
                for result in pool.map(self._embed_batch, window):
                    vectors.extend(result)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        import openai

        inputs = [_embedding_input(t) for t in texts]
        start = time.perf_counter()
        try:
            if self._dimensions:
                response = self._client.embeddings.create(
                    model=self._model, input=inputs, dimensions=self._dimensions
                )
            else:
                response = self._client.embeddings.create(model=self._model, input=inputs)
        except openai.AuthenticationError as exc:
            raise EmbeddingError("Embedding provider rejected the API key") from exc
        except openai.RateLimitError as exc:
            raise EmbeddingError("Embedding provider rate limit hit after retries") from exc
        except openai.APITimeoutError as exc:
            raise EmbeddingError("Embedding request timed out after retries") from exc
        except openai.APIStatusError as exc:
            raise EmbeddingError(
                f"Embedding provider error {exc.status_code}: {exc.message}"
            ) from exc
        except openai.APIConnectionError as exc:
            raise EmbeddingError(f"Could not reach the embedding provider: {exc}") from exc
        if self.on_response:
            usage = getattr(response, "usage", None)
            self.on_response(usage.total_tokens if usage else 0, time.perf_counter() - start)
        data = sorted(response.data, key=lambda item: item.index)
        if len(data) != len(texts):
            raise EmbeddingError(
                f"Embedding provider returned {len(data)} vectors for {len(texts)} texts"
            )
        return [list(item.embedding) for item in data]


def _embedding_input(text: str) -> str:
    """The endpoint rejects empty strings and over-long inputs."""
    return text[:EMBEDDING_MAX_CHARS] or "-"


class Reranker:
    def __init__(self, model_name: str) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        log.info("Loading reranker %s", model_name)
        self.model = CrossEncoder(
            model_name,
            device=_device(),
            max_length=RERANK_MAX_TOKENS,
            activation_fn=torch.nn.Sigmoid(),
        )

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Return a relevance probability in [0, 1] per passage."""
        if not passages:
            return []
        with model_lock.held(interactive=True):
            probabilities = self.model.predict(
                [(query, passage) for passage in passages],
                batch_size=RERANK_BATCH_SIZE,
                show_progress_bar=False,
            )
        return [float(p) for p in probabilities]


# Loaded once per process. Not `lru_cache`: it only stores a result after the call returns,
# so concurrent first calls (ingestion and chat, or an eval's worker threads) each loaded
# their own copy onto the GPU. Construction happens under the model lock, so a second
# caller waits and then finds the instance.
_embedder: Embedder | None = None
_reranker: Reranker | None = None
_reranker_loaded = False
_hosted_lock = threading.Lock()


def get_embedder() -> Embedder:
    global _embedder
    settings = get_settings()
    if settings.embedding_provider == "local":
        with model_lock.held(interactive=True):
            if _embedder is None:
                _embedder = LocalEmbedder(settings)
            return _embedder
    with _hosted_lock:
        if _embedder is None:
            _embedder = OpenAIEmbedder(settings)
        return _embedder


def get_reranker() -> Reranker | None:
    global _reranker, _reranker_loaded
    settings = get_settings()
    if settings.reranker_provider != "local":
        return None
    with model_lock.held(interactive=True):
        if not _reranker_loaded:
            _reranker = Reranker(settings.reranker_model)
            _reranker_loaded = True
        return _reranker


def models_to_load() -> list[str]:
    """Names of the local models the next retrieval would have to load first (empty with
    hosted embeddings and no reranker, or once they are loaded). The chat shows this as
    its own stage instead of a silent wait."""
    settings = get_settings()
    pending = []
    if settings.embedding_provider == "local" and _embedder is None:
        pending.append(settings.embedding_model)
    if settings.reranker_provider == "local" and not _reranker_loaded:
        pending.append(settings.reranker_model)
    return pending


def reset_models() -> None:
    """Forget the loaded instances (tests and the eval change settings between runs)."""
    global _embedder, _reranker, _reranker_loaded
    _embedder = None
    _reranker = None
    _reranker_loaded = False
