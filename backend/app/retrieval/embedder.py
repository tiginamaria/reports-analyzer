"""Local embedding and reranking models (sentence-transformers). Loaded lazily, once."""

from __future__ import annotations

import logging
import threading

from app.config import get_settings

log = logging.getLogger(__name__)

# One lock for loading and running both models. Loading a model onto the GPU while another
# model is encoding crashes Metal (MPS), so construction takes the same lock as inference.
_model_lock = threading.Lock()

RERANK_BATCH_SIZE = 8
RERANK_MAX_TOKENS = 1024


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


class Embedder:
    def __init__(self, model_name: str, batch_size: int) -> None:
        from sentence_transformers import SentenceTransformer

        log.info("Loading embedding model %s", model_name)
        self.model = SentenceTransformer(model_name, device=_device())
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        with _model_lock:
            vectors = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


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
        with _model_lock:
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


def get_embedder() -> Embedder:
    global _embedder
    with _model_lock:
        if _embedder is None:
            settings = get_settings()
            _embedder = Embedder(settings.embedding_model, settings.embedding_batch_size)
        return _embedder


def get_reranker() -> Reranker | None:
    global _reranker, _reranker_loaded
    with _model_lock:
        if not _reranker_loaded:
            settings = get_settings()
            _reranker = Reranker(settings.reranker_model) if settings.reranker_model else None
            _reranker_loaded = True
        return _reranker
