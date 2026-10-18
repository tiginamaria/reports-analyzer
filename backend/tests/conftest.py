import os

# Unit tests never load a local model; the integration tests use the hosted embedder.
os.environ.setdefault("RERANKER_PROVIDER", "none")
