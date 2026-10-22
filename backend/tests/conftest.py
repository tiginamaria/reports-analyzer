import os

import pytest

# Unit tests never load a local model; the integration tests use the hosted embedder.
os.environ.setdefault("RERANKER_PROVIDER", "none")


@pytest.fixture
def anyio_backend() -> str:
    """Async tests (`pytest.mark.anyio`) run on asyncio, the app's event loop."""
    return "asyncio"
