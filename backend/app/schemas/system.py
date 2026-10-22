"""Provider and model status."""

from pydantic import BaseModel


class StatusOut(BaseModel):
    llm_provider: str
    llm_model: str
    embedding_provider: str
    # The configured embedding signature (`openai/text-embedding-3-large`); reports whose
    # `embedding_model` differs are not searched until they are re-embedded.
    embedding_model: str
    reranker_model: str | None
    # Indexed reports whose vectors come from another embedding model.
    reports_to_reembed: int
