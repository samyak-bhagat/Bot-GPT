from typing import Literal, Protocol

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings
from app.rag.embeddings import embed_text

EmbeddingProvider = Literal["huggingface", "openai"]


class EmbeddingProviderUnavailableError(RuntimeError):
    """Raised when selected embedding backend is not configured/available."""


class EmbeddingModel(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class DeterministicEmbeddings:
    """Stable fallback when external embedding providers are unavailable."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [embed_text(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return embed_text(text)


def get_embedding_model(provider: EmbeddingProvider | None = None) -> EmbeddingModel:
    resolved_provider, _ = resolve_embedding_config(provider)
    if resolved_provider == "openai":
        if not settings.openai_api_key:
            raise EmbeddingProviderUnavailableError(
                "OpenAI embeddings selected but OPENAI_API_KEY is not configured."
            )
        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )

    # Default: HuggingFace sentence-transformers
    try:
        return HuggingFaceEmbeddings(model_name=settings.embedding_model)
    except Exception as exc:
        raise EmbeddingProviderUnavailableError(
            f"HuggingFace embeddings unavailable for model={settings.embedding_model}: {exc}"
        ) from exc


def resolve_embedding_config(provider: EmbeddingProvider | None = None) -> tuple[EmbeddingProvider, str]:
    resolved_provider: EmbeddingProvider = provider or settings.embedding_provider.lower()  # type: ignore[assignment]
    resolved_model = (
        settings.openai_embedding_model if resolved_provider == "openai" else settings.embedding_model
    )
    return resolved_provider, resolved_model
