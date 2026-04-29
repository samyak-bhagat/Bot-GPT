from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from app.core.config import settings

Provider = Literal["groq", "openai", "ollama"]


class ProviderUnavailableError(RuntimeError):
    """Raised when the selected provider is not configured or unavailable."""


def get_chat_model(
    provider: Provider,
    model: str,
    *,
    temperature: float = 0.2,
    timeout: int = 30,
) -> BaseChatModel:
    if provider == "groq":
        if not settings.groq_api_key:
            raise ProviderUnavailableError(
                "Groq is selected but GROQ_API_KEY is not configured."
            )
        return ChatGroq(
            model=model,
            temperature=temperature,
            timeout=timeout,
            api_key=settings.groq_api_key,
        )
    if provider == "openai":
        if not settings.openai_api_key:
            raise ProviderUnavailableError(
                "OpenAI is selected but OPENAI_API_KEY is not configured."
            )
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            timeout=timeout,
            api_key=settings.openai_api_key,
        )
    if provider == "ollama":
        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=settings.ollama_base_url,
        )
    raise ValueError(f"Unknown provider: {provider}")
