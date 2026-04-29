from io import BytesIO
from typing import Any

from pypdf import PdfReader

from app.embeddings.factory import EmbeddingProvider, get_embedding_model, resolve_embedding_config


def extract_text_from_upload(file_name: str, file_bytes: bytes) -> str:
    lower_name = file_name.lower()
    if lower_name.endswith(".pdf"):
        reader = PdfReader(BytesIO(file_bytes))
        pages = [(page.extract_text() or "") for page in reader.pages]
        return "\n".join(pages).strip()
    return file_bytes.decode("utf-8", errors="ignore").strip()


def chunk_text(text: str, *, words_per_chunk: int = 180) -> list[str]:
    words = text.split()
    if not words:
        return []
    return [
        " ".join(words[idx : idx + words_per_chunk]) for idx in range(0, len(words), words_per_chunk)
    ]


def build_chunk_payloads(
    chunks: list[str], *, embedding_provider: EmbeddingProvider | None = None
) -> list[dict[str, Any]]:
    if not chunks:
        return []
    model = get_embedding_model(embedding_provider)
    embeddings = model.embed_documents(chunks)
    return [
        {
            "content": chunk,
            "embedding": embedding,
            "chunk_index": idx,
        }
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]


def estimate_embedding_usage(
    chunks: list[str], *, embedding_provider: EmbeddingProvider | None = None
) -> dict[str, Any]:
    resolved_provider, resolved_model = resolve_embedding_config(embedding_provider)
    approx_input_tokens = int(sum(max(1, len(chunk.split())) for chunk in chunks) * 1.3)
    # Simple pricing table for prototype accounting.
    price_per_1k_tokens = 0.00002 if resolved_provider == "openai" else 0.0
    estimated_cost_usd = round((approx_input_tokens / 1000) * price_per_1k_tokens, 8)
    return {
        "embedding_provider": resolved_provider,
        "embedding_model": resolved_model,
        "approx_input_tokens": approx_input_tokens,
        "estimated_cost_usd": estimated_cost_usd,
    }
