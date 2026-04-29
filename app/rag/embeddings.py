import hashlib
import math


EMBEDDING_DIMS = 64


def embed_text(text: str, *, dims: int = EMBEDDING_DIMS) -> list[float]:
    """
    Deterministic lightweight embedding for scaffold/dev.
    This is a placeholder until a real embedding model is wired.
    """
    vector = [0.0 for _ in range(dims)]
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for idx in range(dims):
            vector[idx] += digest[idx % len(digest)] / 255.0
    return _normalize(vector)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]
