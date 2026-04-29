from typing import Any
from uuid import UUID

from sqlalchemy import bindparam, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.models import DocumentChunk
from app.rag.embeddings import cosine_similarity, embed_text


def rank_candidate_contexts(
    user_query: str, candidate_contexts: list[dict[str, Any]], *, top_k: int = 3
) -> list[dict[str, Any]]:
    """
    Embedding-based context ranking.
    - Uses precomputed chunk embeddings when present.
    - Falls back to on-the-fly deterministic chunk embeddings.
    """
    query_embedding = embed_text(user_query)
    scored: list[tuple[float, dict[str, Any]]] = []

    for item in candidate_contexts:
        maybe_embedding = item.get("embedding")
        if isinstance(maybe_embedding, list) and maybe_embedding:
            chunk_embedding = [float(value) for value in maybe_embedding]
        else:
            chunk_embedding = embed_text(str(item.get("content", "")))

        score = cosine_similarity(query_embedding, chunk_embedding)
        enriched = dict(item)
        enriched["score"] = round(float(score), 6)
        scored.append((score, enriched))

    scored.sort(key=lambda row: row[0], reverse=True)
    return [row[1] for row in scored[: max(1, top_k)]]


def get_candidate_contexts(
    db: Session,
    user_query: str,
    active_document_ids: list[str],
    *,
    candidate_limit: int = 50,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """
    Retrieval adapter:
    1) Try Postgres+pgvector SQL ordering first.
    2) Fall back to ORM fetch + Python semantic ranking.
    """
    if not active_document_ids:
        return []
    normalized_document_ids: list[Any] = []
    for document_id in active_document_ids:
        try:
            normalized_document_ids.append(UUID(str(document_id)))
        except ValueError:
            normalized_document_ids.append(str(document_id))

    pgvector_results = _try_pgvector_query(db, user_query, normalized_document_ids, top_k=top_k)
    if pgvector_results:
        return pgvector_results

    rows = db.scalars(
        select(DocumentChunk)
        .where(
            DocumentChunk.document_id.in_(normalized_document_ids),
            DocumentChunk.parent_summary_id.is_(None),
        )
        .order_by(DocumentChunk.created_at.desc())
        .limit(candidate_limit)
    ).all()
    candidates = [
        {
            "document_id": str(row.document_id),
            "chunk_id": str(row.id),
            "content": row.content,
            "embedding": row.metadata_json.get("embedding") if isinstance(row.metadata_json, dict) else None,
        }
        for row in rows
    ]
    return rank_candidate_contexts(user_query, candidates, top_k=top_k)


def _try_pgvector_query(
    db: Session, user_query: str, active_document_ids: list[Any], *, top_k: int
) -> list[dict[str, Any]]:
    bind = db.get_bind()
    if bind is None or bind.dialect.name != "postgresql":
        return []

    query_embedding = embed_text(user_query)
    query_vector_literal = "[" + ",".join(f"{value:.8f}" for value in query_embedding) + "]"
    sql = (
        text(
            """
            SELECT
              id::text AS chunk_id,
              document_id::text AS document_id,
              content,
              1 - ((metadata->>'embedding')::vector <=> CAST(:query_vector AS vector)) AS score
            FROM document_chunks
            WHERE parent_summary_id IS NULL
              AND metadata ? 'embedding'
              AND document_id IN :document_ids
            ORDER BY ((metadata->>'embedding')::vector <=> CAST(:query_vector AS vector)) ASC
            LIMIT :top_k
            """
        )
        .bindparams(bindparam("document_ids", expanding=True))
    )
    try:
        rows = db.execute(
            sql,
            {
                "query_vector": query_vector_literal,
                "document_ids": active_document_ids,
                "top_k": top_k,
            },
        ).mappings()
        return [
            {
                "document_id": row["document_id"],
                "chunk_id": row["chunk_id"],
                "content": row["content"],
                "score": round(float(row["score"]), 6) if row["score"] is not None else 0.0,
                "retrieval_strategy": "pgvector",
            }
            for row in rows
        ]
    except SQLAlchemyError:
        # Clear failed transaction so ORM fallback queries can run.
        db.rollback()
        return []
