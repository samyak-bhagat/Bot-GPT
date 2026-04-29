from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Document, DocumentChunk, User
from app.db.session import get_db
from app.embeddings.factory import EmbeddingProvider, EmbeddingProviderUnavailableError
from app.services.ingestion import (
    build_chunk_payloads,
    chunk_text,
    estimate_embedding_usage,
    extract_text_from_upload,
)

router = APIRouter()


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    status: str
    chunk_count: int
    created_at: datetime


class DocumentListResponse(BaseModel):
    items: list[DocumentSummary]


def _get_or_create_user(db: Session, user_key: str) -> User:
    user = db.scalar(select(User).where(User.email == user_key))
    if user:
        return user
    user = User(email=user_key)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _resolve_user(
    db: Session = Depends(get_db), x_user_id: str | None = Header(default=None, alias="X-User-Id")
) -> User:
    user_key = x_user_id or "demo@example.com"
    return _get_or_create_user(db, user_key)


@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    db: Session = Depends(get_db),
    user: User = Depends(_resolve_user),
    limit: int = 100,
) -> DocumentListResponse:
    safe_limit = max(1, min(limit, 500))
    rows = db.scalars(
        select(Document)
        .where(Document.user_id == user.id)
        .order_by(Document.created_at.desc())
        .limit(safe_limit)
    ).all()
    return DocumentListResponse(
        items=[
            DocumentSummary(
                document_id=str(row.id),
                filename=row.filename,
                status=row.status,
                chunk_count=row.chunk_count,
                created_at=row.created_at,
            )
            for row in rows
        ]
    )


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def ingest_document(
    file: UploadFile = File(...),
    embedding_provider: EmbeddingProvider | None = None,
    db: Session = Depends(get_db),
    user: User = Depends(_resolve_user),
) -> dict:
    raw_bytes = await file.read()
    document = Document(
        user_id=user.id,
        filename=file.filename or "uploaded-document",
        status="processing",
        chunk_count=0,
    )
    db.add(document)
    db.flush()

    try:
        text = extract_text_from_upload(document.filename, raw_bytes)
        chunks = chunk_text(text)
        usage = estimate_embedding_usage(chunks, embedding_provider=embedding_provider)
        payloads = build_chunk_payloads(chunks, embedding_provider=embedding_provider)
        for item in payloads:
            db.add(
                DocumentChunk(
                    document_id=document.id,
                    chunk_index=item["chunk_index"],
                    content=item["content"],
                    metadata_json={
                        "embedding": item["embedding"],
                        "embedding_provider": usage["embedding_provider"],
                        "embedding_model": usage["embedding_model"],
                        "approx_input_tokens": usage["approx_input_tokens"],
                        "estimated_cost_usd": usage["estimated_cost_usd"],
                    },
                )
            )
        document.chunk_count = len(payloads)
        document.status = "ready"
        db.commit()
    except EmbeddingProviderUnavailableError as exc:
        document.status = "failed"
        db.commit()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        document.status = "failed"
        db.commit()
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {exc}") from exc

    return {
        "document_id": str(document.id),
        "status": document.status,
        "chunk_count": document.chunk_count,
        "embedding_usage": usage,
    }


@router.get("/{document_id}")
async def get_document_status(
    document_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(_resolve_user),
) -> dict:
    document = db.scalar(
        select(Document).where(
            Document.id == document_id,
            Document.user_id == user.id,
        )
    )
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    chunk_rows = db.scalars(
        select(DocumentChunk).where(DocumentChunk.document_id == document.id).order_by(DocumentChunk.chunk_index.asc())
    ).all()
    embedding_provider = "unknown"
    embedding_model = "unknown"
    approx_input_tokens = 0
    estimated_cost_usd = 0.0
    if chunk_rows:
        first_metadata = chunk_rows[0].metadata_json if isinstance(chunk_rows[0].metadata_json, dict) else {}
        embedding_provider = first_metadata.get("embedding_provider", "unknown")
        embedding_model = first_metadata.get("embedding_model", "unknown")
        approx_input_tokens = int(first_metadata.get("approx_input_tokens", 0) or 0)
        estimated_cost_usd = float(first_metadata.get("estimated_cost_usd", 0.0) or 0.0)
    return {
        "document_id": str(document.id),
        "status": document.status,
        "chunk_count": document.chunk_count,
        "filename": document.filename,
        "embedding_usage": {
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
            "approx_input_tokens": approx_input_tokens,
            "estimated_cost_usd": round(estimated_cost_usd, 8),
        },
    }
