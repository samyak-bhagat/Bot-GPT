from datetime import datetime
from decimal import Decimal
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Response, status
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.agent.graph import run_graph
from app.core.config import settings
from app.db.models import Conversation, ConversationDocument, Document, DocumentChunk, Message, User
from app.db.session import get_db
from app.llm.factory import Provider, ProviderUnavailableError
from app.rag.retrieval import get_candidate_contexts

router = APIRouter()
def _estimate_llm_cost_usd(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    if provider != "openai":
        return 0.0
    # Lightweight pricing table for demo accounting.
    pricing_per_million: dict[str, tuple[float, float]] = {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o": (5.0, 15.0),
    }
    prompt_rate, completion_rate = pricing_per_million.get(model, pricing_per_million["gpt-4o-mini"])
    prompt_cost = (max(prompt_tokens, 0) / 1_000_000) * prompt_rate
    completion_cost = (max(completion_tokens, 0) / 1_000_000) * completion_rate
    return round(prompt_cost + completion_cost, 8)




class CreateConversationRequest(BaseModel):
    model: str | None = None
    provider: str | None = None
    document_ids: list[str] = Field(default_factory=list)


class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    model: str | None
    created_at: datetime


class ConversationResponse(BaseModel):
    id: UUID
    user_id: UUID
    title: str
    mode: str
    total_tokens: int
    llm_cost_usd: float
    embedding_cost_usd: float
    total_cost_usd: float
    created_at: datetime
    updated_at: datetime


class ConversationDetailResponse(ConversationResponse):
    messages: list[MessageResponse] = Field(default_factory=list)


class ConversationListResponse(BaseModel):
    items: list[ConversationResponse]
    next_cursor: str | None = None


class EmbeddingDocumentCostItem(BaseModel):
    document_id: str
    filename: str
    approx_input_tokens: int
    estimated_cost_usd: float


class ConversationCostBreakdownResponse(BaseModel):
    conversation_id: str
    llm: dict[str, float | int]
    embeddings: dict[str, float | int | list[EmbeddingDocumentCostItem]]
    totals: dict[str, float | int]


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


def _compute_embedding_usage_for_conversation(db: Session, conversation_id: UUID) -> tuple[int, float]:
    link_rows = db.scalars(
        select(ConversationDocument.document_id).where(ConversationDocument.conversation_id == conversation_id)
    ).all()
    if not link_rows:
        return 0, 0.0

    chunk_rows = db.scalars(
        select(DocumentChunk)
        .where(DocumentChunk.document_id.in_(link_rows))
        .order_by(DocumentChunk.chunk_index.asc())
    ).all()
    if not chunk_rows:
        return 0, 0.0

    # Ingestion writes identical aggregate usage metadata to every chunk in a document.
    seen_docs: set[str] = set()
    total_tokens = 0
    total_cost = 0.0
    for row in chunk_rows:
        doc_key = str(row.document_id)
        if doc_key in seen_docs:
            continue
        seen_docs.add(doc_key)
        metadata = row.metadata_json if isinstance(row.metadata_json, dict) else {}
        total_tokens += int(metadata.get("approx_input_tokens", 0) or 0)
        total_cost += float(metadata.get("estimated_cost_usd", 0.0) or 0.0)
    return total_tokens, round(total_cost, 8)


def _compute_embedding_cost_breakdown(
    db: Session, conversation_id: UUID
) -> tuple[list[EmbeddingDocumentCostItem], int, float]:
    document_rows = db.execute(
        select(Document.id, Document.filename)
        .join(ConversationDocument, ConversationDocument.document_id == Document.id)
        .where(ConversationDocument.conversation_id == conversation_id)
    ).all()
    if not document_rows:
        return [], 0, 0.0

    items: list[EmbeddingDocumentCostItem] = []
    total_tokens = 0
    total_cost = 0.0

    for doc_id, filename in document_rows:
        first_chunk = db.scalar(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == doc_id)
            .order_by(DocumentChunk.chunk_index.asc())
            .limit(1)
        )
        metadata = first_chunk.metadata_json if first_chunk and isinstance(first_chunk.metadata_json, dict) else {}
        doc_tokens = int(metadata.get("approx_input_tokens", 0) or 0)
        doc_cost = float(metadata.get("estimated_cost_usd", 0.0) or 0.0)
        total_tokens += doc_tokens
        total_cost += doc_cost
        items.append(
            EmbeddingDocumentCostItem(
                document_id=str(doc_id),
                filename=str(filename),
                approx_input_tokens=doc_tokens,
                estimated_cost_usd=round(doc_cost, 8),
            )
        )

    return items, total_tokens, round(total_cost, 8)


def _to_conversation_response(db: Session, conversation: Conversation) -> ConversationResponse:
    _, embedding_cost = _compute_embedding_usage_for_conversation(db, conversation.id)
    llm_cost = float(conversation.total_cost_usd or Decimal("0"))
    total_cost = round(llm_cost + embedding_cost, 8)
    return ConversationResponse(
        id=conversation.id,
        user_id=conversation.user_id,
        title=conversation.title,
        mode=conversation.mode,
        total_tokens=conversation.total_tokens,
        llm_cost_usd=llm_cost,
        embedding_cost_usd=embedding_cost,
        total_cost_usd=total_cost,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


@router.post("", status_code=status.HTTP_201_CREATED, response_model=ConversationResponse)
async def create_conversation(
    payload: CreateConversationRequest,
    db: Session = Depends(get_db),
    user: User = Depends(_resolve_user),
) -> ConversationResponse:
    mode = "rag" if payload.document_ids else "open"
    conversation = Conversation(
        user_id=user.id,
        title="New conversation",
        mode=mode,
        total_tokens=0,
        total_cost_usd=Decimal("0"),
    )
    db.add(conversation)
    db.flush()

    if payload.document_ids:
        document_uuids: list[UUID] = []
        for doc_id in payload.document_ids:
            try:
                document_uuids.append(UUID(doc_id))
            except ValueError:
                continue
        if document_uuids:
            owned_documents = db.scalars(
                select(Document).where(
                    Document.id.in_(document_uuids),
                    Document.user_id == user.id,
                )
            ).all()
            for document in owned_documents:
                db.add(
                    ConversationDocument(
                        conversation_id=conversation.id,
                        document_id=document.id,
                    )
                )

    db.commit()
    db.refresh(conversation)
    return _to_conversation_response(db, conversation)


class SendMessageRequest(BaseModel):
    content: str
    provider: Provider | None = None
    model: str | None = None


@router.post("/{conversation_id}/messages")
async def send_message(
    conversation_id: UUID,
    payload: SendMessageRequest,
    db: Session = Depends(get_db),
    user: User = Depends(_resolve_user),
) -> dict:
    conversation = db.scalar(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user.id,
        )
    )
    if not conversation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    user_message = Message(
        conversation_id=conversation.id,
        role="user",
        content=payload.content,
        model=payload.model or settings.default_model,
    )
    db.add(user_message)
    db.flush()

    history_messages = db.scalars(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.id.asc())
    ).all()
    history_payload = [{"role": item.role, "content": item.content} for item in history_messages[:-1]]
    active_document_ids = db.scalars(
        select(ConversationDocument.document_id).where(ConversationDocument.conversation_id == conversation.id)
    ).all()
    candidate_contexts = get_candidate_contexts(
        db,
        payload.content,
        [str(document_id) for document_id in active_document_ids],
        candidate_limit=50,
        top_k=3,
    )
    provider = payload.provider or settings.default_provider
    model = payload.model or settings.default_model
    try:
        graph_result = await run_graph(
            {
                "user_message": payload.content,
                "provider": provider,
                "model": model,
                "mode": conversation.mode,
                "active_document_ids": [str(document_id) for document_id in active_document_ids],
                "history": history_payload,
                "candidate_contexts": candidate_contexts,
                "selected_contexts": [],
                "assistant_message": "",
                "usage": {},
            }
        )
    except ProviderUnavailableError as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM request failed for provider={provider}, model={model}: {exc}",
        ) from exc
    assistant_text = graph_result["assistant_message"]
    usage = graph_result.get("usage", {})
    prompt_tokens = int(usage.get("input_tokens", 0) or 0)
    completion_tokens = int(usage.get("output_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)

    assistant_message = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=assistant_text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        model=model,
    )
    db.add(assistant_message)
    conversation.total_tokens = int(conversation.total_tokens or 0) + total_tokens
    llm_cost = _estimate_llm_cost_usd(provider, model, prompt_tokens, completion_tokens)
    conversation.total_cost_usd = Decimal(conversation.total_cost_usd or Decimal("0")) + Decimal(str(llm_cost))
    db.commit()

    return {
        "conversation_id": str(conversation_id),
        "user_message": payload.content,
        "assistant_message": assistant_text,
        "citations": graph_result.get("selected_contexts", []),
    }


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    db: Session = Depends(get_db),
    user: User = Depends(_resolve_user),
    limit: int = 20,
) -> ConversationListResponse:
    safe_limit = max(1, min(limit, 100))
    rows = db.scalars(
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
        .limit(safe_limit)
    ).all()
    return ConversationListResponse(
        items=[_to_conversation_response(db, row) for row in rows],
        next_cursor=None,
    )


@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    conversation_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(_resolve_user),
) -> ConversationDetailResponse:
    conversation = db.scalar(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user.id,
        )
    )
    if not conversation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    messages = db.scalars(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.id.asc())
    ).all()
    return ConversationDetailResponse(
        **_to_conversation_response(db, conversation).model_dump(),
        messages=[
            MessageResponse(
                id=message.id,
                role=message.role,
                content=message.content,
                model=message.model,
                created_at=message.created_at,
            )
            for message in messages
        ],
    )


@router.get("/{conversation_id}/costs", response_model=ConversationCostBreakdownResponse)
async def get_conversation_costs(
    conversation_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(_resolve_user),
) -> ConversationCostBreakdownResponse:
    conversation = db.scalar(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user.id,
        )
    )
    if not conversation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    llm_prompt_tokens = db.execute(
        select(func.coalesce(func.sum(Message.prompt_tokens), 0)).where(
            Message.conversation_id == conversation.id,
            Message.role == "assistant",
        )
    ).scalar_one()
    llm_completion_tokens = db.execute(
        select(func.coalesce(func.sum(Message.completion_tokens), 0)).where(
            Message.conversation_id == conversation.id,
            Message.role == "assistant",
        )
    ).scalar_one()
    embedding_items, embedding_tokens, embedding_cost = _compute_embedding_cost_breakdown(db, conversation.id)
    llm_cost = float(conversation.total_cost_usd or Decimal("0"))
    return ConversationCostBreakdownResponse(
        conversation_id=str(conversation.id),
        llm={
            "prompt_tokens": int(llm_prompt_tokens or 0),
            "completion_tokens": int(llm_completion_tokens or 0),
            "total_tokens": int(conversation.total_tokens or 0),
            "cost_usd": llm_cost,
        },
        embeddings={
            "total_input_tokens": embedding_tokens,
            "cost_usd": embedding_cost,
            "documents": embedding_items,
        },
        totals={
            "cost_usd": round(llm_cost + embedding_cost, 8),
            "total_tokens": int(conversation.total_tokens or 0) + int(embedding_tokens),
        },
    )


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(_resolve_user),
) -> Response:
    conversation = db.scalar(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user.id,
        )
    )
    if not conversation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    db.execute(delete(Conversation).where(Conversation.id == conversation.id))
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
