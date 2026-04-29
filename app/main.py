from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from app.api.v1.router import api_router
from app.core.config import settings
from app.db.base import Base
from app.db.session import engine
import app.db.models  # noqa: F401

from app.embeddings.factory import resolve_embedding_config

logger = logging.getLogger("uvicorn.error")


def _log_runtime_warmup_config() -> None:
    embedding_provider, embedding_model = resolve_embedding_config(None)
    logger.info(
        "Warmup config | llm_provider=%s llm_model=%s embedding_provider=%s embedding_model=%s",
        settings.default_provider,
        settings.default_model,
        embedding_provider,
        embedding_model,
    )


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Prototype bootstrap: create tables automatically.
    Base.metadata.create_all(bind=engine)
    _log_runtime_warmup_config()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
