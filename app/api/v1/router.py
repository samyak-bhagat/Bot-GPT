from fastapi import APIRouter

from app.api.v1.routes.conversations import router as conversations_router
from app.api.v1.routes.documents import router as documents_router

api_router = APIRouter()
api_router.include_router(conversations_router, prefix="/conversations", tags=["conversations"])
api_router.include_router(documents_router, prefix="/documents", tags=["documents"])
