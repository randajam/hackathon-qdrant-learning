from contextlib import asynccontextmanager

from fastapi import FastAPI
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from qdrant_client import QdrantClient

from app.api.routes.query import router as query_router
from app.core.config import settings
from app.generation.gigachat_client import generate_answer, get_client
from app.retrieval.qdrant_store import hybrid_search
from app.services.rag_service import RAGService


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.reranker = FlagReranker(settings.reranker_model, use_fp16=False)
    app.state.model = BGEM3FlagModel(
        settings.embedding_model, use_fp16=False, normalize=True
    )
    app.state.client = QdrantClient(url=settings.qdrant_url)
    app.state.llm_client = get_client()
    app.state.rag_service = RAGService(
        search_fn=hybrid_search,
        generate_fn=generate_answer,
        client=app.state.client,
        model=app.state.model,
        rerank_model=app.state.reranker,
        llm_client=app.state.llm_client,
    )
    yield
    app.state.client.close()
    app.state.llm_client.close()

app = FastAPI(lifespan=lifespan)
app.include_router(query_router, prefix="/api/v1")
