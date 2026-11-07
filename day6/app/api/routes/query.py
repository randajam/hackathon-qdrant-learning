from app.api.schemas.query import QueryRequest, QueryResponse
from fastapi import APIRouter, Request

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest, http_request: Request) -> QueryResponse:
    rag_service = http_request.app.state.rag_service
    return rag_service.answer(request.query)