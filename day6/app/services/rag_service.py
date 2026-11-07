from app.api.schemas.query import QueryResponse, Source
from app.retrieval.reranker import points_to_documents, rerank_results


class RAGService:
    def __init__(self, search_fn, generate_fn, client, model, rerank_fn, llm_client):
        self.search_fn = search_fn
        self.generate_fn = generate_fn
        self.client = client
        self.model = model
        self.rerank_fn = rerank_fn
        self.llm_client = llm_client

    def answer(self, query: str) -> QueryResponse:
        points = self.search_fn(query=query, client=self.client, model=self.model)
        if self.rerank_fn:
            points = rerank_results(reranker=self.rerank_fn, query=query, candidates=points, top_k=3)
    
        documents = points_to_documents(points)

        answer_text, source_map = self.generate_fn(self.llm_client, query, documents)
        sources = [
            Source(label=k, source=v["source"], section=v["section"])
            for k, v in source_map.items()
        ]
        return QueryResponse(answer=answer_text, sources=sources)
