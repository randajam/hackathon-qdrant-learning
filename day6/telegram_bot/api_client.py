from pydantic import BaseModel

class ApiSource(BaseModel):
    label: str
    source: str
    section: str


class ApiQueryResponse(BaseModel):
    answer: str
    sources: list[ApiSource]


class RagApiClient:

    def __init__(self, http_client):
        self.http_client = http_client

    async def ask(self, query: str) -> ApiQueryResponse:
        response = await self.http_client.post(
            "/api/v1/query",
            json={"query": query},
        )
        response.raise_for_status()
        data = response.json()
        return ApiQueryResponse.model_validate(data)