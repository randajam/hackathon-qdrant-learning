from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=200)


class Source(BaseModel):
    label: str
    source: str
    section: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]