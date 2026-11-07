import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    collection: str = os.environ.get("COLLECTION", "")
    dense_name: str = os.environ.get("DENSE_NAME", "")
    text_field: str = os.environ.get("TEXT_FIELD", "")
    qdrant_url: str = os.environ.get("QDRANT_URL", "http://localhost:6333")
    embedding_model: str = os.environ.get("EMBEDDING_MODEL", "")
    reranker_model: str = os.environ.get("RERANKER_MODEL", "")
    gigachat_credentials: str = os.environ.get("GIGACHAT_CREDENTIALS", "")
    gigachat_scope: str = os.environ.get("GIGACHAT_SCOPE", "")
    gigachat_base_url: str = os.environ.get("GIGACHAT_BASE_URL", "")
    gigachat_model: str = os.environ.get("GIGACHAT_MODEL", "")

settings = Settings()