from qdrant_client import models

from app.core.config import settings
from app.retrieval.embeddings import get_embedding

COLLECTION = settings.collection
DENSE_NAME = settings.dense_name
TEXT_FIELD = settings.text_field


def dense_search(query: str, client, model, limit: int = 3):
    query_vector = get_embedding(query, model)

    results = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        using=DENSE_NAME,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="is_active",
                    match=models.MatchValue(value=True),
                )
            ]
        ),
        limit=limit,
        with_payload=True,
    ).points

    return results


def bm25_search(query: str, client, limit: int = 3):
    results = client.query_points(
        collection_name=COLLECTION,
        query=models.TextQuery(text=query),
        using=TEXT_FIELD,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="is_active",
                    match=models.MatchValue(value=True),
                )
            ]
        ),
        limit=limit,
        with_payload=True,
    ).points

    return results


def hybrid_search(query: str, client, model, limit: int = 3):
    query_vector = get_embedding(query, model)

    active_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="is_active",
                match=models.MatchValue(value=True),
            )
        ]
    )

    results = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(
                query=query_vector,
                using=DENSE_NAME,
                filter=active_filter,
                limit=10,
            ),
            models.Prefetch(
                query=models.TextQuery(text=query),
                using=TEXT_FIELD,
                filter=active_filter,
                limit=10,
            ),
        ],
        query=models.FusionQuery(
            fusion=models.Fusion.RRF,
        ),
        limit=limit,
        with_payload=True,
    ).points

    return results
