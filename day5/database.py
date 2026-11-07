import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from qdrant_client import QdrantClient, models


load_dotenv()


EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "")
EMBEDDING_SIZE = int(os.environ.get("EMBEDDING_SIZE", "1024"))
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "")
COLLECTION = os.environ.get("COLLECTION", "")
DENSE_NAME = os.environ.get("DENSE_NAME", "")
TEXT_FIELD = os.environ.get("TEXT_FIELD", "")
DOCUMENTS_JSON = os.environ.get("DOCUMENTS_JSON", "documents.json")
with open(DOCUMENTS_JSON, encoding="utf-8") as f:
    DOCUMENTS = json.load(f)


model = BGEM3FlagModel(EMBEDDING_MODEL, use_fp16=True)
reranker = FlagReranker(RERANKER_MODEL, use_fp16=False, normalize=True)


def _initial_database():
    client = QdrantClient(
        url="http://localhost:6333",
    )

    stemmer = models.SnowballParams(
        type=models.Snowball.SNOWBALL,
        language=models.SnowballLanguage.RUSSIAN,
    )

    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                DENSE_NAME: models.VectorParams(
                    size=EMBEDDING_SIZE, distance=models.Distance.COSINE
                )
            },
        )

        client.create_payload_index(
            collection_name=COLLECTION,
            field_name=TEXT_FIELD,
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.MULTILINGUAL,
                lowercase=True,
                stemmer=stemmer,
                scoring=models.TextScoringParams(
                    type=models.TextScoringType.BM25,
                ),
            ),
            wait=True,
        )

    return client


def get_embedding(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text must be a non-empty string")

    output = model.encode(text)

    return output["dense_vecs"]  # возввращает numpy.ndarray


def _update_database(client):
    texts_to_emb = [doc["text"] for doc in DOCUMENTS]

    embeddings = model.encode(
        texts_to_emb,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )["dense_vecs"]

    points = [
        models.PointStruct(
            id=doc["id"],
            vector={
                DENSE_NAME: embeddings[i].tolist(),
            },
            payload={k: v for k, v in doc.items() if k != "id"},
        )
        for i, doc in enumerate(DOCUMENTS)
    ]

    client.upsert(
        collection_name=COLLECTION,
        points=points,
        wait=True,
    )
    return client


def dense_search(query: str, client, limit: int = 3):
    query_vector = get_embedding(query)

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


def hybrid_search(query: str, client, limit: int = 3):
    query_vector = get_embedding(query)

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


def print_results(method, query, results):
    print(f"\n{method}: {query}")

    for rank, point in enumerate(results, start=1):
        print(f"{rank}. {point.payload['section']} | score={point.score:.4f}")


def reciprocal_rank(results, true_section):
    for rank, point in enumerate(results, start=1):
        if point.payload["section"] == true_section:
            return 1 / rank

    return 0.0


def rerank_results(query: str, candidates, top_k=3):
    if not candidates:
        return []

    pairs = [[query, point.payload["text"]] for point in candidates]

    scores = reranker.compute_score(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    for point, score in ranked:
        point.score = float(score)

    return [point for point, _ in ranked[:top_k]]


def points_to_documents(points):
    return [
        {
            "source": pt.payload.get("source", "unknown"),
            "section": pt.payload.get("section", "без раздела"),
            "text": pt.payload.get("text", ""),
        }
        for pt in points
    ]

