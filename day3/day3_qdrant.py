import time

from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient, models

client = QdrantClient(
    url="http://localhost:6333",
)
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

COLLECTION = "hackathon_day3"
DENSE_NAME = "dense"
TEXT_FIELD = "text"

DOCUMENTS = [
    {
        "id": 1,
        "text": (
            "Для восстановления пароля пользователь должен "
            "перейти на страницу авторизации и выбрать "
            "«Забыли пароль?». На корпоративную почту "
            "будет отправлена ссылка для восстановления."
        ),
        "source": "access.pdf",
        "section": "Восстановление доступа",
        "is_active": True,
    },
    {
        "id": 2,
        "text": (
            "Для настройки интеграции с ЕГИСЗ откройте "
            "раздел Администрирование — Интеграции, "
            "выберите модуль ЕГИСЗ и укажите путь "
            "к сертификату."
        ),
        "source": "integration.pdf",
        "section": "Настройка ЕГИСЗ",
        "is_active": True,
    },
    {
        "id": 3,
        "text": (
            "Если приложение не запускается после обновления, "
            "очистите локальный кэш и перезапустите службу."
        ),
        "source": "support.pdf",
        "section": "Ошибки после обновления",
        "is_active": True,
    },
    {
        "id": 4,
        "text": (
            "Для изменения прав доступа откройте карточку "
            "пользователя и назначьте необходимую роль. "
            "Изменения вступают в силу после повторного входа."
        ),
        "source": "access.pdf",
        "section": "Управление ролями",
        "is_active": True,
    },
    {
        "id": 5,
        "text": (
            "При ошибке подключения проверьте срок "
            "действия сертификата и доступность сервера."
        ),
        "source": "integration.pdf",
        "section": "Диагностика подключения",
        "is_active": True,
    },
    {
        "id": 6,
        "text": (
            "Для настройки модуля ликвидности товара "
            "введите EGISZ-API-v2 в специальное поле."
        ),
        "source": "identification.pdf",
        "section": "Идентификаторы",
        "is_active": True,
    },
]

GROUP_QUERIES = {
    "Восстановление доступа" : ["Забыл пароль", "Не могу войти", "Как вернуть доступ к аккаунту?", "Как найти процедуру восстановления авторизационных данных?"],
    "Настройка ЕГИСЗ" : ["Где прописать сертификат?", "Как подключить ЕГИСЗ?"],
    "Управление ролями" : ["Как выдать права сотруднику?", "Как изменить роль пользователя?", "Кто может менять полномочия?"],
    "Диагностика подключения" : [ "Почему истёкший сертификат мешает соединению?"],
    "Идентификаторы" : ["Где указан идентификатор EGISZ-API-v2?"],
}

def get_embedding(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text must be a non-empty string")

    output = model.encode(text)

    return output["dense_vecs"]  # возввращает numpy.ndarray


def dense_search(query_vector: str, limit: int = 3):
    # query_vector = get_embedding(query)

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


def bm25_search(query: str, limit: int = 3):
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


def hybrid_search(query: str, query_vector: str, limit: int = 3):
    # query_vector = get_embedding(query)

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
        print(
            f"{rank}. "
            f"{point.payload['section']} | "
            f"score={point.score:.4f}"
        )


def reciprocal_rank(results, true_section):
    for rank, point in enumerate(results, start=1):
        if point.payload["section"] == true_section:
            return 1 / rank

    return 0.0



if __name__=="__main__":
    stemmer=models.SnowballParams(
        type=models.Snowball.SNOWBALL,
        language=models.SnowballLanguage.RUSSIAN,
    )

    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                DENSE_NAME: models.VectorParams(
                    size=1024,
                    distance=models.Distance.COSINE,
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
            payload={
                k: v
                for k, v in doc.items()
                if k != "id"
            },
        )
        for i, doc in enumerate(DOCUMENTS)
    ]

    client.upsert(
        collection_name=COLLECTION,
        points=points,
        wait=True,
    )

    timings = {"DENSE": [], "BM25": [], "HYBRID": [], "EMBEDDING" : []}
    recall3 = {"DENSE": 0, "BM25": 0, "HYBRID": 0}
    mmr3 = {"DENSE": 0, "BM25": 0, "HYBRID": 0}
    for true_section, queries in GROUP_QUERIES.items():
        for query in queries:
            start = time.perf_counter()
            emb_query = get_embedding(query)
            timings["EMBEDDING"].append(time.perf_counter() - start)


            start = time.perf_counter()
            dense_result = dense_search(emb_query)
            timings["DENSE"].append(time.perf_counter() - start)
            print_results(
                method="DENSE",
                query=query,
                results=dense_result,
            )
            sections = [point.payload['section'] for point in dense_result]
            if true_section in sections:
                recall3["DENSE"] += 1
            mmr3["DENSE"] += reciprocal_rank(dense_result, true_section)


            start = time.perf_counter()
            bm25_result = bm25_search(query)
            timings["BM25"].append(time.perf_counter() - start)
            print_results(
                method="BM25",
                query=query,
                results=bm25_result,
            )
            sections = [point.payload['section'] for point in bm25_result]
            if true_section in sections:
                recall3["BM25"] += 1
            mmr3["BM25"] += reciprocal_rank(bm25_result, true_section)

            start = time.perf_counter()
            hybrid_result = hybrid_search(query, emb_query)
            timings["HYBRID"].append(time.perf_counter() - start)
            print_results(
                method="HYBRID",
                query=query,
                results=hybrid_result,
            )
            sections = [point.payload['section'] for point in hybrid_result]
            if true_section in sections:
                recall3["HYBRID"] += 1
            mmr3["HYBRID"] += reciprocal_rank(hybrid_result, true_section)

    total_queries = sum(
        len(queries)
        for queries in GROUP_QUERIES.values()
    )

    recall3 = {k: i / total_queries for k,i in recall3.items()}
    mrr3 = {k: v / total_queries for k, v in mmr3.items()}

    print("\n=== Recall3 ===")
    for method, r3 in recall3.items():
        print(f"{method:8} recalll3={r3:5.2}")

    print("\n=== MRR ===")
    for method, r3 in mrr3.items():
        print(f"{method:8} mrr3={r3:5.2}")

    # Итоговая статистика
    print("\n=== Среднее время выполнения (мс) ===")
    for method, times in timings.items():
        avg_ms = sum(times) / len(times) * 1000
        min_ms = min(times) * 1000
        max_ms = max(times) * 1000
        print(f"{method:8} avg={avg_ms:7.2f}  min={min_ms:7.2f}  max={max_ms:7.2f}  (n={len(times)})")