from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient, models

client = QdrantClient(
    url="http://localhost:6333",
)

COLLECTION = "hackathon_day2"

# Инициализация (при первом запуске автоматически скачает модель)
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

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
]


def get_embedding(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text must be a non-empty string")

    output = model.encode(text)

    return output["dense_vecs"]  # возввращает numpy.ndarray


def search_documents(query, limit=3):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Text must be a non-empty string")

    query_vec = get_embedding(query)
    answer = client.query_points(
        collection_name=COLLECTION,
        query=query_vec,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="is_active",
                    match=models.MatchValue(value=True),
                )
            ],
        ),
        limit=limit,
        with_payload=True,
    ).points

    return [
        {"id": point.id, "score": point.score, "payload": point.payload}
        for point in answer
    ]


if __name__ == "__main__":

    texts_to_emb = [doc["text"] for doc in DOCUMENTS]
    embeddings = model.encode(texts_to_emb)["dense_vecs"]

    points = [
        models.PointStruct(
            id=doc["id"],
            vector=embeddings[i].tolist(),
            payload={k: v for k, v in doc.items() if k != "id"},
        )
        for i, doc in enumerate(DOCUMENTS)
    ]

    emb_size = embeddings.shape[1]

    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=emb_size,
                distance=models.Distance.COSINE,
            ),
        )

    client.upsert(
        collection_name=COLLECTION,
        points=points,
    )

    queries = [
        "Как мне снова попасть в личный кабинет, если я забыл пароль?",
        "Где прописать сертификат для подключения к ЕГИСЗ?",
        "После установки новой версии программа перестала запускаться",
        "Как предоставить сотруднику дополнительные полномочия?",
        "Не устанавливается соединение с сервером",
        "Как приготовить плов?",
    ]

    for query in queries:
        answer = search_documents(query, limit=1)
        if not answer:
            print("Document is not found.")
        else:
            print(
                f"Query: {query}\nSection (top-1): {answer[0]['payload']['section']}\nScore: {answer[0]['score']}\n"
            )

    group_queries = {
        "Восстановление доступа" : ["Забыл пароль", "Не могу войти", "Как вернуть доступ к аккаунту?"],
        "Настройка ЕГИСЗ" : ["Где прописать сертификат?", "Как подключить ЕГИСЗ?", "Почему истёкший сертификат мешает соединению?"],
        "Управление ролями" : ["Как выдать права сотруднику?", "Как изменить роль пользователя?", "Кто может менять полномочия?"],
    }

    recall3 = 0
    for true_section, qs in group_queries.items():
        print(f"\nSection: {true_section}:")

        for query in qs:
            answer = search_documents(query, limit=3)
            if not answer:
                print("Documents is not found.")
            else:
                print(
                    f"Query: {query}\n\
                    Section (top-1): {answer[0]['payload']['section']}\n\
                    Score: {answer[0]['score']}\n\
                    Section (top-1): {answer[1]['payload']['section']}\n\
                    Score: {answer[1]['score']}\n\
                    Section (top-1): {answer[2]['payload']['section']}\n\
                    Score: {answer[2]['score']}\n"
                )
                sections = [x['payload']['section'] for x in answer]
                if true_section in sections:
                    recall3 += 1
    recall3 /= 9
    print(f"Recall3: {recall3}")