from qdrant_client import QdrantClient, models

client = QdrantClient(
    url="http://localhost:6333",
)

COLLECTION = "hackathon_day1"

if not client.collection_exists(COLLECTION):
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(
            size=3,
            distance=models.Distance.COSINE,
        ),
    )

points = [
    models.PointStruct(
        id=101,
        vector=[1.0, 0.0, 0.0],
        payload={
            "text": "Настройка сертификата",
            "source": "integration.pdf",
            "version": 2,
            "is_active": True,
        },
    ),
    models.PointStruct(
        id=102,
        vector=[0.9, 0.1, 0.0],
        payload={
            "text": "Восстановление доступа",
            "source": "access.pdf",
            "version": 1,
            "is_active": True,
        },
    ),
    models.PointStruct(
        id=103,
        vector=[0.99, 0.01, 0.0],
        payload={
            "text": "Старая инструкция",
            "source": "old_manual.pdf",
            "version": 1,
            "is_active": False,
        },
    ),
    models.PointStruct(
        id=104,
        vector=[0.65, 0.02, 0.2],
        payload={
            "text": "Обновление системы",
            "source": "update.pdf",
            "version": 1,
            "is_active": True,
        }
    )
]

client.upsert(
    collection_name=COLLECTION,
    points=points,
    wait=True,
)

query_vector = [0.98, 0.02, 0.0]

# unused filter

results = client.query_points(
    collection_name=COLLECTION,
    query=query_vector,
    # limit=3,
    with_payload=True,
).points

print("\nUnused filter:")
for point in results:
    print(
        point.id,
        point.score,
        point.payload,
    )

# used filter

results = client.query_points(
    collection_name=COLLECTION,
    query=query_vector,
    query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="is_active",
                match=models.MatchValue(value=True),
            )
        ]
    ),
    # limit=2,
    with_payload=True,
).points

print("\nWith filter:")
for point in results:
    print(
        point.id,
        point.score,
        point.payload,
    )

# Deleted poin on IDs

client.delete(
    collection_name=COLLECTION,
    points_selector=models.PointIdsList(
        points=[1,2,3],
    ),
)

# Deleted poin on filter

client.delete(
    collection_name=COLLECTION,
    points_selector=models.Filter(
        must=[
            models.FieldCondition(
                key="is_active",
                match=models.MatchValue(value=False),
            )
        ],
    ),
)

results = client.query_points(
    collection_name=COLLECTION,
    query=query_vector,
    # limit=3,
    with_payload=True,
).points

print("\nUnused filter(after delete: [1,2,3] IDs, is_active=False):")
for point in results:
    print(
        point.id,
        point.score,
        point.payload,
    )