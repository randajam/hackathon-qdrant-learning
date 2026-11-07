import argparse

import database
from generation import generate_answer

if __name__ == "__main__":
    # User query
    # parser = argparse.ArgumentParser(description="Получение ответа по запросу")
    # parser.add_argument("--query", required=True, help="Запрос пользователя")
    # args = parser.parse_args()

    # query = args.query
    queries = [
        "Как восстановить пароль?",
        "Где указать путь к сертификату ЕГИСЗ?",
        "Как настроить ЕГИСЗ и что проверить, если соединение не устанавливается?",
        "Какую именно роль нужно назначить сотруднику и когда изменения вступят в силу?",
        "Как изменить корпоративную почту сотрудника?",
    ]

    # Retrieval + Reranking
    client = database._initial_database()
    database._update_database(client)

    for query in queries:
        print("\n\n")
        hybrid_result = database.hybrid_search(
            query=query,
            client=client,
            limit=10,
        )

        rerank_result = database.rerank_results(
            query=query,
            candidates=hybrid_result,
            top_k=3,
        )

        # Context Builder
        documents = database.points_to_documents(rerank_result)

        # LLM generation
        answer_text, source_map = generate_answer(query=query, documents=documents)

        # Answer + Source
        print(f"Answer:\n{answer_text}\n")
        print("Sources:")
        for label, meta in source_map.items():
            print(f"  [{label}] {meta['source']} — {meta['section']}")
