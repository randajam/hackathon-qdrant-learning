def rerank_results(reranker, query: str, candidates, top_k=3):
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

