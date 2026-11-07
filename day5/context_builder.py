def build_context(documents):
    if not documents:
        return "", {}

    context_parts = []
    source_map = {}

    for idx, doc in enumerate(documents, start=1):
        label = f"S{idx}"
        source = doc.get("source", "unknown")
        text = doc.get("text", "")
        section = doc.get("section", "без раздела")

        context_parts.append(
            f"[{label}]\nИсточник: {source}\nРаздел: {section}\n\nТекст:\n{text}\n"
        )
        source_map[label] = {"source": source, "section": section}

    context = "\n".join(context_parts)

    return context, source_map
