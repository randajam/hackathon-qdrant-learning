def get_embedding(text, model):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text must be a non-empty string")

    output = model.encode(text)

    return output["dense_vecs"]