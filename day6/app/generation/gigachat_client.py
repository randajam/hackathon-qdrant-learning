import re

from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

from app.core.config import settings
from app.generation.context_builder import build_context
from app.generation.prompts import get_system_message

GIGACHAT_CREDENTIALS = settings.gigachat_credentials
GIGACHAT_SCOPE = settings.gigachat_scope
GIGACHAT_BASE_URL = settings.gigachat_base_url
GIGACHAT_MODEL = settings.gigachat_model


def get_client() -> GigaChat:
    if not GIGACHAT_CREDENTIALS:
        raise ValueError("Установите GIGACHAT_CREDENTIALS в .env")
    return GigaChat(
        base_url=GIGACHAT_BASE_URL,
        credentials=GIGACHAT_CREDENTIALS,
        scope=GIGACHAT_SCOPE,
        verify_ssl_certs=False,
    )


def generate_answer(client: GigaChat, query: str, documents: list) -> tuple[str, dict]:
    if not documents:
        return "По предоставленным документам ответ сформировать не удалось.", {}

    context, source_map = build_context(documents)

    system_msg = get_system_message()["content"]
    user_msg = f"Контекст:\n{context}\n\nВопрос:\n{query}"

    chat = Chat(
        model=GIGACHAT_MODEL,
        messages=[
            Messages(role=MessagesRole.SYSTEM, content=system_msg),
            Messages(role=MessagesRole.USER, content=user_msg),
        ],
        temperature=0.1,
        max_tokens=500,
    )

    resp = client.chat(chat)

    answer = resp.choices[0].message.content.strip()

    cited = set(re.findall(r"\[(S\d+)\]", answer))
    valid = set(source_map.keys())
    if cited - valid:
        answer += f"\n!!! Несуществующие ссылки: {cited - valid}"

    return answer, source_map
