import os
import re

from context_builder import build_context
from dotenv import load_dotenv
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from prompts import get_system_message

load_dotenv()

GIGACHAT_CREDENTIALS = os.environ.get("GIGACHAT_CREDENTIALS")
GIGACHAT_SCOPE = os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
GIGACHAT_BASE_URL = os.environ.get("GIGACHAT_BASE_URL", "https://api.giga.chat/v1")
GIGACHAT_MODEL = os.environ.get("GIGACHAT_MODEL", "GigaChat-2")


def _get_client() -> GigaChat:
    if not GIGACHAT_CREDENTIALS:
        raise ValueError("Установите GIGACHAT_CREDENTIALS в .env")
    return GigaChat(
        base_url=GIGACHAT_BASE_URL,
        credentials=GIGACHAT_CREDENTIALS,
        scope=GIGACHAT_SCOPE,
        verify_ssl_certs=False,
    )


def generate_answer(query: str, documents: list) -> tuple[str, dict]:
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

    client = _get_client()
    resp = client.chat(chat)

    answer = resp.choices[0].message.content.strip()

    cited = set(re.findall(r"\[(S\d+)\]", answer))
    valid = set(source_map.keys())
    if cited - valid:
        answer += f"\n!!! Несуществующие ссылки: {cited - valid}"

    return answer, source_map
