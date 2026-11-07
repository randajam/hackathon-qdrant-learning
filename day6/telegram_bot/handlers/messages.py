import logging
import httpx

from aiogram import F, Router
from aiogram.filters import CommandStart
from aiogram.types import Message

from telegram_bot.api_client import RagApiClient

router = Router()
logger = logging.getLogger(__name__)

MAX_QUERY_LEN = 200


@router.message(CommandStart())
async def start_handler(message: Message):
    await message.answer(
        "Привет! Я корпоративный ассистент. "
        "Задай вопрос по документации."
    )


@router.message(F.text.startswith("/"))
async def unknown_command_handler(message: Message):
    await message.answer(
        "Неизвестная команда. Используйте /start."
    )


@router.message(F.text)
async def text_handler(
    message: Message,
    rag_api: RagApiClient
):
    query = message.text.strip()

    if not query:
        await message.answer(
            "Пустой запрос. Напишите вопрос по документации."
        )
        return

    if len(query) > MAX_QUERY_LEN:
        await message.answer(
            f"Вопрос слишком длинный ({len(query)} символов). "
            f"Максимум — {MAX_QUERY_LEN}."
        )
        return

    processing_message = await message.answer(
        "⏳ Обрабатываю запрос..."
    )

    try:
        result = await rag_api.ask(query)

    except httpx.TimeoutException:
        logger.exception("Backend request timed out")
        await processing_message.edit_text(
            "Сервис отвечает слишком долго. Попробуйте позже."
        )
        return

    except httpx.HTTPStatusError as exc:
        logger.exception(
            "Backend returned HTTP %s",
            exc.response.status_code
        )
        await processing_message.edit_text(
            "Не удалось обработать запрос. Попробуйте позже."
        )
        return

    except httpx.RequestError:
        logger.exception("Backend connection error")
        await processing_message.edit_text(
            "Backend сейчас недоступен. Попробуйте позже."
        )
        return

    parts = [result.answer]

    if result.sources:
        parts.extend(["", "Источники:"])

        for source in result.sources:
            parts.append(
                f"• [{source.label}] "
                f"{source.source} — {source.section}"
            )

    await processing_message.edit_text(
        "\n".join(parts)
    )