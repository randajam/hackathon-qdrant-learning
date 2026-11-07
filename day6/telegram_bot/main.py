import asyncio
import logging
import httpx

from aiogram import Bot, Dispatcher

from telegram_bot.config import settings
from telegram_bot.api_client import RagApiClient
from telegram_bot.handlers.messages import router
from aiogram.client.session.aiohttp import AiohttpSession

session = AiohttpSession(
    proxy="socks5://127.0.0.1:10808"
)

bot = Bot(
    token=settings.bot_token,
    session=session,
)

async def main():
    logging.basicConfig(
        level=logging.DEBUG,
        force=True,
    )

    print("[1] main() запущен", flush=True)

    print(
        f"[2] BOT_TOKEN загружен: {bool(settings.bot_token)}",
        flush=True,
    )

    print(
        f"[3] BACKEND_BASE_URL: {settings.backend_base_url}",
        flush=True,
    )

    if not settings.bot_token:
        raise RuntimeError("BOT_TOKEN не загружен из .env")

    # bot = Bot(token=settings.bot_token)
    print("[4] Объект Bot создан", flush=True)

    try:
        print("[5] Проверяю соединение с Telegram...", flush=True)

        me = await asyncio.wait_for(
            bot.get_me(),
            timeout=15,
        )

        print(
            f"[6] Бот успешно авторизован: @{me.username}",
            flush=True,
        )

        dp = Dispatcher()
        dp.include_router(router)

        timeout = httpx.Timeout(
            60.0,
            connect=5.0,
        )

        async with httpx.AsyncClient(
            base_url=settings.backend_base_url,
            timeout=timeout,
        ) as http_client:

            rag_api = RagApiClient(http_client)

            print("[7] Запускаю Long Polling...", flush=True)

            await dp.start_polling(
                bot,
                rag_api=rag_api,
                tasks_concurrency_limit=1,
                close_bot_session=False,
            )

    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())