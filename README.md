# Corporate RAG Telegram Assistant

Учебный прототип корпоративного Telegram-ассистента, который отвечает на вопросы по проектной документации. Поиск выполняется через Qdrant, а ответ формируется GigaChat на основании найденных фрагментов с указанием источников.

Проект демонстрирует полный базовый RAG-процесс: **Hybrid Retrieval → Reranking → Context Building → LLM Generation**, а также разделение Telegram-клиента и Backend через HTTP API.

> **Статус:** учебный прототип. Перед использованием с реальной корпоративной документацией необходимы дополнительные меры безопасности, тестирование на полноценном корпусе и настройка production-развёртывания.

## Возможности

- **Dense retrieval:** семантический поиск по эмбеддингам `BAAI/bge-m3`.
- **BM25:** полнотекстовый поиск Qdrant по полю `text`; для русского языка может использоваться Snowball stemming.
- **Hybrid search:** объединение Dense- и BM25-выдач через Reciprocal Rank Fusion (RRF).
- **Reranking:** повторное ранжирование кандидатов моделью `BAAI/bge-reranker-v2-m3`.
- **RAG generation:** формирование ответа через GigaChat на основе переданного контекста.
- **Grounding и citations:** ответ сопровождается ссылками на найденные источники (`[S1]`, `[S2]` и т. д.).
- **Abstention:** если подходящей информации нет, ассистент должен сообщить, что ответ не найден в документации.
- **FastAPI Backend:** HTTP API для обработки пользовательских вопросов.
- **Telegram Bot:** интерфейс на aiogram 3, который обращается к Backend по HTTP.

## Архитектура

```text
Telegram user
     |
     v
aiogram Telegram Bot
     |
     | HTTP POST /api/v1/query
     v
FastAPI Backend
     |
     v
RAGService
     |
     +--> BGE-M3 embeddings
     |
     +--> Qdrant: Dense + BM25 + RRF
     |
     +--> BGE Reranker
     |
     +--> Context Builder
     |
     +--> GigaChat API
     |
     v
Answer + Sources
     |
     v
Telegram user
```

### Компоненты

| Компонент | Назначение |
|---|---|
| `telegram_bot` | Получает сообщения Telegram и вызывает Backend API |
| `FastAPI` | Принимает HTTP-запросы, проверяет данные и возвращает результат |
| `RAGService` | Координирует поиск, reranking и генерацию |
| `Qdrant` | Хранит векторы, полнотекстовый индекс и payload документов |
| `BGE-M3` | Преобразует запросы в dense-векторы |
| `BGE Reranker` | Оценивает пары «запрос — документ» и переставляет кандидатов |
| `GigaChat` | Формирует ответ по переданному контексту |

BGE-M3 и Reranker загружаются при старте FastAPI-приложения через `lifespan`, а не при каждом запросе. Индексация корпуса должна выполняться отдельно от обработки пользовательских запросов.

## Структура проекта

Основные модули проекта:

```text
.
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── routes/
│   │   │   └── query.py
│   │   └── schemas/
│   │       └── query.py
│   ├── core/
│   │   └── config.py
│   ├── retrieval/
│   │   ├── embeddings.py
│   │   ├── qdrant_store.py
│   │   └── reranker.py
│   ├── generation/
│   │   ├── context_builder.py
│   │   ├── gigachat_client.py
│   │   └── prompts.py
│   └── services/
│       └── rag_service.py
│
├── telegram_bot/
│   ├── main.py
│   ├── config.py
│   ├── api_client.py
│   └── handlers/
│       └── messages.py
│
├── documents.json          # Учебный корпус, если хранится в репозитории
├── .env.example             # Шаблон переменных окружения (без секретов)
├── .gitignore
└── README.md
```

Вспомогательные файлы (тесты, скрипты индексации и manifests зависимостей) могут находиться в отдельных каталогах.

## Технологический стек

- Python 3.11
- [FastAPI](https://fastapi.tiangolo.com/) и Uvicorn
- [aiogram 3](https://docs.aiogram.dev/en/latest/)
- [HTTPX](https://www.python-httpx.org/)
- [Qdrant](https://qdrant.tech/documentation/)
- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
- `BAAI/bge-m3`
- `BAAI/bge-reranker-v2-m3`
- [GigaChat Python SDK](https://github.com/ai-forever/gigachat)
- Pydantic и python-dotenv

## Требования

1. Python 3.11.
2. Доступный экземпляр Qdrant.
3. Учётные данные GigaChat API.
4. Telegram Bot Token, полученный у [@BotFather](https://t.me/BotFather).
5. Доступ к Hugging Face/источнику моделей при первом скачивании весов BGE-M3 и Reranker.
6. Для работы Telegram API из сетей, где прямое соединение недоступно, — локальный SOCKS5/HTTP-прокси.

## Настройка окружения

Создайте виртуальное окружение и установите зависимости проекта. Пример установки основных библиотек:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Backend
pip install fastapi "uvicorn[standard]" qdrant-client FlagEmbedding gigachat python-dotenv

# Telegram Bot
pip install aiogram httpx aiohttp-socks
```

Если в репозитории есть `requirements`-файлы, предпочтительно устанавливать зафиксированные зависимости из них.

Создайте `.env` в корне проекта. Пример `.env.example`:

```dotenv
# Qdrant
QDRANT_URL=http://127.0.0.1:6333
COLLECTION=hackathon_day5
DENSE_NAME=dense
TEXT_FIELD=text

# Embedding and reranking models
EMBEDDING_MODEL=BAAI/bge-m3
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# GigaChat
GIGACHAT_CREDENTIALS=replace_with_your_credentials
GIGACHAT_SCOPE=GIGACHAT_API_PERS
GIGACHAT_BASE_URL=https://api.giga.chat/v1
GIGACHAT_MODEL=GigaChat-2

# Telegram Bot
BOT_TOKEN=replace_with_your_telegram_bot_token
BACKEND_BASE_URL=http://127.0.0.1:8000
```

Значение `COLLECTION` должно совпадать с именем уже созданной и заполненной коллекции Qdrant. При необходимости измените его под свою конфигурацию.

**Не коммитьте `.env` и реальные токены/ключи.** В Git должен попасть только `.env.example` с фиктивными значениями.

### Запуск Qdrant

Если Qdrant уже запущен локально и доступен по `QDRANT_URL`, этот шаг можно пропустить.

Для локального запуска отдельного экземпляра через Docker:

```bash
docker run --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant
```

Проверьте доступность сервера:

```bash
curl http://127.0.0.1:6333/collections
```

Перед запуском Backend коллекция должна быть создана и заполнена документами. Она должна содержать именованный dense-вектор, текстовое поле `text` и payload-поле `is_active`. Для BM25 должен быть настроен полнотекстовый индекс с соответствующей конфигурацией scoring/tokenizer/stemmer.

> Создание коллекции и ingestion документов — отдельная операция. Не запускайте пересчёт эмбеддингов всего корпуса при каждом пользовательском запросе.

## Запуск приложений

Backend и Telegram Bot запускаются как два отдельных процесса. Оба должны оставаться запущенными одновременно.

### 1. FastAPI Backend

Из корня проекта:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

После запуска документация API доступна по адресу:

- Swagger UI: http://127.0.0.1:8000/docs
- OpenAPI schema: http://127.0.0.1:8000/openapi.json

### 2. Проверка Backend без Telegram

```bash
curl -X POST http://127.0.0.1:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Как восстановить пароль?"}'
```

Пример структуры ответа:

```json
{
  "answer": "Для восстановления пароля перейдите на страницу авторизации... [S1]",
  "sources": [
    {
      "label": "S1",
      "source": "access.pdf",
      "section": "Восстановление доступа"
    }
  ]
}
```

### 3. Telegram Bot

Во втором терминале, также из корня проекта:

```bash
python -m telegram_bot.main
```

Откройте Telegram, найдите бота по username, полученному от BotFather, нажмите **Start** или отправьте `/start`, затем задайте вопрос по документации.

Бот отправляет запрос на `BACKEND_BASE_URL`, получает JSON-ответ и показывает текст вместе с источниками.

### Прокси для Telegram API

В локальной конфигурации для доступа к Telegram может использоваться SOCKS5-прокси Happ/Xray:

```python
AiohttpSession(proxy="socks5://127.0.0.1:10808")
```

Этот адрес является примером локальной конфигурации. Если прокси не нужен, используйте обычную сессию aiogram без `proxy`. Если порт или адрес прокси отличаются, укажите фактические значения. Прокси Telegram-клиента не требуется для локального HTTP-вызова FastAPI.

## API-контракт

### `POST /api/v1/query`

Запрос:

```json
{
  "query": "Как восстановить пароль?"
}
```

Ограничения текущей схемы: строка `query` должна содержать от 1 до 200 символов.

Ответ:

```json
{
  "answer": "Текст ответа с цитатами [S1]",
  "sources": [
    {
      "label": "S1",
      "source": "access.pdf",
      "section": "Восстановление доступа"
    }
  ]
}
```

## Принцип работы поиска

1. BGE-M3 преобразует пользовательский запрос в dense-вектор.
2. Qdrant выполняет Dense и полнотекстовый BM25-поиск по активным документам (`is_active=true`).
3. Qdrant объединяет результаты через RRF.
4. Cross-Encoder `bge-reranker-v2-m3` повторно оценивает пары «запрос — документ» и формирует итоговый список кандидатов.
5. Context Builder собирает тексты и метаданные источников, присваивая им метки `[S1]`, `[S2]` и т. д.
6. GigaChat получает вопрос и контекст. Системные инструкции ограничивают ответ информацией из документов и предписывают отказываться от неподтверждённых утверждений.
7. Backend возвращает ответ и список источников.

## Ограничения текущей версии

- Проект является прототипом и не заявляется как production-ready система.
- Качество поиска зависит от состава документов, разбиения на фрагменты, настроек индексов и качества тестовой выборки.
- Reranker увеличивает задержку; на CPU-инференсе время может быть существенным.
- Промпт и проверка меток источников не гарантируют абсолютную защиту от prompt injection и фактических ошибок LLM.
- Текущая схема API ограничивает запрос 200 символами.
- В текущей версии нет полноценной авторизации пользователей, rate limiting, истории диалога и комплексного мониторинга.
- FastAPI и Telegram Bot запускаются отдельными Python-процессами; Docker Compose для приложения в описанном локальном сценарии не используется.
- Наличие найденных кандидатов само по себе не означает, что они действительно отвечают на вопрос; для надёжного отказа нужны отдельные проверки релевантности и тесты.

## Устранение неполадок

- **Telegram Bot не проходит `get_me()` или не запускает polling.** Проверьте доступность `api.telegram.org`. Если прямое соединение недоступно, настройте SOCKS5/HTTP-прокси в `AiohttpSession`. Для SOCKS5 через aiohttp требуется `aiohttp-socks`.
- **Бот не отвечает, а Backend недоступен.** Убедитесь, что FastAPI запущен и `BACKEND_BASE_URL` указывает на правильный адрес. Локально это обычно `http://127.0.0.1:8000`.
- **Процесс бота завершился.** Long Polling не должен завершаться из-за отсутствия пользовательских сообщений. Таймаут `httpx.Timeout` ограничивает ожидание отдельного HTTP-запроса от бота к Backend, а не время простоя самого бота. Проверьте traceback и логи процесса.
- **Qdrant возвращает ошибку подключения.** Проверьте `QDRANT_URL` и доступность `http://127.0.0.1:6333/collections`.

## Безопасность перед публикацией

- Убедитесь, что `.env`, токены Telegram и уч