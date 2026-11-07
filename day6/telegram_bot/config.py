import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    bot_token: str = os.environ.get("BOT_TOKEN", "")
    backend_base_url: str = os.environ.get("BACKEND_BASE_URL", "")

settings = Settings()