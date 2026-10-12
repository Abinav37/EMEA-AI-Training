from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = (PROJECT_ROOT / ".env").resolve()
load_dotenv(dotenv_path=str(ENV_PATH), override=True)


class Settings:
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "").strip()
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "openrouter/free").strip() or "openrouter/free"


settings = Settings()
