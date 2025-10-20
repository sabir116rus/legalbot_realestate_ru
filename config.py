from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    telegram_bot_token: str
    openai_api_key: str
    openai_model: str
    rag_top_k: int
    knowledge_base_path: Path
    system_prompt: str
    log_path: Path

    @classmethod
    def load(cls) -> "Config":
        load_dotenv()

        telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        rag_top_k = int(os.getenv("RAG_TOP_K", "3"))

        if not telegram_bot_token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
        if not openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        base_dir = Path(__file__).resolve().parent
        data_dir = base_dir / "data"
        knowledge_base_path = data_dir / "knowledge.csv"
        log_path = data_dir / "log.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        system_prompt_path = base_dir / "prompt_system_ru.txt"
        system_prompt = system_prompt_path.read_text(encoding="utf-8")

        return cls(
            telegram_bot_token=telegram_bot_token,
            openai_api_key=openai_api_key,
            openai_model=openai_model,
            rag_top_k=rag_top_k,
            knowledge_base_path=knowledge_base_path,
            system_prompt=system_prompt,
            log_path=log_path,
        )
