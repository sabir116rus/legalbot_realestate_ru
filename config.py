from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
    consultation_log_path: Path
    consent_store_path: Path
    privacy_policy_message: str
    privacy_policy_webapp_url: str
    google_drive_credentials_file: Optional[Path]
    google_drive_logs_folder_id: Optional[str]
    google_drive_consultations_folder_id: Optional[str]
    google_drive_reports_folder_id: Optional[str]
    google_drive_logs_file_id: Optional[str]
    google_drive_consultations_file_id: Optional[str]
    google_drive_reports_file_id: Optional[str]

    @classmethod
    def load(cls, *, allow_missing: bool = False) -> "Config":
        base_dir = Path(__file__).resolve().parent

        load_dotenv(dotenv_path=base_dir / ".env", override=False)

        telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        rag_top_k = int(os.getenv("RAG_TOP_K", "3"))

        if not telegram_bot_token and not allow_missing:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
        if not openai_api_key and not allow_missing:
            raise RuntimeError("OPENAI_API_KEY is not set")

        data_dir = base_dir / "data"
        knowledge_base_path = data_dir / "knowledge.csv"
        log_path = data_dir / "log.csv"
        consultation_log_path = data_dir / "consultations.csv"
        consent_store_path = data_dir / "consents.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        consultation_log_path.parent.mkdir(parents=True, exist_ok=True)
        consent_store_path.parent.mkdir(parents=True, exist_ok=True)

        system_prompt_path = base_dir / "prompt_system_ru.txt"
        system_prompt = system_prompt_path.read_text(encoding="utf-8")

        privacy_policy_message = os.getenv(
            "PRIVACY_POLICY_MESSAGE",
            (
                "Перед началом работы откройте политику конфиденциальности,"
                " изучите условия и подтвердите своё согласие."
            ),
        )

        privacy_policy_webapp_url = os.getenv(
            "PRIVACY_POLICY_WEBAPP_URL",
            "https://sabir116rus.github.io/legalbot-policy/",
        )

        google_drive_credentials_env = os.getenv("GOOGLE_DRIVE_CREDENTIALS_FILE")
        google_drive_credentials_file = (
            Path(google_drive_credentials_env).expanduser()
            if google_drive_credentials_env
            else None
        )

        google_drive_logs_folder_id = os.getenv("GOOGLE_DRIVE_LOGS_FOLDER_ID")
        google_drive_consultations_folder_id = os.getenv(
            "GOOGLE_DRIVE_CONSULTATIONS_FOLDER_ID"
        )
        google_drive_reports_folder_id = os.getenv("GOOGLE_DRIVE_REPORTS_FOLDER_ID")
        google_drive_logs_file_id = os.getenv("GOOGLE_DRIVE_LOGS_FILE_ID")
        google_drive_consultations_file_id = os.getenv(
            "GOOGLE_DRIVE_CONSULTATIONS_FILE_ID"
        )
        google_drive_reports_file_id = os.getenv("GOOGLE_DRIVE_REPORTS_FILE_ID")

        return cls(
            telegram_bot_token=telegram_bot_token or "",
            openai_api_key=openai_api_key or "",
            openai_model=openai_model,
            rag_top_k=rag_top_k,
            knowledge_base_path=knowledge_base_path,
            system_prompt=system_prompt,
            log_path=log_path,
            consultation_log_path=consultation_log_path,
            consent_store_path=consent_store_path,
            privacy_policy_message=privacy_policy_message,
            privacy_policy_webapp_url=privacy_policy_webapp_url,
            google_drive_credentials_file=google_drive_credentials_file,
            google_drive_logs_folder_id=google_drive_logs_folder_id,
            google_drive_consultations_folder_id=google_drive_consultations_folder_id,
            google_drive_reports_folder_id=google_drive_reports_folder_id,
            google_drive_logs_file_id=google_drive_logs_file_id,
            google_drive_consultations_file_id=google_drive_consultations_file_id,
            google_drive_reports_file_id=google_drive_reports_file_id,
        )
