from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import tiktoken

if TYPE_CHECKING:
    from .google_drive_client import GoogleDriveClient


class InteractionLogger:
    HEADER = [
        "timestamp",
        "user_id",
        "username",
        "question",
        "answer_preview",
        "top_score",
        "tokens",
        "model",
        "status",
    ]

    def __init__(
        self,
        log_path: Path,
        *,
        drive_client: Optional["GoogleDriveClient"] = None,
        drive_folder_id: Optional[str] = None,
        drive_file_id: Optional[str] = None,
        drive_file_id_env_var: Optional[str] = None,
    ) -> None:
        self._log_path = log_path
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._drive_client = drive_client
        self._drive_folder_id = drive_folder_id
        self._drive_file_id = drive_file_id
        self._drive_file_id_env_var = drive_file_id_env_var
        self._logger = logging.getLogger(__name__)

    def log(
        self,
        *,
        user_id: int,
        username: str | None,
        question: str,
        answer: str,
        top_score: int,
        model: str,
        status: str = "ok",
    ) -> None:
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_id": user_id,
            "username": username or "",
            "question": question,
            "answer_preview": self._answer_preview(answer),
            "top_score": top_score,
            "tokens": self._count_tokens(answer, model),
            "model": model,
            "status": status,
        }

        file_exists = self._log_path.exists()
        with self._log_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.HEADER)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        self._sync_drive()

    @staticmethod
    def _count_tokens(text: str, model: str) -> int:
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except Exception:  # pragma: no cover - fallback when encoding missing
            return len(text.split())

    @staticmethod
    def _answer_preview(answer: str, limit: int = 150) -> str:
        if len(answer) > limit:
            return f"{answer[:limit]}..."
        return answer

    def _sync_drive(self) -> None:
        if not self._drive_client or not self._drive_folder_id:
            return

        try:
            self._drive_client.upload_or_update_file(
                self._log_path,
                self._drive_folder_id,
                mime_type="text/csv",
                file_id=self._drive_file_id,
                file_id_env_var=self._drive_file_id_env_var,
            )
        except Exception as exc:  # pragma: no cover - network/API failures
            self._logger.warning("Failed to sync interaction log to Google Drive: %s", exc)
