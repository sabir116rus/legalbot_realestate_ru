from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import tiktoken


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

    def __init__(self, log_path: Path) -> None:
        self._log_path = log_path
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

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
