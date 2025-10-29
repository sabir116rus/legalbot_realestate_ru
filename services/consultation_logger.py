from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


class ConsultationLogger:
    HEADER = [
        "timestamp",
        "user_id",
        "username",
        "name",
        "contact",
        "request",
    ]

    def __init__(self, log_path: Path) -> None:
        self._log_path = log_path
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        *,
        user_id: int,
        username: str | None,
        name: str,
        contact: str,
        request: str,
    ) -> None:
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_id": user_id,
            "username": username or "",
            "name": name,
            "contact": contact,
            "request": request,
        }

        file_exists = self._log_path.exists()
        with self._log_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.HEADER)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
