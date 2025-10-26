"""Retrieval helper utilities for the knowledge base."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from rapidfuzz import fuzz, process

from models.knowledge import KnowledgeRecord, validate_records


class KnowledgeBase:
    """Search interface over normalized knowledge records."""

    def __init__(self, data_path: str):
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Knowledge base not found: {data_path}")

        with path.open(encoding="utf-8") as fh:
            raw_data = json.load(fh)

        if isinstance(raw_data, dict):
            records_data = raw_data.get("records") or raw_data.get("data")
            if records_data is None:
                raise ValueError("Knowledge JSON must contain an array of records")
        else:
            records_data = raw_data

        if not isinstance(records_data, list):
            raise ValueError("Knowledge JSON must be a list of records")

        self.records: List[KnowledgeRecord] = validate_records(records_data)
        self.corpus = [record.build_search_text() for record in self.records]

    def query(self, user_question: str, top_k: int = 3):
        if not user_question or not user_question.strip():
            return []

        matches = process.extract(
            query=user_question,
            choices=self.corpus,
            scorer=fuzz.WRatio,
            limit=top_k,
        )

        results = []
        for _, score, idx in matches:
            record = self.records[idx]
            data = record.model_dump_for_storage()
            data["score"] = int(score)
            results.append(data)
        return results


def _format_bullets(items: Iterable[str]) -> str:
    items = [item for item in items if item]
    if not items:
        return "-"
    return "\n".join(f"- {item}" for item in items)


def build_context_snippets(rows):
    """Формирует текстовый контекст для подсказки модели."""

    parts = []
    for r in rows:
        row_id = r.get("id")
        row_id = "" if row_id is None else str(row_id)

        steps = r.get("steps") or []
        docs = r.get("docs") or []
        law_refs_norm = r.get("law_refs_norm") or []
        sources = r.get("sources") or []
        if isinstance(sources, dict):
            sources = [sources]

        source_lines = []
        for source in sources:
            if isinstance(source, dict):
                title = source.get("title") or "Источник"
                url = source.get("url")
            else:  # pragma: no cover - defensive branch for malformed data
                title = str(source)
                url = None
            if url:
                source_lines.append(f"{title} ({url})")
            else:
                source_lines.append(title)

        law_refs_text = ", ".join(law_refs_norm) or (r.get("law_refs_text") or "")

        part = (
            f"[ID:{row_id}] Тема: {r.get('topic')}\n"
            f"Вопрос: {r.get('question')}\n"
            f"Ответ: {r.get('answer')}\n"
            f"Ключевые шаги:\n{_format_bullets(steps)}\n"
            f"Документы:\n{_format_bullets(docs)}\n"
            f"Правовые ссылки: {law_refs_text}\n"
            f"Источники:\n{_format_bullets(source_lines)}\n"
            f"(релевантность: {r.get('score')})\n"
        )
        parts.append(part)
    return "\n---\n".join(parts)
