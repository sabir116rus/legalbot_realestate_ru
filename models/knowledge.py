"""Pydantic schemas for knowledge base records."""
from __future__ import annotations

import re
from typing import Iterable, List, Sequence

from pydantic import BaseModel, Field, HttpUrl, ValidationInfo, field_validator


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


class Source(BaseModel):
    """Information about a reference source."""

    title: str = Field(..., min_length=1)
    url: HttpUrl | None = None

    @field_validator("title", mode="before")
    @classmethod
    def _strip_title(cls, value: object) -> str:
        if value is None:
            raise ValueError("source title must not be empty")
        if isinstance(value, (int, float)):
            value = str(value)
        if not isinstance(value, str):
            raise TypeError("source title must be a string")
        normalized = _normalize_text(value)
        if not normalized:
            raise ValueError("source title must not be empty")
        return normalized


class KnowledgeRecord(BaseModel):
    """Normalized knowledge base entry."""

    id: str
    topic: str
    question: str
    answer: str
    summary: str | None = None
    problem: str | None = None
    steps: List[str] = Field(default_factory=list)
    docs: List[str] = Field(default_factory=list)
    law_refs_text: str | None = None
    law_refs_norm: List[str] = Field(default_factory=list)
    sources: List[Source] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    related_questions: List[str] = Field(default_factory=list)

    @field_validator("id", mode="before")
    @classmethod
    def _ensure_id(cls, value: object) -> str:
        if value is None:
            raise ValueError("knowledge record id is required")
        if isinstance(value, (int, float)):
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            value = str(value)
        if not isinstance(value, str):
            raise TypeError("knowledge record id must be a string")
        normalized = value.strip()
        if not normalized:
            raise ValueError("knowledge record id must not be empty")
        return normalized

    @field_validator("topic", "question", "answer", mode="before")
    @classmethod
    def _normalize_required_field(cls, value: object, info: ValidationInfo) -> str:
        field_name = info.field_name or "field"
        if value is None:
            raise ValueError(f"knowledge record {field_name} is required")
        if isinstance(value, (int, float)):
            value = str(value)
        if not isinstance(value, str):
            raise TypeError(f"knowledge record {field_name} must be a string")
        normalized = _normalize_text(value)
        if not normalized:
            raise ValueError(f"knowledge record {field_name} must not be empty")
        return normalized

    @field_validator("summary", "problem", "law_refs_text", mode="before")
    @classmethod
    def _normalize_optional_field(cls, value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            value = str(value)
        if not isinstance(value, str):
            raise TypeError("optional field values must be strings")
        normalized = _normalize_text(value)
        return normalized or None

    @staticmethod
    def _prepare_list(value: object) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            parts = re.split(r"[\n;]+", value)
            return [_normalize_text(part) for part in parts if _normalize_text(part)]
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            normalized_items: List[str] = []
            for item in value:
                if item is None:
                    continue
                if isinstance(item, (int, float)):
                    item = str(item)
                if not isinstance(item, str):
                    raise TypeError("list items must be strings")
                normalized = _normalize_text(item)
                if normalized:
                    normalized_items.append(normalized)
            return normalized_items
        raise TypeError("value must be a string or a sequence of strings")

    @field_validator("steps", "docs", "law_refs_norm", "tags", "related_questions", mode="before")
    @classmethod
    def _normalize_string_lists(cls, value: object) -> List[str]:
        return cls._prepare_list(value)

    @field_validator("steps", "docs", "law_refs_norm", "tags", "related_questions", mode="after")
    @classmethod
    def _ensure_unique(cls, value: List[str]) -> List[str]:
        seen = set()
        unique_items: List[str] = []
        for item in value:
            lowered = item.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            unique_items.append(item)
        return unique_items

    @field_validator("sources", mode="before")
    @classmethod
    def _prepare_sources(cls, value: object) -> Iterable[object]:
        if value is None:
            return []
        if isinstance(value, (Source, dict)):
            return [value]
        if isinstance(value, str):
            return [value]
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            return list(value)
        raise TypeError("sources must be provided as a list")

    def build_search_text(self) -> str:
        extra_parts = [
            " ".join(self.steps),
            " ".join(self.docs),
            self.law_refs_text or "",
            " ".join(self.law_refs_norm),
            " ".join(source.title for source in self.sources),
        ]
        fragments = [self.topic, self.question, self.answer, *extra_parts]
        normalized: List[str] = []
        for fragment in fragments:
            if not fragment:
                continue
            cleaned = _normalize_text(fragment)
            if cleaned:
                normalized.append(cleaned)
        return " | ".join(normalized)

    def model_dump_for_storage(self) -> dict:
        data = self.model_dump()
        data["sources"] = [source.model_dump(mode="json") for source in self.sources]
        return data


def validate_records(data: Iterable[dict]) -> List[KnowledgeRecord]:
    records: List[KnowledgeRecord] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(data, start=1):
        try:
            record = KnowledgeRecord.model_validate(item)
        except Exception as exc:  # pragma: no cover - defensive rewrap
            raise ValueError(
                f"Invalid knowledge record at index {index}: {exc}"
            ) from exc
        if record.id in seen_ids:
            raise ValueError(f"Duplicate knowledge record id detected: {record.id}")
        seen_ids.add(record.id)
        records.append(record)
    return records
