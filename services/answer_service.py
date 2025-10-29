from __future__ import annotations

import re
from dataclasses import dataclass

from openai import AsyncOpenAI

from rag import KnowledgeBase, build_context_snippets


@dataclass
class AnswerResult:
    text: str
    top_score: int
    status: str = "ok"


class AnswerService:
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        openai_client: AsyncOpenAI,
        *,
        model: str,
        system_prompt: str,
        rag_top_k: int,
    ) -> None:
        self._knowledge_base = knowledge_base
        self._openai_client = openai_client
        self._model = model
        self._system_prompt = system_prompt
        self._rag_top_k = rag_top_k

    async def generate_answer(self, user_question: str) -> AnswerResult:
        hits = self._knowledge_base.query(user_question, top_k=self._rag_top_k)
        top_score = hits[0]["score"] if hits else 0
        context_text = (
            build_context_snippets(hits)
            if hits
            else "Контекст из базы знаний не найден."
        )

        messages = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": (
                    "Вопрос пользователя:\n"
                    f"{user_question}\n\n"
                    "Контекст (из базы знаний):\n"
                    f"{context_text}"
                ),
            },
        ]

        try:
            response = await self._openai_client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.2,
            )
            text = response.choices[0].message.content.strip()
            text = self._strip_markdown(text)
            text = self._ensure_required_sections(text)
            status = "ok"
        except Exception as exc:  # pragma: no cover - network errors
            text = (
                "Извини, сейчас не удалось получить ответ от модели.\n"
                f"Техническая ошибка: {exc}"
            )
            status = "error"

        return AnswerResult(text=text, top_score=top_score, status=status)

    @staticmethod
    def _strip_markdown(text: str) -> str:
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        text = re.sub(r"#+\s*", "", text)
        text = re.sub(r"_([^_]+)_", r"\1", text)
        return text

    @staticmethod
    def _ensure_required_sections(text: str) -> str:
        required_sections = (
            ("Суть ситуации", "- Раздел не был сформирован моделью."),
            (
                "Что нужно уточнить",
                "- Уточните region, object_type, role, mortgage, minors_involved.",
            ),
            ("Рекомендации", "- Раздел не был сформирован моделью."),
            (
                "Возможные пути решения",
                "- Раздел не был сформирован моделью.",
            ),
            ("Правовые основания", "- Раздел не был сформирован моделью."),
            (
                "Предупреждения и ограничения",
                "- Раздел не был сформирован моделью. Помните, что ассистент не заменяет юриста и информация требует проверки.",
            ),
        )

        normalized_text = text.strip()
        for section, placeholder in required_sections:
            pattern = re.compile(rf"^\s*{re.escape(section)}\b", re.IGNORECASE | re.MULTILINE)
            if pattern.search(normalized_text):
                continue

            if normalized_text:
                normalized_text += "\n\n"
            normalized_text += f"{section}:\n{placeholder}"

        return normalized_text

    @property
    def model(self) -> str:
        return self._model
