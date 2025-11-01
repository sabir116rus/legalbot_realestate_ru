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

    async def generate_answer(
        self,
        user_question: str,
        *,
        history: list[dict[str, str]] | None = None,
        history_limit: int = 10,
    ) -> AnswerResult:
        hits = self._knowledge_base.query(user_question, top_k=self._rag_top_k)
        top_score = hits[0]["score"] if hits else 0
        context_text = (
            build_context_snippets(hits)
            if hits
            else "Контекст из базы знаний не найден."
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt}
        ]

        normalized_history = list(history or [])
        if history_limit > 0:
            normalized_history = normalized_history[-history_limit:]
        elif history_limit == 0:
            normalized_history = []

        if normalized_history:
            messages.extend(normalized_history)

        messages.append(
            {
                "role": "user",
                "content": (
                    "Вопрос пользователя:\n"
                    f"{user_question}\n\n"
                    "Контекст (из базы знаний):\n"
                    f"{context_text}"
                ),
            }
        )

        try:
            response = await self._openai_client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.2,
            )
            text = response.choices[0].message.content.strip()
            text = self._strip_markdown(text)
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
        text = re.sub(r"(?<!\w)_(?!\s)([^_]+?)(?<!\s)_(?!\w)", r"\1", text)
        return text

    @property
    def model(self) -> str:
        return self._model
