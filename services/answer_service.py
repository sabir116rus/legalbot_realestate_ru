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
        section_titles = (
            "Суть ситуации",
            "Что нужно уточнить",
            "Рекомендации",
            "Возможные пути решения",
            "Правовые основания",
            "Предупреждения и ограничения",
        )

        canonical_titles = {title.lower(): title for title in section_titles}
        section_regex = re.compile(
            rf"^(?P<title>{'|'.join(re.escape(title) for title in section_titles)})\s*:\s*",
            re.IGNORECASE | re.MULTILINE,
        )

        normalized_text = text.strip()
        matches = list(section_regex.finditer(normalized_text))
        if not matches:
            return normalized_text

        parts: list[str] = []

        preamble = normalized_text[: matches[0].start()].strip()
        if preamble:
            parts.append(preamble)

        for index, match in enumerate(matches):
            title_raw = match.group("title")
            title = canonical_titles.get(title_raw.lower(), title_raw)
            content_start = match.end()
            content_end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized_text)
            content = normalized_text[content_start:content_end].strip()

            if not content:
                continue

            parts.append(f"{title}:\n{content}")

        return "\n\n".join(parts).strip()

    @property
    def model(self) -> str:
        return self._model
