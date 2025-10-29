import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from services.answer_service import AnswerService, AnswerResult


@pytest.mark.asyncio
def test_generate_answer_uses_context_and_strips_markdown():
    knowledge_hits = [
        {
            "id": 1,
            "topic": "Договор аренды",
            "question": "Как оформить аренду?",
            "answer": "Ответ в базе",
            "law_refs": "Статья 123",
            "url": "https://example.com",
            "score": 95,
        }
    ]
    kb = Mock()
    kb.query.return_value = knowledge_hits

    response_content = "**Привет**, вот _ответ_."
    mock_create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=response_content)
                )
            ]
        )
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=mock_create))
    )

    service = AnswerService(
        knowledge_base=kb,
        openai_client=client,
        model="gpt-test",
        system_prompt="Системное сообщение",
        rag_top_k=3,
    )

    result = asyncio.run(service.generate_answer("Что с арендой?"))

    assert isinstance(result, AnswerResult)
    assert result.text.startswith("Привет, вот ответ.")
    assert "Суть ситуации:" in result.text
    assert "Что нужно уточнить:" in result.text
    assert "Рекомендации:" in result.text
    assert "Возможные пути решения:" in result.text
    assert "Правовые основания:" in result.text
    assert "Предупреждения и ограничения:" in result.text
    assert result.top_score == 95
    assert result.status == "ok"

    mock_create.assert_awaited_once()
    _, kwargs = mock_create.await_args
    assert kwargs["model"] == "gpt-test"
    messages = kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert "Контекст (из базы знаний):" in messages[1]["content"]
    assert "[ID:1]" in messages[1]["content"]


@pytest.mark.asyncio
def test_generate_answer_handles_exceptions():
    kb = Mock()
    kb.query.return_value = []

    mock_create = AsyncMock(side_effect=RuntimeError("boom"))
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=mock_create))
    )

    service = AnswerService(
        knowledge_base=kb,
        openai_client=client,
        model="gpt-test",
        system_prompt="Системное сообщение",
        rag_top_k=3,
    )

    result = asyncio.run(service.generate_answer("Вопрос"))

    assert result.status == "error"
    assert result.top_score == 0
    assert "Извини" in result.text
    assert "Техническая ошибка" in result.text


@pytest.mark.asyncio
def test_generate_answer_adds_missing_sections_with_placeholders():
    kb = Mock()
    kb.query.return_value = []

    response_content = "Просто ответ без разделов"
    mock_create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=response_content)
                )
            ]
        )
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=mock_create))
    )

    service = AnswerService(
        knowledge_base=kb,
        openai_client=client,
        model="gpt-test",
        system_prompt="Системное сообщение",
        rag_top_k=3,
    )

    result = asyncio.run(service.generate_answer("Вопрос"))

    assert "Суть ситуации:\n- Раздел не был сформирован моделью." in result.text
    assert (
        "Что нужно уточнить:\n- Уточните region, object_type, role, mortgage, minors_involved."
        in result.text
    )
    assert "Рекомендации:\n- Раздел не был сформирован моделью." in result.text
    assert "Возможные пути решения:\n- Раздел не был сформирован моделью." in result.text
    assert "Правовые основания:\n- Раздел не был сформирован моделью." in result.text
    assert (
        "Предупреждения и ограничения:\n- Раздел не был сформирован моделью. Помните, что ассистент не заменяет юриста и информация требует проверки."
        in result.text
    )
