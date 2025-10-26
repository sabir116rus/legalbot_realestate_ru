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
    assert result.text == "Привет, вот ответ."
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
def test_generate_answer_does_not_add_missing_sections():
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

    assert "Суть ситуации:" not in result.text
    assert "Что нужно уточнить:" not in result.text
    assert "Рекомендации:" not in result.text
    assert "Возможные пути решения:" not in result.text
    assert "Правовые основания:" not in result.text
    assert "Предупреждения и ограничения:" not in result.text


@pytest.mark.asyncio
def test_generate_answer_removes_empty_sections():
    kb = Mock()
    kb.query.return_value = []

    response_content = (
        "Вступление\n\nСуть ситуации: Описание ситуации\n\nРекомендации:\n\n"
        "Возможные пути решения:   \n  \nПравовые основания: Статья 1\n"
    )
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

    assert "Вступление" in result.text
    assert "Суть ситуации:\nОписание ситуации" in result.text
    assert "Рекомендации:" not in result.text
    assert "Возможные пути решения:" not in result.text
    assert "Правовые основания:\nСтатья 1" in result.text
