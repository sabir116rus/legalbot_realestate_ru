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
    assert "Рекомендации:" in result.text
    assert "Возможные пути решения:" in result.text
    assert "Правовые основания:" in result.text
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
def test_generate_answer_adds_missing_sections():
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

    assert "Рекомендации:" in result.text
    assert "Возможные пути решения:" in result.text
    assert "Правовые основания:" in result.text


@pytest.mark.asyncio
def test_generate_answer_includes_history_in_order():
    kb = Mock()
    kb.query.return_value = []

    response_content = "Ответ"
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

    history = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"},
        {"role": "user", "content": "Q3"},
        {"role": "assistant", "content": "A3"},
    ]
    original_history = [msg.copy() for msg in history]

    asyncio.run(
        service.generate_answer(
            "Новый вопрос",
            history=history,
            history_limit=4,
        )
    )

    mock_create.assert_awaited_once()
    _, kwargs = mock_create.await_args
    messages = kwargs["messages"]

    assert messages[0]["role"] == "system"
    assert messages[1:-1] == history[-4:]
    assert messages[-1]["role"] == "user"
    assert "Вопрос пользователя:" in messages[-1]["content"]
    assert "Контекст (из базы знаний):" in messages[-1]["content"]
    assert history == original_history
