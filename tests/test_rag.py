import json

from models.knowledge import KnowledgeRecord
from rag import KnowledgeBase, build_context_snippets


def _make_record(**overrides) -> dict:
    base = {
        "id": "1",
        "topic": "Тема",
        "question": "Что делать?",
        "answer": "Сделать то-то",
        "steps": ["Шаг 1", "Шаг 2"],
        "docs": ["Паспорт"],
        "law_refs_text": "ГК РФ",
        "law_refs_norm": ["ГК РФ"],
        "sources": [{"title": "example.com", "url": "https://example.com"}],
    }
    base.update(overrides)
    return base


def test_knowledge_base_loads_json(tmp_path):
    data = [_make_record(id="1"), _make_record(id="2", question="Новый вопрос")]
    path = tmp_path / "kb.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    kb = KnowledgeBase(str(path))

    assert len(kb.records) == 2
    assert isinstance(kb.records[0], KnowledgeRecord)
    assert kb.records[0].question == "Что делать?"


def test_knowledge_base_query_respects_top_k(tmp_path):
    data = [
        _make_record(id="1", question="Как оформить аренду квартиры?", answer="Ответ1"),
        _make_record(id="2", question="Как оформить продажу?", answer="Ответ2"),
        _make_record(id="3", question="Как зарегистрировать ИП?", answer="Ответ3"),
    ]
    path = tmp_path / "kb.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    kb = KnowledgeBase(str(path))

    results = kb.query("оформить аренду", top_k=2)

    assert len(results) == 2
    assert results[0]["id"] == "1"
    assert all("score" in r for r in results)


def test_knowledge_base_query_empty_question(tmp_path):
    path = tmp_path / "kb.json"
    path.write_text(json.dumps([_make_record()], ensure_ascii=False), encoding="utf-8")
    kb = KnowledgeBase(str(path))

    assert kb.query("", top_k=3) == []


def test_build_context_snippets_contains_metadata():
    snippet = build_context_snippets([
        {
            "id": "42",
            "topic": "Topic",
            "question": "Question?",
            "answer": "Answer!",
            "steps": ["Step 1"],
            "docs": ["Passport"],
            "law_refs_norm": ["ГК РФ"],
            "sources": [{"title": "site", "url": "https://example.com"}],
            "score": 99,
        }
    ])

    assert "[ID:42]" in snippet
    assert "Ключевые шаги" in snippet
    assert "Источники" in snippet
