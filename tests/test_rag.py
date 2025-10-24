from rag import KnowledgeBase, build_context_snippets


def test_knowledge_base_normalizes_id_column(tmp_path):
    csv_path = tmp_path / "kb.csv"
    csv_path.write_text(
        "\ufeffid,topic,question,answer,law_refs,url\n"
        "1,Topic,Question,Answer,Law,https://example.com\n",
        encoding="utf-8-sig",
    )

    kb = KnowledgeBase(str(csv_path))

    assert "id" in kb.df.columns
    assert "\ufeffid" not in kb.df.columns
    assert kb.df["id"].iloc[0] == 1


def test_knowledge_base_generates_ids_for_missing_values(tmp_path):
    csv_path = tmp_path / "kb.csv"
    csv_path.write_text(
        "id,topic,question,answer,law_refs,url\n"
        ",Topic1,Question1,Answer1,Law1,https://example.com/1\n"
        " ,Topic2,Question2,Answer2,Law2,https://example.com/2\n",
        encoding="utf-8-sig",
    )

    kb = KnowledgeBase(str(csv_path))

    ids = kb.df["id"].tolist()
    assert ids[0] == "auto_1"
    assert ids[1] == "auto_2"


def test_knowledge_base_converts_float_like_ids(tmp_path):
    csv_path = tmp_path / "kb.csv"
    csv_path.write_text(
        "id,topic,question,answer,law_refs,url\n"
        "1.0,Topic,Question,Answer,Law,https://example.com\n",
        encoding="utf-8-sig",
    )

    kb = KnowledgeBase(str(csv_path))

    assert kb.df["id"].tolist() == [1]


def test_build_context_snippets_contains_id():
    snippet = build_context_snippets([
        {
            "id": 42,
            "topic": "Topic",
            "question": "Question?",
            "answer": "Answer!",
            "law_refs": "Law",
            "url": "https://example.com",
            "score": 99,
        }
    ])

    assert "[ID:42]" in snippet


def test_knowledge_base_query_respects_top_k(tmp_path):
    csv_path = tmp_path / "kb.csv"
    csv_path.write_text(
        "id,topic,question,answer,law_refs,url\n"
        "1,Аренда,Как оформить аренду квартиры?,Ответ1,Закон1,http://a\n"
        "2,Продажа,Как оформить продажу?,Ответ2,Закон2,http://b\n"
        "3,Регистрация,Как зарегистрировать ИП?,Ответ3,Закон3,http://c\n",
        encoding="utf-8-sig",
    )

    kb = KnowledgeBase(str(csv_path))

    results = kb.query("оформить аренду", top_k=2)

    assert len(results) == 2
    assert results[0]["id"] == 1
    assert all("score" in r for r in results)


def test_knowledge_base_query_empty_question(tmp_path):
    csv_path = tmp_path / "kb.csv"
    csv_path.write_text(
        "id,topic,question,answer,law_refs,url\n"
        "1,Тема,Вопрос,Ответ,Закон,http://example\n",
        encoding="utf-8-sig",
    )

    kb = KnowledgeBase(str(csv_path))

    assert kb.query("", top_k=3) == []
