from rag import KnowledgeBase, build_context_snippets
import pytest

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
