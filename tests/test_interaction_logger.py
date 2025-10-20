import csv

from services.interaction_logger import InteractionLogger


class DummyEncoding:
    def encode(self, text: str):
        return text.split()


def test_log_writes_header_and_row(tmp_path, monkeypatch):
    log_path = tmp_path / "logs" / "interactions.csv"
    monkeypatch.setattr(
        "services.interaction_logger.tiktoken.encoding_for_model",
        lambda model: DummyEncoding(),
    )

    logger = InteractionLogger(log_path)

    long_answer = "слово " * 40  # length greater than preview limit
    logger.log(
        user_id=1,
        username="user",
        question="Вопрос",
        answer=long_answer,
        top_score=88,
        model="gpt-test",
        status="ok",
    )

    assert log_path.exists()

    with log_path.open(encoding="utf-8") as file:
        reader = csv.reader(file)
        rows = list(reader)

    assert rows[0] == InteractionLogger.HEADER
    data_row = rows[1]
    row_dict = dict(zip(InteractionLogger.HEADER, data_row))

    assert row_dict["user_id"] == "1"
    assert row_dict["username"] == "user"
    assert row_dict["question"] == "Вопрос"
    assert row_dict["top_score"] == "88"
    assert row_dict["model"] == "gpt-test"
    assert row_dict["status"] == "ok"

    expected_preview = ("слово " * 40)[:150] + "..."
    assert row_dict["answer_preview"] == expected_preview
    assert row_dict["tokens"] == str(len(long_answer.split()))


def test_count_tokens_fallback_on_encoding_error(monkeypatch):
    monkeypatch.setattr(
        "services.interaction_logger.tiktoken.encoding_for_model",
        lambda model: (_ for _ in ()).throw(UnicodeDecodeError("utf-8", b"", 0, 1, "")),
    )

    tokens = InteractionLogger._count_tokens("одно два три", "gpt-test")

    assert tokens == 3
