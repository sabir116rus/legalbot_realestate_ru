import pytest

from config import Config


def test_config_loads_with_env_and_temp_prompt(tmp_path, monkeypatch):
    config_dir = tmp_path / "app"
    data_dir = config_dir / "data"
    data_dir.mkdir(parents=True)

    prompt_path = config_dir / "prompt_system_ru.txt"
    prompt_content = "Системный текст"
    prompt_path.write_text(prompt_content, encoding="utf-8")

    fake_config_file = config_dir / "config.py"
    fake_config_file.write_text("# temp config", encoding="utf-8")

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("RAG_TOP_K", "4")
    monkeypatch.setattr("config.__file__", str(fake_config_file))

    cfg = Config.load()

    assert cfg.telegram_bot_token == "token"
    assert cfg.openai_api_key == "key"
    assert cfg.openai_model == "gpt-test"
    assert cfg.rag_top_k == 4
    assert cfg.system_prompt == prompt_content
    assert cfg.knowledge_base_path == data_dir / "knowledge.csv"
    assert cfg.log_path == data_dir / "log.csv"
    assert cfg.log_path.parent == data_dir
    assert cfg.log_path.parent.exists()


@pytest.mark.parametrize("missing_var", ["TELEGRAM_BOT_TOKEN", "OPENAI_API_KEY"])
def test_config_requires_tokens(tmp_path, monkeypatch, missing_var):
    config_dir = tmp_path / "app"
    config_dir.mkdir()
    prompt_path = config_dir / "prompt_system_ru.txt"
    prompt_path.write_text("prompt", encoding="utf-8")
    fake_config_file = config_dir / "config.py"
    fake_config_file.write_text("# temp config", encoding="utf-8")

    monkeypatch.setattr("config.__file__", str(fake_config_file))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("OPENAI_API_KEY", "key")

    monkeypatch.delenv(missing_var, raising=False)

    with pytest.raises(RuntimeError) as exc:
        Config.load()

    assert missing_var in str(exc.value)
