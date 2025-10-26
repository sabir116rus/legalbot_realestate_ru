import json
from pathlib import Path

from models.knowledge import validate_records


def test_knowledge_dataset_is_valid():
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / "knowledge.json"
    assert data_path.exists(), "knowledge.json is missing"

    raw_data = json.loads(data_path.read_text(encoding="utf-8"))
    records = validate_records(raw_data)

    assert len(records) >= 50
