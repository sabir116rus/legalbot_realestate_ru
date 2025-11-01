import csv
from pathlib import Path

from services.consultation_logger import ConsultationLogger


class DummyDriveClient:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple, dict]] = []

    def upload_or_update_file(self, *args, **kwargs):  # pragma: no cover - helper used in tests
        self.calls.append((args, kwargs))


def test_consultation_logger_creates_file_and_syncs_drive(tmp_path: Path):
    log_path = tmp_path / "logs" / "consultations.csv"
    drive_client = DummyDriveClient()
    logger = ConsultationLogger(
        log_path,
        drive_client=drive_client,
        drive_folder_id="folder-id",
        drive_file_id="file-id",
        drive_file_id_env_var="ENV_VAR",
    )

    logger.log(
        user_id=123,
        username="user",
        name="John Doe",
        contact="+1234567890",
        request="Need consultation",
    )

    assert log_path.exists()

    with log_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        header = reader.fieldnames

    assert header == ConsultationLogger.HEADER
    assert len(rows) == 1
    row = rows[0]
    assert row["user_id"] == "123"
    assert row["username"] == "user"
    assert row["name"] == "John Doe"
    assert row["contact"] == "+1234567890"
    assert row["request"] == "Need consultation"
    assert row["timestamp"]  # timestamp is populated

    assert len(drive_client.calls) == 1
    args, kwargs = drive_client.calls[0]
    assert args[0] == log_path
    assert args[1] == "folder-id"
    assert kwargs["mime_type"] == "text/csv"
    assert kwargs["file_id"] == "file-id"
    assert kwargs["file_id_env_var"] == "ENV_VAR"


def test_consultation_logger_appends_without_duplicate_header(tmp_path: Path):
    log_path = tmp_path / "consultations.csv"
    logger = ConsultationLogger(log_path)

    logger.log(
        user_id=1,
        username=None,
        name="Alice",
        contact="alice@example.com",
        request="First",
    )
    logger.log(
        user_id=2,
        username="bob",
        name="Bob",
        contact="bob@example.com",
        request="Second",
    )

    content = log_path.read_text(encoding="utf-8").strip().splitlines()
    header_line = ",".join(ConsultationLogger.HEADER)

    assert content[0] == header_line
    assert content.count(header_line) == 1
    assert len(content) == 1 + 2
