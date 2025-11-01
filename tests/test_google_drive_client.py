from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaInMemoryUpload

from services.google_drive_client import GoogleDriveClient, upload_csv_content


class DummyRequest:
    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self._result = result
        self._error = error

    def execute(self):
        if self._error:
            raise self._error
        return self._result


class DummyFilesResource:
    def __init__(self, folder_metadata: dict[str, Any] | None, *, get_error: Exception | None = None):
        self._folder_metadata = folder_metadata
        self._get_error = get_error
        self.list_called = False

    def get(self, **kwargs):
        if self._get_error:
            raise self._get_error
        return DummyRequest(self._folder_metadata)

    def list(self, **kwargs):
        self.list_called = True
        return DummyRequest({"files": []})


class DummyService:
    def __init__(self, files_resource: DummyFilesResource) -> None:
        self._files_resource = files_resource

    def files(self):
        return self._files_resource


class DummyUpdateRequest:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.called_with: dict[str, Any] | None = None

    def execute(self):
        if self.error:
            raise self.error
        return {"id": self.called_with["fileId"]}


class DummyUpdateFiles(DummyFilesResource):
    def __init__(self, request: DummyUpdateRequest) -> None:
        super().__init__({"id": "folder"})
        self._request = request

    def update(self, **kwargs):
        self._request.called_with = kwargs
        return self._request


def make_http_error(status: int = 500, reason: str = "error") -> HttpError:
    class Response:
        def __init__(self, status: int, reason: str) -> None:
            self.status = status
            self.reason = reason

        def get(self, name: str, default=None):
            return getattr(self, name, default)

    return HttpError(Response(status, reason), b"error")


def test_upload_csv_content_updates_file():
    request = DummyUpdateRequest()
    service = DummyService(DummyUpdateFiles(request))

    result = upload_csv_content(service, "file123", "col1\nvalue", mime_type="text/plain")

    assert result == "file123"
    assert isinstance(request.called_with["media_body"], MediaInMemoryUpload)
    assert request.called_with["fileId"] == "file123"
    assert request.called_with["supportsAllDrives"] is True


def test_upload_csv_content_handles_http_error():
    error = make_http_error()
    request = DummyUpdateRequest(error=error)
    service = DummyService(DummyUpdateFiles(request))

    result = upload_csv_content(service, "file123", b"data")

    assert result is None


def test_google_drive_client_updates_existing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    local_path = tmp_path / "report.csv"
    local_path.write_text("header\nvalue", encoding="utf-8")

    files_resource = DummyFilesResource({"id": "folder", "driveId": "drive"})
    service = DummyService(files_resource)

    client = GoogleDriveClient(credentials_file=None)
    client._service = service  # bypass credential loading

    monkeypatch.setattr(client, "_ensure_service", lambda: service)
    monkeypatch.setattr(
        client,
        "_get_file_metadata",
        lambda svc, file_id: {"id": file_id, "name": "report.csv", "mimeType": "text/csv"},
    )

    captured: dict[str, Any] = {}

    def fake_update(service_arg, file_id, file_content, *, remote_mime_type, detected_mime_type):
        captured.update(
            {
                "service": service_arg,
                "file_id": file_id,
                "file_content": file_content,
                "remote_mime_type": remote_mime_type,
                "detected_mime_type": detected_mime_type,
            }
        )
        return file_id

    monkeypatch.setattr(client, "_update_drive_file", fake_update)

    client.upload_or_update_file(local_path, "folder", file_id="existing-id", mime_type="text/csv")

    assert captured["service"] is service
    assert captured["file_id"] == "existing-id"
    assert captured["file_content"] == local_path.read_bytes()
    assert captured["remote_mime_type"] == "text/csv"
    assert captured["detected_mime_type"] == "text/csv"
    assert files_resource.list_called is False


def test_google_drive_client_logs_folder_access_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog):
    local_path = tmp_path / "report.csv"
    local_path.write_text("header\nvalue", encoding="utf-8")

    error = make_http_error(status=404, reason="notFound")
    files_resource = DummyFilesResource(None, get_error=error)
    service = DummyService(files_resource)

    client = GoogleDriveClient(credentials_file=None)
    client._service = service

    monkeypatch.setattr(client, "_ensure_service", lambda: service)

    with caplog.at_level("ERROR"):
        client.upload_or_update_file(local_path, "folder")

    assert "Failed to access Google Drive folder" in caplog.text
