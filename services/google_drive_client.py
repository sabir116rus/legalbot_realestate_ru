from __future__ import annotations

import csv
import json
import logging
import mimetypes
from io import StringIO
from pathlib import Path
from typing import Any, Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaInMemoryUpload
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials as UserCredentials

LOGGER = logging.getLogger(__name__)

DEFAULT_SCOPES = ["https://www.googleapis.com/auth/drive"]


def upload_csv_content(
    service: Any,
    file_id: str,
    content: bytes | str,
    *,
    mime_type: str = "text/csv",
) -> str | None:
    payload = content.encode("utf-8") if isinstance(content, str) else content

    media = MediaInMemoryUpload(payload, mimetype=mime_type, resumable=False)
    try:
        updated = (
            service.files()
            .update(fileId=file_id, media_body=media, supportsAllDrives=True)
            .execute()
        )
        return updated.get("id")
    except HttpError as e:  # pragma: no cover - network/API failures
        LOGGER.error("Failed to update Google Drive file '%s': %s", file_id, e)
        return None


class GoogleDriveClient:
    """Thin wrapper around Google Drive API for uploading CSV reports."""

    def __init__(
        self,
        credentials_file: Optional[Path],
        *,
        scopes: Optional[list[str]] = None,
    ) -> None:
        if credentials_file:
            self._credentials_file = Path(credentials_file).expanduser().resolve()
        else:
            self._credentials_file = None
        self._scopes = scopes or DEFAULT_SCOPES
        self._service = None
        self._sheets_service = None
        self._credentials = None

    @property
    def is_configured(self) -> bool:
        return bool(self._credentials_file and self._credentials_file.exists())

    def upload_or_update_file(
        self,
        local_path: Path,
        folder_id: str,
        *,
        file_name: Optional[str] = None,
        mime_type: Optional[str] = None,
        file_id: Optional[str] = None,
        file_id_env_var: Optional[str] = None,
    ) -> None:
        """Create or update a file in the provided Drive folder."""

        service = self._ensure_service()
        if not service:
            LOGGER.debug("Google Drive client is not configured; skipping upload")
            return

        if not folder_id:
            LOGGER.debug("Google Drive folder ID is empty; skipping upload")
            return

        if not local_path.exists():
            LOGGER.warning("Local file does not exist, cannot upload: %s", local_path)
            return

        resolved_path = local_path.expanduser().resolve()
        upload_name = file_name or resolved_path.name
        detected_mime_type = mime_type or mimetypes.guess_type(upload_name)[0] or "application/octet-stream"

        safe_name = upload_name.replace("'", "\\'")
        query = (
            f"'{folder_id}' in parents and name = '{safe_name}' and trashed = false"
        )

        try:
            folder_metadata = (
                service.files()
                .get(
                    fileId=folder_id,
                    fields="id, driveId",
                    supportsAllDrives=True,
                )
                .execute()
            )
        except HttpError as exc:
            LOGGER.error(
                "Failed to access Google Drive folder '%s': %s", folder_id, exc
            )
            return
        except Exception as exc:  # pragma: no cover - unexpected failures
            LOGGER.error(
                "Unexpected error retrieving Google Drive folder '%s': %s",
                folder_id,
                exc,
            )
            return

        drive_id = folder_metadata.get("driveId")

        files: list[dict[str, Any]] = []

        if not file_id:
            try:
                list_kwargs: dict[str, Any] = {
                    "q": query,
                    "spaces": "drive",
                    "fields": "files(id, name, mimeType)",
                    "supportsAllDrives": True,
                }

                if drive_id:
                    list_kwargs.update({"corpora": "drive", "driveId": drive_id})
                else:  # fall back to all drives for My Drive or unknown parents
                    list_kwargs["includeItemsFromAllDrives"] = True

                response = service.files().list(**list_kwargs).execute()
                files = response.get("files", [])
            except HttpError as exc:
                LOGGER.error("Failed to query Google Drive files: %s", exc)
                return
            except Exception as exc:  # pragma: no cover - unexpected failures
                LOGGER.error("Unexpected error during Google Drive query: %s", exc)
                return

        file_content = resolved_path.read_bytes()

        target_metadata: dict[str, Any] | None = None

        if file_id:
            target_metadata = self._get_file_metadata(service, file_id)
            if not target_metadata:
                return

        if not target_metadata and files:
            target_metadata = files[0]

        if target_metadata:
            updated_id = self._update_drive_file(
                service,
                target_metadata["id"],
                file_content,
                remote_mime_type=target_metadata.get("mimeType"),
                detected_mime_type=detected_mime_type,
            )
            if updated_id:
                remote_name = target_metadata.get("name") or upload_name
                LOGGER.info(
                    "Обновлён файл Google Drive '%s' (%s)", remote_name, updated_id
                )
            return

        env_hint = (
            f" и укажите {file_id_env_var} в .env"
            if file_id_env_var
            else " и укажите соответствующий идентификатор файла в настройках"
        )
        LOGGER.error(
            "Нельзя создать новый файл на 'Мой диск' от сервисного аккаунта. "
            "Создайте пустой %s вручную%s.",
            upload_name,
            env_hint,
        )

    def _ensure_service(self) -> Any:
        if self._service is not None:
            return self._service

        if not self._credentials_file:
            return None

        if not self._credentials_file.exists():
            LOGGER.error(
                "Google Drive credentials file not found: %s", self._credentials_file
            )
            return None

        if self._credentials is None:
            try:
                self._credentials = self._load_credentials(self._credentials_file)
            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.error("Failed to load Google Drive credentials: %s", exc)
                return None

        try:
            self._service = build("drive", "v3", credentials=self._credentials)
        except Exception as exc:  # pragma: no cover - API discovery failures
            LOGGER.error("Failed to initialise Google Drive service: %s", exc)
            return None

        return self._service

    def _ensure_sheets_service(self) -> Any:
        if self._sheets_service is not None:
            return self._sheets_service

        if self._credentials is None and not self._ensure_service():
            return None

        if self._credentials is None:
            return None

        try:
            self._sheets_service = build("sheets", "v4", credentials=self._credentials)
        except Exception as exc:  # pragma: no cover - API discovery failures
            LOGGER.error("Failed to initialise Google Sheets service: %s", exc)
            return None

        return self._sheets_service

    def _get_file_metadata(self, service: Any, file_id: str) -> dict[str, Any] | None:
        try:
            return (
                service.files()
                .get(
                    fileId=file_id,
                    fields="id, name, mimeType",
                    supportsAllDrives=True,
                )
                .execute()
            )
        except HttpError as exc:
            LOGGER.error("Failed to access Google Drive file '%s': %s", file_id, exc)
        except Exception as exc:  # pragma: no cover - unexpected failures
            LOGGER.error(
                "Unexpected error retrieving Google Drive file '%s': %s", file_id, exc
            )
        return None

    def _update_drive_file(
        self,
        service: Any,
        file_id: str,
        file_content: bytes,
        *,
        remote_mime_type: Optional[str],
        detected_mime_type: str,
    ) -> str | None:
        if remote_mime_type == "application/vnd.google-apps.spreadsheet":
            return self._update_spreadsheet(file_id, file_content)

        effective_mime = remote_mime_type or detected_mime_type
        return upload_csv_content(
            service,
            file_id,
            file_content,
            mime_type=effective_mime,
        )

    def _update_spreadsheet(self, file_id: str, file_content: bytes) -> str | None:
        sheets_service = self._ensure_sheets_service()
        if not sheets_service:
            LOGGER.error(
                "Google Sheets service is not configured; cannot update spreadsheet '%s'",
                file_id,
            )
            return None

        try:
            text = file_content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = file_content.decode("utf-8-sig")
            except UnicodeDecodeError:
                LOGGER.error(
                    "Failed to decode CSV content for Google Sheet '%s'", file_id
                )
                return None

        reader = csv.reader(StringIO(text))
        rows = [row for row in reader]
        if not rows:
            rows = [[]]

        try:
            metadata = (
                sheets_service.spreadsheets()
                .get(spreadsheetId=file_id, fields="sheets.properties.title")
                .execute()
            )
        except HttpError as exc:
            LOGGER.error("Failed to fetch spreadsheet metadata for '%s': %s", file_id, exc)
            return None
        except Exception as exc:  # pragma: no cover - unexpected failures
            LOGGER.error(
                "Unexpected error retrieving spreadsheet metadata for '%s': %s",
                file_id,
                exc,
            )
            return None

        sheets = metadata.get("sheets", [])
        if not sheets:
            LOGGER.error("Spreadsheet '%s' has no sheets to update", file_id)
            return None

        sheet_title = sheets[0].get("properties", {}).get("title", "Sheet1")
        target_range = f"{sheet_title}!A1"

        try:
            sheets_service.values().clear(
                spreadsheetId=file_id,
                range=sheet_title,
            ).execute()
            sheets_service.values().update(
                spreadsheetId=file_id,
                range=target_range,
                valueInputOption="RAW",
                body={"values": rows},
            ).execute()
        except HttpError as exc:
            LOGGER.error("Failed to update Google Sheet '%s': %s", file_id, exc)
            return None
        except Exception as exc:  # pragma: no cover - unexpected failures
            LOGGER.error(
                "Unexpected error updating Google Sheet '%s': %s", file_id, exc
            )
            return None

        return file_id

    def _load_credentials(self, credentials_path: Path):
        data = json.loads(credentials_path.read_text(encoding="utf-8"))
        cred_type = data.get("type")

        if cred_type == "service_account":
            return service_account.Credentials.from_service_account_info(data, scopes=self._scopes)

        return UserCredentials.from_authorized_user_info(data, scopes=self._scopes)
