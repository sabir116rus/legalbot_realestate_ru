from __future__ import annotations

import json
import logging
import mimetypes
from pathlib import Path
from typing import Any, Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials as UserCredentials

LOGGER = logging.getLogger(__name__)

DEFAULT_SCOPES = ["https://www.googleapis.com/auth/drive.file"]


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

        try:
            list_kwargs: dict[str, Any] = {
                "q": query,
                "spaces": "drive",
                "fields": "files(id)",
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

        media = MediaFileUpload(str(resolved_path), mimetype=detected_mime_type, resumable=False)

        if files:
            file_id = files[0]["id"]
            try:
                service.files().update(
                    fileId=file_id,
                    media_body=media,
                    supportsAllDrives=True,
                ).execute()
                LOGGER.info("Updated Google Drive file '%s' (%s)", upload_name, file_id)
            except HttpError as exc:
                LOGGER.error("Failed to update Google Drive file '%s': %s", upload_name, exc)
        else:
            metadata = {"name": upload_name, "parents": [folder_id]}
            try:
                created = (
                    service.files()
                    .create(
                        body=metadata,
                        media_body=media,
                        fields="id",
                        supportsAllDrives=True,
                    )
                    .execute()
                )
                LOGGER.info("Created Google Drive file '%s' (%s)", upload_name, created.get("id"))
            except HttpError as exc:
                LOGGER.error("Failed to create Google Drive file '%s': %s", upload_name, exc)

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

        try:
            creds = self._load_credentials(self._credentials_file)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.error("Failed to load Google Drive credentials: %s", exc)
            return None

        try:
            self._service = build("drive", "v3", credentials=creds)
        except Exception as exc:  # pragma: no cover - API discovery failures
            LOGGER.error("Failed to initialise Google Drive service: %s", exc)
            return None

        return self._service

    def _load_credentials(self, credentials_path: Path):
        data = json.loads(credentials_path.read_text(encoding="utf-8"))
        cred_type = data.get("type")

        if cred_type == "service_account":
            return service_account.Credentials.from_service_account_info(data, scopes=self._scopes)

        return UserCredentials.from_authorized_user_info(data, scopes=self._scopes)
