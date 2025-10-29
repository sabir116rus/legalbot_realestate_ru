from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Iterable, Set


class ConsentStore:
    """Persistent storage for user consent flags."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    async def load_consents(self) -> Set[int]:
        """Load all consented user identifiers from storage."""

        async with self._lock:
            return await self._read_consents_unlocked()

    async def add_consent(self, user_id: int) -> None:
        """Persist consent for a user identifier."""

        async with self._lock:
            consents = await self._read_consents_unlocked()
            consents.add(int(user_id))
            await self._write_consents_unlocked(consents)

    async def remove_consent(self, user_id: int) -> None:
        """Remove consent for a user identifier if present."""

        async with self._lock:
            consents = await self._read_consents_unlocked()
            consents.discard(int(user_id))
            await self._write_consents_unlocked(consents)

    async def _read_consents_unlocked(self) -> Set[int]:
        if not self._path.exists():
            return set()

        try:
            raw = await asyncio.to_thread(self._path.read_text, encoding="utf-8")
        except FileNotFoundError:
            return set()

        raw = raw.strip()
        if not raw:
            return set()

        try:
            values: Iterable[int] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Consent store file is corrupted") from exc

        return {int(value) for value in values}

    async def _write_consents_unlocked(self, consents: Set[int]) -> None:
        serialized = json.dumps(sorted(consents))
        await asyncio.to_thread(self._atomic_write, serialized)

    def _atomic_write(self, payload: str) -> None:
        tmp_path = self._path.with_name(self._path.name + ".tmp")
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.replace(self._path)
