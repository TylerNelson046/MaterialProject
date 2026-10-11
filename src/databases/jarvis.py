"""JARVIS-DFT database client (Figshare dft_3d via jarvis-tools)."""

from __future__ import annotations

from typing import Any, Iterator

from config.constants import DataSource
from config.settings import Settings, get_settings
from databases.base import DatabaseClient


class JarvisClient(DatabaseClient):
    """Load JARVIS-DFT 3D records from the local Figshare cache / download."""

    source = DataSource.JARVIS
    dataset_name = "dft_3d"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._cache: list[dict[str, Any]] | None = None

    def _load(self) -> list[dict[str, Any]]:
        if self._cache is None:
            from jarvis.db.figshare import data

            print(f"Loading JARVIS dataset '{self.dataset_name}'...")
            records = data(self.dataset_name)
            if not isinstance(records, list):
                raise TypeError(f"Unexpected JARVIS payload type: {type(records)!r}")
            self._cache = records
            print(f"JARVIS {self.dataset_name}: {len(self._cache)} records available")
        return self._cache

    def search(
        self,
        *,
        max_materials: int | None = None,
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        return list(self.iter_search(max_materials=max_materials))

    def iter_search(
        self,
        *,
        max_materials: int | None = None,
        **_kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        records = self._load()
        limit = len(records) if max_materials is None else min(int(max_materials), len(records))
        for record in records[:limit]:
            yield record

    def get_by_id(self, source_id: str) -> dict[str, Any] | None:
        for record in self._load():
            if str(record.get("jid") or "") == source_id:
                return record
        return None
