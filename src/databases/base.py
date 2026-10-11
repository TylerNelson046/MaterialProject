"""Common interface for external materials database adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from config.constants import DataSource


class DatabaseClient(ABC):
    """Thin adapter for retrieving data from one external source."""

    source: DataSource

    @abstractmethod
    def search(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Return source documents as plain dictionaries."""

    @abstractmethod
    def get_by_id(self, source_id: str) -> dict[str, Any] | None:
        """Fetch a single document by its source-native ID."""
