"""Crystallography Open Database (COD) REST client."""

from __future__ import annotations

import time
from typing import Any, Iterator

import requests

from config.constants import DataSource
from config.settings import Settings, get_settings
from databases.base import DatabaseClient

DEFAULT_BASE_URL = "https://www.crystallography.net/cod/result"
DEFAULT_ENTRY_URL = "https://www.crystallography.net/cod"


class CODClient(DatabaseClient):
    """COD search + CIF download client (focused inorganic slice by default)."""

    source = DataSource.COD

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        cfg = self.settings.database_config.get("cod", {})
        self.base_url = str(cfg.get("base_url") or DEFAULT_BASE_URL)
        self.entry_url = str(cfg.get("entry_url") or DEFAULT_ENTRY_URL).rstrip("/")
        self.request_timeout = int(cfg.get("timeout_seconds") or 180)
        self.max_retries = int(cfg.get("max_retries") or 8)
        self.strictmin = int(cfg.get("strictmin") or 1)
        self.strictmax = int(cfg.get("strictmax") or 4)
        # Elements that must NOT appear (COD ``nel1``…``nel4``).
        exclude = cfg.get("exclude_elements") or ["C"]
        self.exclude_elements = [str(e) for e in exclude][:4]
        self.include_duplicates = bool(cfg.get("include_duplicates", False))
        self.include_theoretical = bool(cfg.get("include_theoretical", False))

    def _search_params(self, *, fmt: str) -> dict[str, Any]:
        params: dict[str, Any] = {
            "format": fmt,
            "strictmin": self.strictmin,
            "strictmax": self.strictmax,
        }
        for idx, element in enumerate(self.exclude_elements, start=1):
            params[f"nel{idx}"] = element
        if self.include_duplicates:
            params["include_duplicates"] = 1
        if self.include_theoretical:
            params["include_theoretical"] = 1
        return params

    def _get(self, url: str, *, params: dict[str, Any] | None = None) -> requests.Response:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(
                    url,
                    params=params,
                    timeout=self.request_timeout,
                    headers={"Accept": "*/*"},
                )
                response.raise_for_status()
                return response
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                sleep_s = min(2**attempt, 60)
                print(f"COD request failed (attempt {attempt}/{self.max_retries}): {exc}")
                time.sleep(sleep_s)
        raise RuntimeError(f"COD request failed after retries: {last_error}")

    def count_available(self) -> int:
        response = self._get(self.base_url, params=self._search_params(fmt="count"))
        text = response.text.strip()
        try:
            return int(text.split()[0])
        except (TypeError, ValueError, IndexError) as exc:
            raise RuntimeError(f"Unexpected COD count payload: {text!r}") from exc

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
        """Yield COD metadata dicts for the configured focused query.

        COD returns the full match set in one JSON response (no paging).
        """
        response = self._get(self.base_url, params=self._search_params(fmt="json"))
        payload = response.json()
        if not isinstance(payload, list):
            raise TypeError(f"Unexpected COD payload type: {type(payload)!r}")

        yielded = 0
        for row in payload:
            if not isinstance(row, dict):
                continue
            file_id = row.get("file")
            if file_id is None:
                continue
            yield row
            yielded += 1
            if max_materials is not None and yielded >= int(max_materials):
                return

    def get_by_id(self, source_id: str) -> dict[str, Any] | None:
        cod_id = source_id.removeprefix("cod-").removeprefix("COD-")
        response = self._get(
            self.base_url,
            params={"format": "json", "id": str(cod_id)},
        )
        payload = response.json()
        if isinstance(payload, list) and payload:
            return payload[0]
        return None

    def fetch_cif(self, cod_id: str | int) -> str:
        """Download CIF text for a COD entry id."""
        url = f"{self.entry_url}/{int(cod_id)}.cif"
        response = self._get(url)
        text = response.text
        if len(text) < 50:
            raise RuntimeError(f"COD CIF for {cod_id} looks empty")
        return text

    def cif_url(self, cod_id: str | int) -> str:
        return f"{self.entry_url}/{int(cod_id)}.cif"
