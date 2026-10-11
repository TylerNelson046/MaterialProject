"""OQMD REST API client (formationenergy endpoint)."""

from __future__ import annotations

import time
from typing import Any, Iterator

import requests

from config.constants import DataSource
from config.settings import Settings, get_settings
from databases.base import DatabaseClient

DEFAULT_BASE_URL = "https://oqmd.org/oqmdapi/formationenergy"


class OQMDClient(DatabaseClient):
    """Paginated OQMD formation-energy summary client."""

    source = DataSource.OQMD

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        cfg = self.settings.database_config.get("oqmd", {})
        self.base_url = str(cfg.get("base_url") or DEFAULT_BASE_URL)
        self.page_size = int(cfg.get("page_size") or 100)
        self.request_timeout = int(cfg.get("timeout_seconds") or 180)
        self.max_retries = int(cfg.get("max_retries") or 12)
        # Prefer on-hull / near-hull materials rather than the full 1.4M catalog.
        self.stability_max = cfg.get("stability_max", 0.0)
        if self.stability_max in {None, "none", "null"}:
            self.stability_max = None
        else:
            self.stability_max = float(self.stability_max)
        # Bulk summary pulls exclude sites for speed; structure download includes them.
        self.include_sites = bool(cfg.get("include_sites", False))

    def _filter_extra(self) -> list[tuple[str, Any]]:
        if self.stability_max is None:
            return []
        # OQMD "stability" is energy above hull (eV/atom).
        return [("filter", f"stability<={self.stability_max}")]

    def _session(self) -> requests.Session:
        """Session that ignores env/system proxies (Cursor sandbox proxy often 502s)."""
        session = requests.Session()
        session.trust_env = False
        session.headers.update({"Accept": "application/json"})
        return session

    def _get(self, params: list[tuple[str, Any]]) -> dict[str, Any]:
        """GET with ordered query params.

        OQMD field exclusion uses the literal key ``fields!`` (not ``fields=!…``).
        """
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with self._session() as session:
                    response = session.get(
                        self.base_url,
                        params=params,
                        timeout=self.request_timeout,
                        proxies={"http": None, "https": None},
                    )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise TypeError(f"Unexpected OQMD payload type: {type(payload)!r}")
                return payload
            except Exception as exc:  # noqa: BLE001 - retry network/API failures
                last_error = exc
                # Back off harder on gateway / timeout failures.
                sleep_s = min(2**attempt, 120)
                if "502" in str(exc) or "503" in str(exc) or "504" in str(exc):
                    sleep_s = min(30 * attempt, 180)
                print(f"OQMD request failed (attempt {attempt}/{self.max_retries}): {exc}")
                time.sleep(sleep_s)
        raise RuntimeError(f"OQMD request failed after retries: {last_error}")

    def _page_params(
        self,
        *,
        limit: int,
        offset: int,
        extra: list[tuple[str, Any]] | None = None,
        include_sites: bool | None = None,
    ) -> list[tuple[str, Any]]:
        with_sites = self.include_sites if include_sites is None else include_sites
        params: list[tuple[str, Any]] = [
            ("limit", limit),
            ("offset", offset),
            ("format", "json"),
        ]
        if not with_sites:
            # Exclude atomic sites for speed/size; lattice remains via unit_cell.
            params.append(("fields!", "sites"))
        params.extend(self._filter_extra())
        if extra:
            params.extend(extra)
        return params

    def count_available(self) -> int:
        payload = self._get(self._page_params(limit=1, offset=0))
        meta = payload.get("meta") or {}
        return int(meta.get("data_available") or 0)

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
        page_size: int | None = None,
        start_offset: int = 0,
        include_sites: bool | None = None,
        **_kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        size = max(1, min(int(page_size or self.page_size), 1000))
        offset = max(0, int(start_offset))
        yielded = 0

        while True:
            if max_materials is not None and yielded >= int(max_materials):
                break

            this_limit = size
            if max_materials is not None:
                this_limit = min(size, int(max_materials) - yielded)

            payload = self._get(
                self._page_params(
                    limit=this_limit,
                    offset=offset,
                    include_sites=include_sites,
                )
            )
            rows = payload.get("data") or []
            if not rows:
                break

            for row in rows:
                if not isinstance(row, dict) or row.get("entry_id") is None:
                    continue
                yield row
                yielded += 1
                if max_materials is not None and yielded >= int(max_materials):
                    return

            offset += len(rows)
            next_link = (payload.get("links") or {}).get("next")
            if not next_link:
                break

            total_seen = offset
            if total_seen % 5000 == 0 and total_seen:
                print(f"  ... fetched through offset {total_seen} OQMD records")

    def get_by_id(
        self,
        source_id: str,
        *,
        include_sites: bool | None = True,
    ) -> dict[str, Any] | None:
        """Fetch a single formation-energy row.

        Note: the OQMD formationenergy filter DSL does not accept ``entry_id``
        as a property. Prefer paginated ``iter_search`` / structure download.
        This helper scans a short stability-filtered window as a best-effort.
        """
        entry_id = source_id.removeprefix("oqmd-").removeprefix("OQMD-")
        try:
            target = int(entry_id)
        except ValueError:
            return None
        # Best-effort: paginate a small window; not suitable for bulk lookup.
        for offset in range(0, 500, 100):
            payload = self._get(
                self._page_params(
                    limit=100,
                    offset=offset,
                    include_sites=include_sites,
                )
            )
            for row in payload.get("data") or []:
                if row.get("entry_id") == target:
                    return row
            if not (payload.get("links") or {}).get("next"):
                break
        return None
