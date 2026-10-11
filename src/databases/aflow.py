"""AFLOW AFLUX API client (ICSD catalog summaries)."""

from __future__ import annotations

import time
from typing import Any, Iterator
from urllib.parse import quote

import requests

from config.constants import DataSource
from config.settings import Settings, get_settings
from databases.base import DatabaseClient

DEFAULT_BASE_URL = "https://aflow.org/API/aflux/"

# Property matchers requested from AFLUX (comma-separated summons).
DEFAULT_MATCHERS: tuple[str, ...] = (
    "auid",
    "aurl",
    "compound",
    "species",
    "nspecies",
    "natoms",
    "density",
    "volume_cell",
    "spacegroup_relax",
    "Pearson_symbol_relax",
    "Egap",
    "enthalpy_formation_atom",
    "energy_atom",
    "ael_bulk_modulus_vrh",
    "ael_shear_modulus_vrh",
    "catalog",
)


class AFLOWClient(DatabaseClient):
    """Paginated AFLOW AFLUX summary client (focused ICSD catalog by default)."""

    source = DataSource.AFLOW

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        cfg = self.settings.database_config.get("aflow", {})
        self.base_url = str(cfg.get("base_url") or DEFAULT_BASE_URL).rstrip("/") + "/"
        self.page_size = int(cfg.get("page_size") or 500)
        self.request_timeout = int(cfg.get("timeout_seconds") or 180)
        self.max_retries = int(cfg.get("max_retries") or 8)
        self.catalog = cfg.get("catalog", "icsd")
        if self.catalog in {None, "none", "null", False, ""}:
            self.catalog = None
        else:
            self.catalog = str(self.catalog).lower()
        matchers = cfg.get("matchers")
        if isinstance(matchers, list) and matchers:
            self.matchers = tuple(str(m) for m in matchers)
        else:
            self.matchers = DEFAULT_MATCHERS

    def _summons(self, *, page: int, page_size: int) -> str:
        parts = list(self.matchers)
        if self.catalog:
            parts.append(f"catalog({self.catalog})")
        parts.append(f"$paging({page},{page_size})")
        parts.append("format(json)")
        return ",".join(parts)

    def _get(self, summons: str) -> list[dict[str, Any]]:
        url = f"{self.base_url}?{summons}"
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(
                    url,
                    timeout=self.request_timeout,
                    headers={"Accept": "application/json"},
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, list):
                    raise TypeError(f"Unexpected AFLOW payload type: {type(payload)!r}")
                return [row for row in payload if isinstance(row, dict)]
            except Exception as exc:  # noqa: BLE001 - retry network/API failures
                last_error = exc
                sleep_s = min(2**attempt, 30)
                print(f"AFLOW request failed (attempt {attempt}/{self.max_retries}): {exc}")
                time.sleep(sleep_s)
        raise RuntimeError(f"AFLOW request failed after retries: {last_error}")

    def count_available(self) -> int | None:
        """Best-effort count via binary search on paging (AFLUX has no meta.count)."""
        size = 100
        lo, hi = 1, 5000
        last = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            rows = self._get(self._summons(page=mid, page_size=size))
            if not rows:
                hi = mid - 1
            else:
                last = mid
                lo = mid + 1
            time.sleep(0.05)
        if last == 0:
            return 0
        final = self._get(self._summons(page=last, page_size=size))
        return (last - 1) * size + len(final)

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
        **_kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        size = max(1, min(int(page_size or self.page_size), 1000))
        page = 1
        yielded = 0

        while True:
            if max_materials is not None and yielded >= int(max_materials):
                break

            rows = self._get(self._summons(page=page, page_size=size))
            if not rows:
                break

            for row in rows:
                auid = row.get("auid")
                if not auid:
                    continue
                yield row
                yielded += 1
                if max_materials is not None and yielded >= int(max_materials):
                    return

            if len(rows) < size:
                break
            page += 1
            if yielded % 5000 == 0 and yielded:
                print(f"  ... fetched {yielded} AFLOW records")

    def get_by_id(self, source_id: str) -> dict[str, Any] | None:
        auid = source_id
        if auid.startswith("aflow-"):
            auid = "aflow:" + auid.removeprefix("aflow-")
        elif not auid.startswith("aflow:"):
            auid = f"aflow:{auid}"
        # AFLUX matchers: auid(value)
        encoded = quote(auid, safe=":")
        summons = f"auid({encoded})," + ",".join(self.matchers) + ",format(json)"
        rows = self._get(summons)
        return rows[0] if rows else None
