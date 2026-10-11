"""Materials Project API client and local Delta dataset helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from mp_api.client import MPRester

from config.constants import DataSource
from config.settings import Settings, get_settings
from databases.base import DatabaseClient

DEFAULT_SUMMARY_FIELDS: tuple[str, ...] = (
    "material_id",
    "formula_pretty",
    "composition",
    "chemsys",
    "elements",
    "nelements",
    "nsites",
    "volume",
    "density",
    "symmetry",
    "formation_energy_per_atom",
    "energy_above_hull",
    "is_stable",
    "band_gap",
    "is_metal",
    "is_magnetic",
    "total_magnetization",
    "theoretical",
)

DEFAULT_MP_SUMMARY_DATASET = (
    Path.home() / "mp_datasets" / "build" / "collections" / "summary"
)


def mp_summary_dataset_path(settings: Settings | None = None) -> Path:
    """Return the local MP summary Delta table path."""
    settings = settings or get_settings()
    configured = settings.database_config.get("materials_project", {}).get("dataset_path")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else settings.project_root / path
    return DEFAULT_MP_SUMMARY_DATASET


def ensure_mp_summary_dataset(
    settings: Settings | None = None,
    *,
    force_renew: bool = False,
) -> Path:
    """Ensure the local summary Delta dataset exists; download via mp-api if needed."""
    settings = settings or get_settings()
    path = mp_summary_dataset_path(settings)

    if path.exists() and not force_renew:
        return path

    api_key = settings.require_mp_api_key()
    print(
        "Local MP Delta dataset missing or refresh requested — "
        "downloading via mp-api (one-time, may take several minutes)..."
    )
    with MPRester(api_key, use_document_model=False, force_renew=force_renew) as mpr:
        # Touching the summary endpoint materializes/refreshes the Delta cache.
        mpr.materials.summary.search(fields=["material_id"], chunk_size=1, num_chunks=1)

    if not path.exists():
        raise FileNotFoundError(
            f"Expected MP summary dataset at {path} after download, but it was not found."
        )
    return path


def _doc_to_dict(doc: Any) -> dict[str, Any]:
    if isinstance(doc, dict):
        return doc
    if hasattr(doc, "model_dump"):
        return doc.model_dump(mode="json")
    if hasattr(doc, "dict"):
        return doc.dict()
    raise TypeError(f"Unsupported MP document type: {type(doc)!r}")


class MaterialsProjectClient(DatabaseClient):
    """Retrieve summary / structure documents from the Materials Project API."""

    source = DataSource.MATERIALS_PROJECT

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.api_key = self.settings.require_mp_api_key()
        mp_cfg = self.settings.database_config.get("materials_project", {})
        configured = mp_cfg.get("fields") or list(DEFAULT_SUMMARY_FIELDS)
        self.fields = [f for f in configured if f != "structure"]
        self.summary_fields_with_structure = [*self.fields, "structure"]

    def search(
        self,
        *,
        max_materials: int | None = None,
        chunk_size: int | None = None,
        fields: list[str] | None = None,
        **criteria: Any,
    ) -> list[dict[str, Any]]:
        return list(
            self.iter_search(
                max_materials=max_materials,
                chunk_size=chunk_size,
                fields=fields,
                **criteria,
            )
        )

    def iter_search(
        self,
        *,
        max_materials: int | None = None,
        chunk_size: int | None = None,
        fields: list[str] | None = None,
        **criteria: Any,
    ) -> Iterator[dict[str, Any]]:
        """Legacy row iterator (slow for full catalog — prefer Delta export ingest)."""
        limit = self.settings.mp_max_materials if max_materials is None else max_materials
        size = chunk_size or self.settings.mp_chunk_size
        size = max(1, min(int(size), 1000))
        query_fields = fields or self.fields

        num_chunks: int | None = None
        if limit is not None:
            num_chunks = max(1, (int(limit) + size - 1) // size)

        with MPRester(self.api_key, use_document_model=False) as mpr:
            docs = mpr.materials.summary.search(
                fields=query_fields,
                chunk_size=size,
                num_chunks=num_chunks,
                **criteria,
            )
            count = 0
            for doc in docs:
                yield _doc_to_dict(doc)
                count += 1
                if limit is not None and count >= int(limit):
                    break

    def get_by_id(self, source_id: str) -> dict[str, Any] | None:
        results = self.search(max_materials=1, material_ids=[source_id])
        return results[0] if results else None
