"""Download Materials Project insertion-electrode documents via local Delta cache."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pyarrow.dataset as ds
from mp_api.client import MPRester

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config.settings import clear_settings_cache, get_settings  # noqa: E402

ELECTRODE_COLUMNS = [
    "material_ids",
    "battery_formula",
    "working_ion",
    "average_voltage",
    "capacity_grav",
    "capacity_vol",
    "energy_grav",
    "energy_vol",
    "num_steps",
    "framework_formula",
    "chemsys",
    "formula_charge",
    "formula_discharge",
    "stability_charge",
    "stability_discharge",
]

DELTA_PATH = Path.home() / "mp_datasets" / "build" / "collections" / "insertion-electrodes"


def _ensure_delta(api_key: str, *, force_renew: bool = False) -> Path:
    if DELTA_PATH.exists() and not force_renew:
        return DELTA_PATH
    with MPRester(api_key, use_document_model=False, force_renew=force_renew) as mpr:
        mpr.materials.insertion_electrodes.search(
            fields=["average_voltage"],
            chunk_size=1,
            num_chunks=1,
        )
    if not DELTA_PATH.exists():
        raise FileNotFoundError(f"Expected electrode Delta at {DELTA_PATH}")
    return DELTA_PATH


def _cell_to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "as_py"):
        value = value.as_py()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_cell_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _cell_to_jsonable(v) for k, v in value.items()}
    return value


def main() -> int:
    clear_settings_cache()
    settings = get_settings()
    out_dir = settings.raw_source_dir("materials_project")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "insertion_electrodes.jsonl"

    api_key = settings.require_mp_api_key()
    delta = _ensure_delta(api_key)
    table = ds.dataset(str(delta), format="parquet").to_table(columns=ELECTRODE_COLUMNS)
    count = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for batch in table.to_batches(max_chunksize=500):
            rows = batch.to_pylist()
            for row in rows:
                clean = {k: _cell_to_jsonable(v) for k, v in row.items()}
                handle.write(json.dumps(clean, ensure_ascii=True) + "\n")
                count += 1
    print(f"Wrote {count:,} insertion electrodes → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
