"""Resumable frozen MACE formation-energy cache for hull / scoring."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import PROJECT_ROOT
from ml.io_util import relpath_under, write_json
from ml.mace.calculator import load_foundation_calculator
from ml.mace.config import foundation_model_id, load_mace_config
from ml.mace.pack import load_mace_pack_frame, mace_pack_dir
from ml.mace.reference import (
    PROTOCOL_ELEMENTAL_BULK_V1,
    formation_energy_per_atom,
    load_formation_references,
)
from ml.mace.structures import load_structure


@dataclass(frozen=True)
class EformCacheResult:
    """Cached formation energies for a pack slice."""

    target: str
    cache_path: Path
    n_rows: int
    n_failed_loads: int
    n_failed_infer: int
    protocol: str


def default_eform_cache_path(
    target: str = "formation_energy_per_atom",
    *,
    protocol: str = PROTOCOL_ELEMENTAL_BULK_V1,
    model_id: str | None = None,
    config: dict[str, Any] | None = None,
) -> Path:
    cfg = config if config is not None else load_mace_config()
    mid = model_id or foundation_model_id(cfg)
    return (
        mace_pack_dir(cfg)
        / "features"
        / target
        / str(mid)
        / f"eform_{protocol}.parquet"
    )


def build_eform_cache(
    target: str = "formation_energy_per_atom",
    *,
    splits: tuple[str, ...] | None = None,
    limit_per_split: int | None = None,
    device: str | None = "cpu",
    model_id: str | None = None,
    cache_path: Path | None = None,
    overwrite: bool = False,
    config: dict[str, Any] | None = None,
    project_root: Path | None = None,
    progress_every: int = 100,
) -> EformCacheResult:
    """Infer frozen E_form for pack rows; resume by material_id."""
    cfg = config if config is not None else load_mace_config()
    root = project_root or PROJECT_ROOT
    pack = load_mace_pack_frame(target, train_ok_only=True, config=cfg)
    if splits is not None:
        pack = pack.loc[pack["split"].astype(str).isin(set(splits))].copy()
    if limit_per_split is not None:
        parts = [
            g.head(int(limit_per_split))
            for _, g in pack.groupby(pack["split"].astype(str), sort=False)
        ]
        pack = pd.concat(parts, ignore_index=True) if parts else pack.iloc[0:0]

    calc, resolved_model, ckpt_path, resolved_device = load_foundation_calculator(
        model_id=model_id, device=device, config=cfg
    )
    mus, protocol = load_formation_references(
        resolved_model, calc=calc, config=cfg
    )
    out_path = cache_path or default_eform_cache_path(
        target, protocol=protocol, model_id=resolved_model, config=cfg
    )
    if not out_path.is_absolute():
        out_path = root / out_path

    existing: pd.DataFrame | None = None
    if out_path.is_file() and not overwrite:
        existing = pd.read_parquet(out_path)
        have = set(existing["material_id"].astype(str))
        pack = pack.loc[~pack["material_id"].astype(str).isin(have)].copy()
        if pack.empty:
            return EformCacheResult(
                target=target,
                cache_path=out_path,
                n_rows=int(len(existing)),
                n_failed_loads=0,
                n_failed_infer=0,
                protocol=protocol,
            )
        print(
            f"[{target}] eform cache resume: {len(have)} cached, "
            f"{len(pack)} remaining …",
            flush=True,
        )

    rows: list[dict[str, Any]] = []
    n_fail_load = 0
    n_fail_infer = 0
    for i, rec in enumerate(pack.itertuples(index=False), start=1):
        loaded = load_structure(getattr(rec, "structure_path", None), attach_atoms=True)
        if not loaded.ok or loaded.atoms is None:
            n_fail_load += 1
            continue
        atoms = loaded.atoms
        try:
            atoms.calc = calc
            e_tot = float(atoms.get_potential_energy())
            zs = [int(z) for z in atoms.get_atomic_numbers()]
            if any(z not in mus for z in zs):
                n_fail_infer += 1
                continue
            e_form = formation_energy_per_atom(e_tot, zs, mus)
            e_pa = e_tot / float(len(atoms))
        except Exception:  # noqa: BLE001
            n_fail_infer += 1
            continue
        rows.append(
            {
                "material_id": str(rec.material_id),
                "split": str(rec.split),
                "structure_path": loaded.path,
                "formula": loaded.formula or getattr(rec, "formula_pretty", None),
                "formula_pretty": getattr(rec, "formula_pretty", None),
                "chemsys": getattr(rec, "chemsys", None),
                "label": float(rec.label) if hasattr(rec, "label") else None,
                "energy_total_eV": e_tot,
                "energy_per_atom_eV": e_pa,
                "e_form_per_atom": e_form,
                "y_pred_formation": e_form,
                "n_atoms": int(len(atoms)),
            }
        )
        if progress_every and i % int(progress_every) == 0:
            print(f"[{target}] eform cache {i}/{len(pack)} …", flush=True)

    new = pd.DataFrame(rows)
    if existing is not None and len(existing):
        frame = pd.concat([existing, new], ignore_index=True)
        frame = frame.drop_duplicates(subset=["material_id"], keep="last")
    else:
        frame = new
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)
    write_json(
        out_path.with_suffix(".json"),
        {
            "target": target,
            "protocol": protocol,
            "model_id": resolved_model,
            "checkpoint": relpath_under(ckpt_path, root),
            "device": resolved_device,
            "n_rows": int(len(frame)),
            "n_failed_loads": n_fail_load,
            "n_failed_infer": n_fail_infer,
            "created_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit_per_split": limit_per_split,
            "splits": list(splits) if splits else None,
        },
    )
    return EformCacheResult(
        target=target,
        cache_path=out_path,
        n_rows=int(len(frame)),
        n_failed_loads=n_fail_load,
        n_failed_infer=n_fail_infer,
        protocol=protocol,
    )


def load_eform_cache(
    target: str = "formation_energy_per_atom",
    *,
    protocol: str = PROTOCOL_ELEMENTAL_BULK_V1,
    cache_path: Path | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    cfg = config if config is not None else load_mace_config()
    path = cache_path or default_eform_cache_path(
        target, protocol=protocol, config=cfg
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"E_form cache missing: {path}. "
            "Run scripts/cache_mace_eform.py first."
        )
    return pd.read_parquet(path)
