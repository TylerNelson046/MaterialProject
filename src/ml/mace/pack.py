"""Build the Phase 4.3 MACE structure training pack under ``mace_exports``."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import PROJECT_ROOT
from ml.mace.audit import default_audit_dir
from ml.mace.config import load_mace_config
from ml.mace.elements import MACE_MPA0_ATOMIC_NUMBERS
from ml.mace.structures import REASON_OK, load_structure

MACE_PACK_VERSION = "4.3.0"

# Slim columns kept in the structure pack (no composition frac_* features).
PROVENANCE_COLS = (
    "material_id",
    "source",
    "source_id",
    "formula_pretty",
    "chemsys",
    "nelements",
    "nsites",
    "volume",
    "density",
    "space_group_number",
    "space_group_symbol",
    "crystal_system",
    "structure_path",
    "n_sources",
    "group_id",
    "split",
    "label",
    # Phase 4.7 — geometry holdout (density); present on Ready exports.
    "holdout_cod_geometry",
)

STRUCTURE_META_COLS = (
    "n_atoms",
    "structure_formula",
    "n_elements_structure",
    "max_atomic_number",
    "load_reason",
    "mace_train_ok",
)


@dataclass(frozen=True)
class MacePackResult:
    """Artifacts written by ``export_mace_pack``."""

    target: str
    pack_dir: Path
    target_parquet: Path
    manifest_path: Path
    dataset_card_path: Path
    n_source_rows: int
    n_pack_rows: int
    n_train: int
    n_val: int
    n_test: int
    n_excluded_missing_or_bad: int
    n_oversized: int
    max_atoms: int | None


def mace_pack_dir(config: dict[str, Any] | None = None) -> Path:
    cfg = config if config is not None else load_mace_config()
    export = cfg.get("export") or {}
    return PROJECT_ROOT / str(export.get("mace_pack_dir") or "data/processed/mace_exports")


def _training_export_paths(config: dict[str, Any] | None = None) -> dict[str, Path]:
    cfg = config if config is not None else load_mace_config()
    export = cfg.get("export") or {}
    root = PROJECT_ROOT / str(export.get("dir") or "data/processed/training_exports")
    sub = str(export.get("targets_subdir") or "targets")
    return {
        "root": root,
        "targets": root / sub,
        "db_freeze": root / "db_freeze.json",
        "export_manifest": root / "export_manifest.json",
        "dataset_card": root / "DATASET_CARD.md",
    }


def _target_parquet_name(target: str, config: dict[str, Any] | None = None) -> str:
    cfg = config if config is not None else load_mace_config()
    spec = (cfg.get("targets") or {}).get(target) or {}
    return str(spec.get("parquet") or f"{target}.parquet")


def _normalize_audit_meta(audit: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "material_id",
        "load_ok",
        "reason",
        "n_atoms",
        "formula",
        "n_elements",
        "max_atomic_number",
        "structure_path",
    ]
    meta = audit.loc[:, [c for c in keep if c in audit.columns]].copy()
    return meta.rename(
        columns={
            "formula": "structure_formula",
            "n_elements": "n_elements_structure",
            "reason": "load_reason",
        }
    )


def _load_or_build_structure_meta(
    frame: pd.DataFrame,
    *,
    target: str,
    config: dict[str, Any],
    require_audit: bool,
    workers: int,
    reuse_audit_from: str | None = None,
) -> pd.DataFrame:
    """Return per-material structure metadata aligned to ``frame.material_id``."""
    audit_path = default_audit_dir(target, config) / "audit.parquet"
    if audit_path.is_file():
        return _normalize_audit_meta(pd.read_parquet(audit_path))

    if reuse_audit_from:
        reused = default_audit_dir(reuse_audit_from, config) / "audit.parquet"
        if reused.is_file():
            return _normalize_audit_meta(pd.read_parquet(reused))

    if require_audit:
        hint = (
            f"Structure audit not found at {audit_path}. "
            "Run: python scripts/audit_mace_structures.py "
            f"--target {target}"
        )
        if reuse_audit_from:
            hint += (
                f" (or provide --reuse-audit-from {reuse_audit_from} with that "
                "target's audit present)"
            )
        raise FileNotFoundError(hint)

    # Slow path: load each row (only used when audit is absent).
    rows: list[dict[str, Any]] = []
    paths = frame[["material_id", "structure_path"]].itertuples(index=False)
    for material_id, structure_path in paths:
        sp = None if pd.isna(structure_path) else str(structure_path)
        result = load_structure(sp, attach_atoms=False)
        rows.append(
            {
                "material_id": str(material_id),
                "load_ok": bool(result.ok),
                "load_reason": result.reason,
                "n_atoms": result.n_atoms,
                "structure_formula": result.formula,
                "n_elements_structure": (
                    len(set(result.atomic_numbers)) if result.atomic_numbers else None
                ),
                "max_atomic_number": (
                    max(result.atomic_numbers) if result.atomic_numbers else None
                ),
                "structure_path": result.path,
            }
        )
    _ = workers  # reserved for a future parallel slow path
    return pd.DataFrame(rows)


def _label_unit_for_target(target: str, config: dict[str, Any]) -> str:
    spec = (config.get("targets") or {}).get(target) or {}
    if spec.get("label_unit"):
        return str(spec["label_unit"])
    if target == "density":
        return "g/cm^3"
    if target in {"formation_energy_per_atom", "energy_above_hull"}:
        return str((config.get("energy") or {}).get("label_unit") or "eV/atom")
    return str((config.get("energy") or {}).get("label_unit") or "eV/atom")


def export_mace_pack(
    target: str = "formation_energy_per_atom",
    *,
    max_atoms: int | None = None,
    require_audit: bool = True,
    reuse_audit_from: str | None = None,
    workers: int = 1,
    config: dict[str, Any] | None = None,
    out_dir: Path | None = None,
) -> MacePackResult:
    """Write a structure-gated training pack for ``target``.

    Keeps export splits / labels; includes only rows that load as valid ASE
    structures under the MACE-MPA-0 element gate. Optional ``max_atoms`` marks
    oversized cells with ``mace_train_ok=False`` (still written; filter at train).

    ``reuse_audit_from`` lets Phase 4.7 packs reuse the formation-energy audit
    when material_ids overlap (avoids re-auditing ~200k shared structures).
    """
    cfg = config if config is not None else load_mace_config()
    structure_cfg = cfg.get("structure") or {}
    if max_atoms is None:
        raw = structure_cfg.get("max_atoms_train")
        max_atoms = int(raw) if raw is not None else None
    if reuse_audit_from is None:
        spec = (cfg.get("targets") or {}).get(target) or {}
        raw_reuse = spec.get("reuse_audit_from")
        reuse_audit_from = str(raw_reuse) if raw_reuse else None

    paths = _training_export_paths(cfg)
    src_parquet = paths["targets"] / _target_parquet_name(target, cfg)
    if not src_parquet.is_file():
        raise FileNotFoundError(f"Training export target not found: {src_parquet}")

    available = pd.read_parquet(src_parquet).columns.tolist()
    cols = [c for c in PROVENANCE_COLS if c in available]
    if "label" not in cols or "split" not in cols or "material_id" not in cols:
        raise ValueError(f"{src_parquet} missing required columns label/split/material_id")
    source = pd.read_parquet(src_parquet, columns=cols)
    n_source = int(len(source))

    meta = _load_or_build_structure_meta(
        source,
        target=target,
        config=cfg,
        require_audit=require_audit,
        workers=workers,
        reuse_audit_from=reuse_audit_from,
    )
    # Prefer export structure_path; fall back to audit path if export null.
    merged = source.merge(meta, on="material_id", how="left", suffixes=("", "_audit"))
    if "structure_path_audit" in merged.columns:
        merged["structure_path"] = merged["structure_path"].fillna(
            merged["structure_path_audit"]
        )
        merged = merged.drop(columns=["structure_path_audit"])

    merged["load_ok"] = merged["load_ok"].fillna(False).astype(bool)
    merged["load_reason"] = merged["load_reason"].fillna("MISSING_PATH")

    pack = merged.loc[merged["load_ok"] & (merged["load_reason"] == REASON_OK)].copy()
    n_excluded = n_source - int(len(pack))

    if max_atoms is not None:
        oversized = pack["n_atoms"].fillna(max_atoms + 1).astype(int) > int(max_atoms)
        pack["mace_train_ok"] = ~oversized
        n_oversized = int(oversized.sum())
    else:
        pack["mace_train_ok"] = True
        n_oversized = 0

    # Stable column order.
    ordered = [c for c in PROVENANCE_COLS if c in pack.columns]
    ordered.extend([c for c in STRUCTURE_META_COLS if c in pack.columns])
    pack = pack.loc[:, ordered]

    pack_root = out_dir or mace_pack_dir(cfg)
    targets_dir = pack_root / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = targets_dir / f"{target}.parquet"
    pack.to_parquet(out_parquet, index=False)

    # Copy freeze pointer from training export when present.
    if paths["db_freeze"].is_file():
        shutil.copy2(paths["db_freeze"], pack_root / "db_freeze.json")

    split_counts = Counter(pack["split"].astype(str).tolist())
    element_cov = {
        "foundation": "MACE-MPA-0",
        "n_elements_covered": len(MACE_MPA0_ATOMIC_NUMBERS),
        "atomic_numbers": sorted(MACE_MPA0_ATOMIC_NUMBERS),
    }
    n_atoms_series = pack["n_atoms"].dropna().astype(int)
    atom_stats = {
        "min": int(n_atoms_series.min()) if len(n_atoms_series) else None,
        "max": int(n_atoms_series.max()) if len(n_atoms_series) else None,
        "mean": float(n_atoms_series.mean()) if len(n_atoms_series) else None,
        "p95": float(n_atoms_series.quantile(0.95)) if len(n_atoms_series) else None,
    }

    train_export_manifest = None
    if paths["export_manifest"].is_file():
        train_export_manifest = json.loads(paths["export_manifest"].read_text(encoding="utf-8"))

    db_freeze = None
    freeze_path = pack_root / "db_freeze.json"
    if freeze_path.is_file():
        db_freeze = json.loads(freeze_path.read_text(encoding="utf-8"))

    reason_counts = Counter(merged["load_reason"].astype(str).tolist())
    manifest = {
        "mace_pack_version": MACE_PACK_VERSION,
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target": target,
        "label_unit": _label_unit_for_target(target, cfg),
        "training_export": {
            "dir": str(paths["root"].relative_to(PROJECT_ROOT)),
            "export_version": (train_export_manifest or {}).get("export_version"),
            "source_parquet": str(src_parquet.relative_to(PROJECT_ROOT)),
        },
        "db_freeze": db_freeze,
        "structure_gate": {
            "require_load_ok": True,
            "element_coverage": element_cov,
            "max_atoms_train": max_atoms,
            "reuse_audit_from": reuse_audit_from,
            "audit_parquet": str(
                (default_audit_dir(target, cfg) / "audit.parquet").relative_to(PROJECT_ROOT)
            )
            if (default_audit_dir(target, cfg) / "audit.parquet").is_file()
            else (
                str(
                    (
                        default_audit_dir(reuse_audit_from, cfg) / "audit.parquet"
                    ).relative_to(PROJECT_ROOT)
                )
                if reuse_audit_from
                and (default_audit_dir(reuse_audit_from, cfg) / "audit.parquet").is_file()
                else None
            ),
        },
        "counts": {
            "source_rows": n_source,
            "pack_rows": int(len(pack)),
            "excluded_missing_or_bad": n_excluded,
            "oversized": n_oversized,
            "mace_train_ok": int(pack["mace_train_ok"].sum()),
            "split": {
                "train": int(split_counts.get("train", 0)),
                "val": int(split_counts.get("val", 0)),
                "test": int(split_counts.get("test", 0)),
            },
            "load_reason_on_source": dict(reason_counts),
        },
        "n_atoms": atom_stats,
        "columns": list(pack.columns),
        "paths": {
            "pack_dir": str(pack_root.relative_to(PROJECT_ROOT)),
            "target_parquet": str(out_parquet.relative_to(PROJECT_ROOT)),
            "dataset_card": "DATASET_CARD.md",
            "manifest": "export_manifest.json",
        },
        "split_contract": {
            "allow_resplit": False,
            "column": "split",
            "note": "Inherited from training_exports fingerprint-grouped splits (seed 42).",
        },
    }

    # Per-target manifest (top-level export_manifest.json is the latest write;
    # also keep a stable per-target copy as packs accumulate).
    targets_manifest = targets_dir / f"{target}.manifest.json"
    targets_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = pack_root / "export_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # Index of all target packs present.
    index = {
        "mace_pack_version": MACE_PACK_VERSION,
        "updated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "targets": sorted(p.stem for p in targets_dir.glob("*.parquet")),
    }
    (pack_root / "targets_index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    card_path = pack_root / "DATASET_CARD.md"
    card_path.write_text(_dataset_card_md(manifest), encoding="utf-8")

    return MacePackResult(
        target=target,
        pack_dir=pack_root,
        target_parquet=out_parquet,
        manifest_path=manifest_path,
        dataset_card_path=card_path,
        n_source_rows=n_source,
        n_pack_rows=int(len(pack)),
        n_train=int(split_counts.get("train", 0)),
        n_val=int(split_counts.get("val", 0)),
        n_test=int(split_counts.get("test", 0)),
        n_excluded_missing_or_bad=n_excluded,
        n_oversized=n_oversized,
        max_atoms=max_atoms,
    )


def _dataset_card_md(manifest: dict[str, Any]) -> str:
    counts = manifest["counts"]
    split = counts["split"]
    freeze = manifest.get("db_freeze") or {}
    atoms = manifest.get("n_atoms") or {}
    gate = manifest.get("structure_gate") or {}
    lines = [
        f"# MACE structure training pack — `{manifest['target']}`",
        "",
        f"**Pack version:** {manifest['mace_pack_version']}  ",
        f"**Generated (UTC):** {manifest['generated_at_utc']}  ",
        f"**Label unit:** {manifest['label_unit']}  ",
        "",
        "## Provenance",
        "",
        f"- Training export: `{manifest['training_export']['dir']}` "
        f"(version `{manifest['training_export'].get('export_version')}`)",
        f"- Source parquet: `{manifest['training_export']['source_parquet']}`",
        f"- DuckDB freeze SHA-256: `{freeze.get('sha256')}`",
        f"- Freeze snapshot: `{freeze.get('snapshot_path')}`",
        "",
        "## Structure gate",
        "",
        "- Only rows with successful ASE load (`load_reason=OK`).",
        f"- Element coverage: {gate.get('element_coverage', {}).get('foundation')} "
        f"({gate.get('element_coverage', {}).get('n_elements_covered')} elements).",
        f"- `max_atoms_train`: `{gate.get('max_atoms_train')}` "
        "(oversized rows kept with `mace_train_ok=false`).",
        f"- Audit used: `{gate.get('audit_parquet')}`",
        "",
        "## Counts",
        "",
        "| Metric | N |",
        "|--------|--:|",
        f"| Source export rows | {counts['source_rows']} |",
        f"| Pack rows (load OK) | {counts['pack_rows']} |",
        f"| Excluded (missing/bad structure) | {counts['excluded_missing_or_bad']} |",
        f"| Oversized vs max_atoms | {counts['oversized']} |",
        f"| `mace_train_ok` | {counts['mace_train_ok']} |",
        f"| Train | {split['train']} |",
        f"| Val | {split['val']} |",
        f"| Test | {split['test']} |",
        "",
        "## n_atoms",
        "",
    ]
    if atoms.get("mean") is not None:
        lines.append(
            f"- min / max / mean / p95: "
            f"{atoms.get('min')} / {atoms.get('max')} / "
            f"{float(atoms['mean']):.2f} / {float(atoms['p95']):.2f}"
        )
    else:
        lines.append("- (empty pack)")
    lines.extend(
        [
            "",
            "## Split contract",
            "",
            "- **No resplit.** Use the inherited `split` column from the training export.",
            "- Fingerprint-group leak safety unchanged (seed 42).",
            "",
            "## Files",
            "",
            "```text",
            "mace_exports/",
            "  DATASET_CARD.md",
            "  export_manifest.json",
            "  db_freeze.json",
            "  targets/<target>.parquet",
            "  structure_audit/<target>/   # from Phase 4.2",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def load_mace_pack_frame(
    target: str = "formation_energy_per_atom",
    *,
    train_ok_only: bool = True,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Load a packed target table (optional ``mace_train_ok`` filter)."""
    cfg = config if config is not None else load_mace_config()
    path = mace_pack_dir(cfg) / "targets" / f"{target}.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"MACE pack not found: {path}. Run scripts/export_mace_pack.py first."
        )
    frame = pd.read_parquet(path)
    if train_ok_only and "mace_train_ok" in frame.columns:
        frame = frame.loc[frame["mace_train_ok"]].copy()
    return frame
