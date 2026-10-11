"""Export Phase 4 training package: targets, splits, hold-outs, dataset card.

Writes under ``data/processed/training_exports/``. Uses live DuckDB (post
Phase 0–3 hygiene). Also records path to a fresh pretrain freeze copy.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config.settings import clear_settings_cache, get_settings  # noqa: E402
from database.duckdb import MaterialsDatabase  # noqa: E402
from database.exclusion_rules import (  # noqa: E402
    CONFLICT_EXCLUDE_TARGETS,
    EXCLUDE_LABEL_CONFLICTS_V1,
    EXCLUSION_RULES_VERSION,
    export_rule_metadata,
)
from database.fingerprints import structure_fingerprint  # noqa: E402
from ml.composition_features import (  # noqa: E402
    COMP_STAT_COLS,
    FRAC_COLS,
    add_composition_features,
)

EXPORT_VERSION = "4.1.0"
SPLIT_SEED = 42
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
# test = remainder

# Ready + Niche heads (Step 5.2 targets).
TARGET_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "energy_above_hull",
        "tier": "Ready",
        "label_col": "energy_above_hull",
        "gate": "train_ok_thermo",
        "conflict_col": "conflict_energy_above_hull",
        "exclude_conflicts": True,
    },
    {
        "name": "formation_energy_per_atom",
        "tier": "Ready",
        "label_col": "formation_energy_per_atom",
        "gate": "train_ok_thermo",
        "conflict_col": "conflict_formation_energy",
        "exclude_conflicts": True,
    },
    {
        "name": "band_gap",
        "tier": "Ready",
        "label_col": "band_gap",
        "gate": "train_ok_gap",
        "conflict_col": "conflict_band_gap",
        "exclude_conflicts": True,
    },
    {
        "name": "density",
        "tier": "Ready",
        "label_col": "density",
        "gate": None,  # not deprecated + label present (applied in SQL)
        "conflict_col": "conflict_density",
        "exclude_conflicts": True,
        "extra_where": "density IS NOT NULL AND (deprecated IS NOT TRUE)",
    },
    {
        "name": "bulk_modulus_vrh",
        "tier": "Niche",
        "label_col": "bulk_modulus_vrh",
        "gate": "train_ok_elastic",
        "conflict_col": None,
    },
    {
        "name": "shear_modulus_vrh",
        "tier": "Niche",
        "label_col": "shear_modulus_vrh",
        "gate": "train_ok_elastic",
        "conflict_col": None,
    },
    {
        "name": "thermal_conductivity_agl",
        "tier": "Niche",
        "label_col": "thermal_conductivity",
        "gate": "train_ok_thermal_agl",
        "conflict_col": None,
        "extra_where": "thermal_conductivity_method = 'aflow-agl'",
    },
    {
        "name": "debye_temperature",
        "tier": "Niche",
        "label_col": "debye_temperature",
        "gate": "train_ok_thermal_agl",
        "conflict_col": None,
    },
    {
        "name": "heat_capacity_cv",
        "tier": "Niche",
        "label_col": "heat_capacity_cv",
        "gate": "train_ok_thermal_agl",
        "conflict_col": None,
    },
    {
        "name": "battery_capacity_grav",
        "tier": "Niche",
        "label_col": "capacity_grav",
        "gate": "train_ok_battery",
        "conflict_col": None,
    },
)

FEATURE_COLS = (
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
    "deprecated",
    "exclude_deprecated",
    "train_ok_thermo",
    "train_ok_gap",
    "train_ok_elastic",
    "train_ok_thermal_agl",
    "train_ok_battery",
    "label_conflict",
    "conflict_band_gap",
    "conflict_formation_energy",
    "conflict_energy_above_hull",
    "conflict_density",
    "thermal_conductivity_method",
    "thermal_conductivity_source",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _freeze_db(settings, out_dir: Path) -> dict[str, Any]:
    """Copy live DB to a dated freeze for export reproducibility."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    dest = settings.processed_dir / f"materials.duckdb.pretrain-export-{stamp}"
    if not dest.exists():
        print(f"Freezing live DB → {dest} ...")
        shutil.copy2(settings.duckdb_path, dest)
    meta = {
        "snapshot_path": str(dest.relative_to(settings.project_root)),
        "source_path": str(settings.duckdb_path.relative_to(settings.project_root)),
        "sha256": _sha256(dest),
        "size_bytes": dest.stat().st_size,
        "frozen_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (out_dir / "db_freeze.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def _load_si(conn) -> pd.DataFrame:
    cols = conn.execute("PRAGMA table_info('search_index')").fetchall()
    available = {r[1] for r in cols}
    select = [c for c in FEATURE_COLS if c in available]
    # Always need labels/gates used by targets.
    extras = {
        "energy_above_hull",
        "formation_energy_per_atom",
        "band_gap",
        "bulk_modulus_vrh",
        "shear_modulus_vrh",
        "thermal_conductivity",
        "debye_temperature",
        "heat_capacity_cv",
        "capacity_grav",
        "capacity_vol",
        "average_voltage",
        "energy_grav",
        "energy_vol",
        "working_ion",
        "is_metal",
        "is_stable",
    }
    select = sorted(set(select) | (extras & available))
    return conn.execute(
        f"SELECT {', '.join(select)} FROM search_index"
    ).fetchdf()


def _build_entity_groups(conn) -> dict[str, str]:
    """Map material_id → leak-safe group key (fingerprint family or self)."""
    rows = conn.execute(
        """
        SELECT material_id, composition, formula_pretty, reduced_formula,
               space_group_number, space_group_symbol, nsites
        FROM source_records
        """
    ).fetchall()
    # fingerprint → set of material_ids
    fp_to_mats: dict[str, set[str]] = defaultdict(set)
    for (
        mid,
        composition,
        formula,
        reduced,
        sg_n,
        sg_s,
        nsites,
    ) in rows:
        fp = structure_fingerprint(
            composition=composition,
            formula_pretty=formula,
            reduced_formula=reduced,
            space_group_number=sg_n,
            space_group_symbol=sg_s,
            nsites=nsites,
        )
        if fp is None:
            fp_to_mats[f"solo:{mid}"].add(str(mid))
        else:
            fp_to_mats[fp].add(str(mid))

    mid_to_group: dict[str, str] = {}
    for fp, mats in fp_to_mats.items():
        # Canonical group id = sorted join of member material_ids (stable).
        group_id = "grp:" + hashlib.sha1(
            ("|".join(sorted(mats))).encode()
        ).hexdigest()[:16]
        for mid in mats:
            # If a material appears in multiple FPs (multi-source variants),
            # keep the first assigned group; rare for primary SI rows.
            mid_to_group.setdefault(mid, group_id)
    return mid_to_group


def _strat_bin(row: pd.Series) -> str:
    ne = int(row["nelements"]) if pd.notna(row.get("nelements")) else 0
    ne_bin = "1" if ne <= 1 else "2" if ne == 2 else "3" if ne == 3 else "4+"
    src = str(row.get("source") or "unknown")
    # Coarse chemsys family: first element of chemsys or formula prefix.
    chemsys = str(row.get("chemsys") or "")
    family = chemsys.split("-")[0] if chemsys else "X"
    if len(family) > 2:
        family = family[:2]
    return f"{ne_bin}|{src}|{family}"


def assign_splits(
    si: pd.DataFrame,
    mid_to_group: dict[str, str],
    *,
    seed: int = SPLIT_SEED,
) -> pd.DataFrame:
    """Assign train/val/test per leak-safe group with stratification."""
    rng = np.random.default_rng(seed)
    work = si[["material_id", "nelements", "source", "chemsys"]].copy()
    work["group_id"] = work["material_id"].map(
        lambda m: mid_to_group.get(str(m), f"solo:{m}")
    )
    work["strat"] = work.apply(_strat_bin, axis=1)

    # One strat label per group (mode of members).
    group_strat: dict[str, str] = {}
    for gid, frame in work.groupby("group_id"):
        modes = frame["strat"].mode()
        group_strat[str(gid)] = str(modes.iloc[0]) if len(modes) else "other"

    by_strat: dict[str, list[str]] = defaultdict(list)
    for gid, strat in group_strat.items():
        by_strat[strat].append(gid)

    group_split: dict[str, str] = {}
    for strat, gids in by_strat.items():
        gids = list(gids)
        rng.shuffle(gids)
        n = len(gids)
        n_train = int(n * TRAIN_FRAC)
        n_val = int(n * VAL_FRAC)
        for i, gid in enumerate(gids):
            if i < n_train:
                group_split[gid] = "train"
            elif i < n_train + n_val:
                group_split[gid] = "val"
            else:
                group_split[gid] = "test"

    work["split"] = work["group_id"].map(group_split)
    return work[["material_id", "group_id", "split", "strat"]]


def _holdout_masks(si: pd.DataFrame) -> dict[str, pd.Series]:
    """Specialty evaluation hold-outs (not removed from train by default)."""
    return {
        "holdout_cod_geometry": (si["source"] == "cod")
        & si["structure_path"].notna(),
        "holdout_agl_thermal": si.get(
            "train_ok_thermal_agl", pd.Series(False, index=si.index)
        ).fillna(False)
        == True,  # noqa: E712
        "holdout_battery_electrode": si.get(
            "train_ok_battery", pd.Series(False, index=si.index)
        ).fillna(False)
        == True,  # noqa: E712
        "holdout_elastic_mechanical": si.get(
            "train_ok_elastic", pd.Series(False, index=si.index)
        ).fillna(False)
        == True,  # noqa: E712
    }


def export_targets(
    si: pd.DataFrame,
    splits: pd.DataFrame,
    holdouts: pd.DataFrame,
    out_dir: Path,
) -> list[dict[str, Any]]:
    merged = si.merge(splits, on="material_id", how="left").merge(
        holdouts, on="material_id", how="left"
    )
    print("Adding composition feature vectors …")
    merged = add_composition_features(merged, formula_col="formula_pretty")
    feat_cols = list(FRAC_COLS) + list(COMP_STAT_COLS)

    target_dir = out_dir / "targets"
    target_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []

    for spec in TARGET_SPECS:
        label = spec["label_col"]
        gate = spec.get("gate")
        conflict_col = spec.get("conflict_col")
        n_before_conflict = 0
        mask = merged[label].notna()
        if gate and gate in merged.columns:
            mask &= merged[gate].fillna(False) == True  # noqa: E712
        extra = spec.get("extra_where")
        if extra == "density IS NOT NULL AND (deprecated IS NOT TRUE)":
            mask &= merged["density"].notna()
            if "deprecated" in merged.columns:
                mask &= merged["deprecated"].fillna(False) != True  # noqa: E712
        if extra == "thermal_conductivity_method = 'aflow-agl'":
            mask &= merged.get(
                "thermal_conductivity_method", pd.Series(index=merged.index)
            ).eq("aflow-agl")

        n_before_conflict = int(mask.sum())
        exclude_conflicts = bool(spec.get("exclude_conflicts")) and (
            EXCLUDE_LABEL_CONFLICTS_V1
            and spec["name"] in CONFLICT_EXCLUDE_TARGETS
        )
        n_excluded_conflict = 0
        if exclude_conflicts and conflict_col and conflict_col in merged.columns:
            conflict_mask = merged[conflict_col].fillna(False) == True  # noqa: E712
            n_excluded_conflict = int((mask & conflict_mask).sum())
            mask &= ~conflict_mask

        frame = merged.loc[mask].copy()
        frame = frame.rename(columns={label: "label"})
        keep = [
            c
            for c in [
                *FEATURE_COLS,
                *feat_cols,
                "label",
                "group_id",
                "split",
                "strat",
                *holdouts.columns.drop("material_id"),
            ]
            if c in frame.columns
        ]
        out = frame[keep].copy()
        if "label" not in out.columns:
            out["label"] = frame["label"]

        path = target_dir / f"{spec['name']}.parquet"
        out.to_parquet(path, index=False)

        split_counts = out["split"].value_counts(dropna=False).to_dict()
        summaries.append(
            {
                "name": spec["name"],
                "tier": spec["tier"],
                "label_col": label,
                "gate": gate,
                "n_rows": int(len(out)),
                "n_before_conflict_filter": n_before_conflict,
                "n_excluded_conflict": n_excluded_conflict,
                "exclude_conflicts": exclude_conflicts,
                "split_counts": {str(k): int(v) for k, v in split_counts.items()},
                "path": f"data/processed/training_exports/targets/{spec['name']}.parquet",
                "conflict_col": conflict_col,
                "n_composition_features": len(feat_cols),
                "n_label_conflict_true": int(
                    out["label_conflict"].fillna(False).sum()
                )
                if "label_conflict" in out.columns
                else 0,
            }
        )
        print(
            f"  {spec['name']}: {len(out):,} rows "
            f"(excluded conflicts={n_excluded_conflict:,}) → {path.name}"
        )
    return summaries


def write_holdout_docs(holdouts: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    hdir = out_dir / "holdouts"
    hdir.mkdir(parents=True, exist_ok=True)
    docs: dict[str, Any] = {"description": {}, "counts": {}}
    descriptions = {
        "holdout_cod_geometry": (
            "COD-primary materials with structure_path. Use for experimental "
            "geometry / density evaluation; not a DFT label hold-out."
        ),
        "holdout_agl_thermal": (
            "AFLOW AGL thermal rows (train_ok_thermal_agl). Prefer as the only "
            "honest thermal κ / Debye / C_v evaluation set."
        ),
        "holdout_battery_electrode": (
            "MP insertion-electrode hosts (train_ok_battery)."
        ),
        "holdout_elastic_mechanical": (
            "DFT/AEL elastic subset (train_ok_elastic) for B/G evaluation."
        ),
    }
    for col, desc in descriptions.items():
        ids = holdouts.loc[holdouts[col].fillna(False), "material_id"].tolist()
        path = hdir / f"{col}.json"
        path.write_text(
            json.dumps(
                {
                    "name": col,
                    "description": desc,
                    "n": len(ids),
                    "material_ids": ids,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        docs["description"][col] = desc
        docs["counts"][col] = len(ids)
    (hdir / "README.md").write_text(
        "\n".join(
            [
                "# Specialty hold-outs (Step 4.3)",
                "",
                "These sets are **evaluation slices**. Default stratified splits still "
                "include these materials in train/val/test; for honest niche reporting, "
                "evaluate metrics on the hold-out material_ids (or restrict test metrics "
                "to `holdout_* == true`).",
                "",
                *[f"- **{k}** ({docs['counts'][k]:,}): {v}" for k, v in descriptions.items()],
                "",
            ]
        ),
        encoding="utf-8",
    )
    return docs


def write_dataset_card(
    *,
    out_dir: Path,
    freeze: dict[str, Any],
    target_summaries: list[dict[str, Any]],
    split_counts: dict[str, int],
    holdout_docs: dict[str, Any],
    si_n: int,
    deprecated_n: int,
) -> Path:
    card = out_dir / "DATASET_CARD.md"
    lines = [
        "# Materials Predictor — Training Dataset Card",
        "",
        f"**Export version:** {EXPORT_VERSION}  ",
        f"**Generated (UTC):** {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}  ",
        f"**Exclusion rules:** {EXCLUSION_RULES_VERSION}  ",
        "",
        "## Catalog freeze",
        "",
        f"- Live source DB: `{freeze['source_path']}`",
        f"- Export freeze copy: `{freeze['snapshot_path']}`",
        f"- SHA-256: `{freeze['sha256']}`",
        f"- Size: {freeze['size_bytes']:,} bytes",
        f"- Frozen at: {freeze['frozen_at_utc']}",
        f"- Materials in `search_index`: **{si_n:,}**",
        f"- Deprecated excluded from supervised gates: **{deprecated_n:,}**",
        "",
        "## Quality tiers (see `docs/label_policy.md`)",
        "",
        "- **Ready:** E_hull, E_f, band gap, density",
        "- **Niche:** elastic B/G, AGL thermal, battery electrodes",
        "- **Proxy / derived:** never as DFT ground truth (element-mix κ, nuclear scores, …)",
        "",
        "## Known biases",
        "",
        "- Band gaps are predominantly GGA/PBE (underestimate experiment).",
        "- Cross-source hull / E_f / gap / density disagreements are flagged "
        "(`conflict_*`); Ready targets **exclude** conflict rows "
        f"(`EXCLUDE_LABEL_CONFLICTS_V1={EXCLUDE_LABEL_CONFLICTS_V1}`).",
        "- Density mixes DFT cells and COD experimental geometries.",
        "- Thermal κ on SI is AGL-only; proxy κ is in `thermal_conductivity_proxy`.",
        "- Structure models must use `structure_path IS NOT NULL` (OQMD hole scoped).",
        "",
        "## Composition features",
        "",
        f"- `{len(FRAC_COLS)}` elemental fraction columns (`frac_H` … `frac_Og`, Z=1–118).",
        f"- Stats: {', '.join(f'`{c}`' for c in COMP_STAT_COLS)}.",
        "- Derived from `formula_pretty` via pymatgen Composition (no matminer).",
        "",
        "## Split recipe (Step 4.2)",
        "",
        f"- Seed: `{SPLIT_SEED}`",
        f"- Fractions: train {TRAIN_FRAC:.0%} / val {VAL_FRAC:.0%} / test {1 - TRAIN_FRAC - VAL_FRAC:.0%}",
        "- Leak safety: materials sharing a structure fingerprint "
        "(composition + SG + nsites) share one `group_id` and one split.",
        "- Stratification bins: `nelements` (1/2/3/4+) × primary `source` × chemsys family.",
        "",
        "### Global split counts (all materials)",
        "",
        "| Split | N |",
        "|-------|--:|",
        *[f"| {k} | {v:,} |" for k, v in sorted(split_counts.items())],
        "",
        "## Per-target exports (Step 4.1)",
        "",
        "| Target | Tier | Rows | Excl. conflict | Train | Val | Test |",
        "|--------|------|-----:|---------------:|------:|----:|-----:|",
    ]
    for t in target_summaries:
        sc = t["split_counts"]
        lines.append(
            f"| `{t['name']}` | {t['tier']} | {t['n_rows']:,} | "
            f"{t.get('n_excluded_conflict', 0):,} | "
            f"{sc.get('train', 0):,} | {sc.get('val', 0):,} | {sc.get('test', 0):,} |"
        )
    lines += [
        "",
        "## Specialty hold-outs (Step 4.3)",
        "",
        "| Hold-out | N |",
        "|----------|--:|",
        *[
            f"| `{k}` | {v:,} |"
            for k, v in holdout_docs["counts"].items()
        ],
        "",
        "Details: `holdouts/README.md`.",
        "",
        "## Files",
        "",
        "```text",
        "training_exports/",
        "  DATASET_CARD.md",
        "  export_manifest.json",
        "  db_freeze.json",
        "  splits.parquet",
        "  holdouts/",
        "  targets/*.parquet",
        "```",
        "",
        "## Training contract",
        "",
        "1. Filter with the target’s `gate` / export table (already applied).",
        "2. Ready thermo/gap/density exports already exclude `conflict_* == true` "
        f"(targets: {', '.join(sorted(CONFLICT_EXCLUDE_TARGETS))}).",
        "3. Do not train on Proxy columns or `thermal_conductivity_proxy` as DFT κ.",
        "4. Pin experiments to the freeze SHA-256 above.",
        "5. Use `frac_*` + composition-stat columns as tabular features when needed.",
        "",
    ]
    card.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return card


def main() -> int:
    clear_settings_cache()
    settings = get_settings()
    out_dir = settings.processed_dir / "training_exports"
    out_dir.mkdir(parents=True, exist_ok=True)

    freeze = _freeze_db(settings, out_dir)

    db = MaterialsDatabase(settings.duckdb_path, read_only=True)
    with db.connect() as conn:
        print("Loading search_index …")
        si = _load_si(conn)
        print(f"  rows={len(si):,}")
        print("Building fingerprint leak-groups …")
        mid_to_group = _build_entity_groups(conn)
        deprecated_n = int(
            conn.execute(
                "SELECT COUNT(*) FROM search_index WHERE deprecated IS TRUE"
            ).fetchone()[0]
        )

    print("Assigning stratified splits …")
    splits = assign_splits(si, mid_to_group, seed=SPLIT_SEED)
    split_path = out_dir / "splits.parquet"
    splits.to_parquet(split_path, index=False)
    split_counts = {
        str(k): int(v) for k, v in splits["split"].value_counts().to_dict().items()
    }
    print(f"  splits → {split_counts}")

    print("Building hold-outs …")
    masks = _holdout_masks(si)
    holdouts = si[["material_id"]].copy()
    for name, series in masks.items():
        holdouts[name] = series.astype(bool).values
    holdouts_path = out_dir / "holdouts" / "holdout_flags.parquet"
    holdouts_path.parent.mkdir(parents=True, exist_ok=True)
    holdouts.to_parquet(holdouts_path, index=False)
    holdout_docs = write_holdout_docs(holdouts, out_dir)

    print("Exporting per-target tables …")
    target_summaries = export_targets(si, splits, holdouts, out_dir)

    card = write_dataset_card(
        out_dir=out_dir,
        freeze=freeze,
        target_summaries=target_summaries,
        split_counts=split_counts,
        holdout_docs=holdout_docs,
        si_n=len(si),
        deprecated_n=deprecated_n,
    )

    manifest = {
        "export_version": EXPORT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "exclusion_rules": export_rule_metadata(),
        "composition_features": {
            "n_frac_cols": len(FRAC_COLS),
            "frac_cols_sample": list(FRAC_COLS[:5]) + ["…"] + list(FRAC_COLS[-3:]),
            "stat_cols": list(COMP_STAT_COLS),
        },
        "db_freeze": freeze,
        "split": {
            "seed": SPLIT_SEED,
            "train_frac": TRAIN_FRAC,
            "val_frac": VAL_FRAC,
            "counts": split_counts,
        },
        "targets": target_summaries,
        "holdouts": holdout_docs,
        "paths": {
            "export_dir": str(out_dir.relative_to(settings.project_root)),
            "dataset_card": str(card.relative_to(settings.project_root)),
            "splits": str(split_path.relative_to(settings.project_root)),
        },
    }
    man_path = out_dir / "export_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {card}")
    print(f"Wrote {man_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
