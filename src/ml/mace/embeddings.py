"""Extract and cache MACE node descriptors for readout heads (Phase 4.8).

Protocol ``mace_desc_meanstd_v1``:
* Foundation ``MACECalculator.get_descriptors(..., invariants_only=True)``
* Per-structure pooling: concatenate mean and std over atoms → fixed vector
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from config.settings import PROJECT_ROOT
from ml.io_util import relpath_under, write_json
from ml.mace.calculator import load_foundation_calculator
from ml.mace.config import foundation_model_id, load_mace_config
from ml.mace.pack import load_mace_pack_frame, mace_pack_dir
from ml.mace.structures import load_structure

PROTOCOL_DESC_MEANSTD_V1 = "mace_desc_meanstd_v1"
EMBEDDING_PROTOCOLS = frozenset({PROTOCOL_DESC_MEANSTD_V1})


@dataclass(frozen=True)
class EmbeddingExtractResult:
    """Cached embedding table for one target / protocol."""

    target: str
    protocol: str
    cache_path: Path
    n_rows: int
    n_features: int
    n_failed_loads: int
    n_failed_infer: int
    feature_cols: tuple[str, ...]


def pool_descriptors(
    per_atom: np.ndarray,
    *,
    protocol: str = PROTOCOL_DESC_MEANSTD_V1,
) -> np.ndarray:
    """Pool ``(n_atoms, n_feat)`` descriptors to a 1-D structure vector."""
    arr = np.asarray(per_atom, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(f"Expected (n_atoms, n_feat) descriptors, got {arr.shape}")
    if protocol != PROTOCOL_DESC_MEANSTD_V1:
        raise ValueError(
            f"Unsupported embedding protocol {protocol!r}; "
            f"expected one of {sorted(EMBEDDING_PROTOCOLS)}"
        )
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    # Constant columns → std 0; keep finite.
    std = np.nan_to_num(std, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate([mean, std], axis=0)


def feature_column_names(n_features: int) -> list[str]:
    """Stable column names for a pooled embedding of length ``n_features``."""
    half = n_features // 2
    cols = [f"mace_mean_{i}" for i in range(half)]
    cols.extend(f"mace_std_{i}" for i in range(n_features - half))
    return cols


def extract_structure_embedding(
    atoms: Any,
    calc: Any,
    *,
    protocol: str = PROTOCOL_DESC_MEANSTD_V1,
    invariants_only: bool = True,
    num_layers: int = -1,
) -> np.ndarray:
    """Return pooled descriptor vector for one ASE ``Atoms``."""
    atoms.calc = calc
    # Ensure a forward pass has run (some calculator paths need energy first).
    _ = float(atoms.get_potential_energy())
    per_atom = calc.get_descriptors(
        atoms, invariants_only=bool(invariants_only), num_layers=int(num_layers)
    )
    return pool_descriptors(np.asarray(per_atom), protocol=protocol)


def default_embeddings_cache_path(
    target: str,
    *,
    protocol: str = PROTOCOL_DESC_MEANSTD_V1,
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
        / f"{protocol}.parquet"
    )


def extract_embeddings(
    target: str,
    *,
    protocol: str = PROTOCOL_DESC_MEANSTD_V1,
    splits: tuple[str, ...] | None = None,
    limit_per_split: int | None = None,
    device: str | None = "cpu",
    model_id: str | None = None,
    checkpoint: Path | str | None = None,
    cache_path: Path | None = None,
    config: dict[str, Any] | None = None,
    project_root: Path | None = None,
    progress_every: int = 200,
    overwrite: bool = False,
) -> EmbeddingExtractResult:
    """Extract pooled MACE descriptors for pack rows; write a cache parquet."""
    if protocol not in EMBEDDING_PROTOCOLS:
        raise ValueError(f"Unsupported embedding protocol {protocol!r}")

    cfg = config if config is not None else load_mace_config()
    root = project_root or PROJECT_ROOT
    emb_cfg = (cfg.get("embeddings") or {})
    invariants_only = bool(emb_cfg.get("invariants_only", True))
    num_layers = int(emb_cfg.get("num_layers", -1))

    pack = load_mace_pack_frame(target, train_ok_only=True, config=cfg)
    if splits is not None:
        pack = pack.loc[pack["split"].astype(str).isin(set(splits))].copy()

    parts: list[pd.DataFrame] = []
    if limit_per_split is not None:
        for _split, group in pack.groupby(pack["split"].astype(str), sort=False):
            parts.append(group.head(int(limit_per_split)))
        pack = pd.concat(parts, ignore_index=True) if parts else pack.iloc[0:0]

    calc, resolved_model, ckpt_path, resolved_device = load_foundation_calculator(
        model_id=model_id,
        device=device,
        config=cfg,
        checkpoint=checkpoint,
    )
    out_path = cache_path or default_embeddings_cache_path(
        target, protocol=protocol, model_id=resolved_model, config=cfg
    )
    if not out_path.is_absolute():
        out_path = root / out_path
    existing: pd.DataFrame | None = None
    have_ids: set[str] = set()
    if out_path.is_file() and not overwrite:
        existing = pd.read_parquet(out_path)
        have_ids = set(existing["material_id"].astype(str).tolist())
        missing = pack.loc[~pack["material_id"].astype(str).isin(have_ids)].copy()
        if len(missing) == 0:
            feat_cols = [c for c in existing.columns if c.startswith("mace_")]
            return EmbeddingExtractResult(
                target=target,
                protocol=protocol,
                cache_path=out_path,
                n_rows=int(len(existing)),
                n_features=len(feat_cols),
                n_failed_loads=0,
                n_failed_infer=0,
                feature_cols=tuple(feat_cols),
            )
        # Phase 4.9: incremental resume — only extract missing material_ids.
        pack = missing
        print(
            f"[{target}] embedding resume: {len(have_ids)} cached, "
            f"{len(pack)} remaining …",
            flush=True,
        )

    rows: list[dict[str, Any]] = []
    n_failed_loads = 0
    n_failed_infer = 0
    feature_cols: list[str] = []
    if existing is not None:
        feature_cols = [c for c in existing.columns if c.startswith("mace_")]

    for i, rec in enumerate(pack.itertuples(index=False), start=1):
        structure_path = getattr(rec, "structure_path", None)
        loaded = load_structure(structure_path, attach_atoms=True)
        if not loaded.ok or loaded.atoms is None:
            n_failed_loads += 1
            continue
        try:
            vec = extract_structure_embedding(
                loaded.atoms,
                calc,
                protocol=protocol,
                invariants_only=invariants_only,
                num_layers=num_layers,
            )
        except Exception:  # noqa: BLE001
            n_failed_infer += 1
            continue
        if not feature_cols:
            feature_cols = feature_column_names(int(vec.shape[0]))
        row = {
            "material_id": str(getattr(rec, "material_id", "")),
            "split": str(getattr(rec, "split", "")),
            "structure_path": loaded.path,
            "label": float(rec.label),
            "n_atoms": loaded.n_atoms,
            "formula": loaded.formula,
        }
        for name, value in zip(feature_cols, vec, strict=True):
            row[name] = float(value)
        rows.append(row)
        if progress_every and i % int(progress_every) == 0:
            print(f"[{target}] embeddings {i}/{len(pack)} …", flush=True)

    new_frame = pd.DataFrame(rows)
    if existing is not None and len(existing):
        frame = pd.concat([existing, new_frame], ignore_index=True)
        frame = frame.drop_duplicates(subset=["material_id"], keep="last")
    else:
        frame = new_frame
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)
    meta = {
        "target": target,
        "protocol": protocol,
        "model_id": resolved_model,
        "checkpoint": relpath_under(ckpt_path, root),
        "device": resolved_device,
        "invariants_only": invariants_only,
        "num_layers": num_layers,
        "n_rows": int(len(frame)),
        "n_features": len(feature_cols),
        "n_failed_loads": n_failed_loads,
        "n_failed_infer": n_failed_infer,
        "created_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "feature_cols": feature_cols,
        "limit_per_split": limit_per_split,
        "splits": list(splits) if splits else None,
    }
    write_json(out_path.with_suffix(".json"), meta)
    (out_path.parent / "config_snapshot.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=True),
        encoding="utf-8",
    )
    return EmbeddingExtractResult(
        target=target,
        protocol=protocol,
        cache_path=out_path,
        n_rows=int(len(frame)),
        n_features=len(feature_cols),
        n_failed_loads=n_failed_loads,
        n_failed_infer=n_failed_infer,
        feature_cols=tuple(feature_cols),
    )


def load_embeddings_frame(
    target: str,
    *,
    protocol: str = PROTOCOL_DESC_MEANSTD_V1,
    cache_path: Path | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Load a previously extracted embedding cache."""
    cfg = config if config is not None else load_mace_config()
    path = cache_path or default_embeddings_cache_path(
        target, protocol=protocol, config=cfg
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Embedding cache not found: {path}. "
            "Run scripts/extract_mace_embeddings.py first."
        )
    return pd.read_parquet(path)
