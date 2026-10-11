"""Side-by-side LightGBM vs MACE comparison on shared test IDs (Phase 4.6)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import PROJECT_ROOT
from ml.io_util import write_json
from ml.mace.config import load_mace_config
from ml.mace.pack import load_mace_pack_frame, mace_pack_dir
from ml.mace.predict import load_mace_run
from ml.metrics import correlation_metrics, regression_metrics
from ml.registry import load_registry


@dataclass(frozen=True)
class CompareResult:
    """Paths and summary metrics for one comparison report."""

    target: str
    out_dir: Path
    report_md: Path
    report_json: Path
    n_overlap: int
    lightgbm_mae: float | None
    mace_mae: float | None
    lightgbm_spearman: float | None = None
    mace_spearman: float | None = None
    promote_metric: str = "spearman"
    promote_winner: str | None = None


def _models_root(config: dict[str, Any], project_root: Path) -> Path:
    output = config.get("output") or {}
    return project_root / str(output.get("models_dir") or "data/models")


def _latest_lightgbm_predictions(
    target: str,
    *,
    project_root: Path,
    models_dir: Path,
) -> pd.DataFrame:
    registry = load_registry(models_dir)
    runs = list((registry.get("runs") or {}).get(target) or [])
    record = None
    for item in reversed(runs):
        family = str(item.get("model_family") or "lightgbm")
        if family == "lightgbm":
            record = item
            break
    if record is None:
        raise FileNotFoundError(f"No LightGBM run registered for {target!r}")
    run_dir = Path(record["run_dir"])
    if not run_dir.is_absolute():
        run_dir = project_root / run_dir
    pred_path = run_dir / "predictions_test.parquet"
    if not pred_path.is_file():
        raise FileNotFoundError(
            f"LightGBM test predictions missing: {pred_path}. "
            "Re-train with predictions enabled."
        )
    frame = pd.read_parquet(pred_path)
    if "material_id" not in frame.columns or "y_pred" not in frame.columns:
        raise ValueError(f"Unexpected LightGBM prediction columns in {pred_path}")
    out = frame.loc[:, ["material_id", "y_pred"]].copy()
    if "y_true" in frame.columns:
        out["y_true_lgbm_file"] = frame["y_true"]
    out = out.rename(columns={"y_pred": "y_pred_lightgbm"})
    out["lightgbm_run_id"] = record.get("run_id")
    return out


def _latest_mace_predictions(
    target: str,
    *,
    model_family: str,
    project_root: Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    loaded = load_mace_run(
        target, model_family=model_family, config=config, project_root=project_root
    )
    pred_path = loaded.run_dir / "predictions_test.parquet"
    if not pred_path.is_file():
        raise FileNotFoundError(
            f"MACE test predictions missing: {pred_path}. "
            "Run eval/train with predictions enabled."
        )
    frame = pd.read_parquet(pred_path)
    # Fine-tune / pretrained eval writes y_pred_formation.
    pred_col = None
    for candidate in ("y_pred_formation", "y_pred", "formation_energy_per_atom"):
        if candidate in frame.columns:
            pred_col = candidate
            break
    if pred_col is None or "material_id" not in frame.columns:
        raise ValueError(f"Unexpected MACE prediction columns in {pred_path}")
    out = frame.loc[:, ["material_id", pred_col]].copy()
    out = out.rename(columns={pred_col: "y_pred_mace"})
    if "y_true" in frame.columns:
        out["y_true_mace_file"] = frame["y_true"]
    out["mace_run_id"] = loaded.run_id
    out["mace_model_family"] = loaded.model_family
    return out


def _promote_winner(
    metrics: dict[str, Any],
    *,
    promote_metric: str,
) -> tuple[str, str]:
    """Return ``(promote_winner, mae_winner)`` under the named promote metric."""
    mae_winner = (
        "mace"
        if metrics["mace"]["mae"] < metrics["lightgbm"]["mae"]
        else "lightgbm"
        if metrics["lightgbm"]["mae"] < metrics["mace"]["mae"]
        else "tie"
    )
    if promote_metric == "mae":
        return mae_winner, mae_winner
    if promote_metric != "spearman":
        raise ValueError(
            f"Unsupported promote_metric {promote_metric!r}; expected spearman|mae"
        )
    s_m = metrics["mace"].get("spearman_r")
    s_l = metrics["lightgbm"].get("spearman_r")
    if s_m is None or s_l is None or (isinstance(s_m, float) and np.isnan(s_m)) or (
        isinstance(s_l, float) and np.isnan(s_l)
    ):
        return "tie", mae_winner
    if float(s_m) > float(s_l):
        return "mace", mae_winner
    if float(s_l) > float(s_m):
        return "lightgbm", mae_winner
    return "tie", mae_winner


def compare_lightgbm_mace(
    target: str = "formation_energy_per_atom",
    *,
    mace_family: str = "mace_pretrained",
    split: str = "test",
    limit: int | None = None,
    promote_metric: str | None = None,
    config: dict[str, Any] | None = None,
    project_root: Path | None = None,
    out_dir: Path | None = None,
) -> CompareResult:
    """Compare LightGBM vs MACE on the structure-gated pack intersection.

    Default promote gate is **Spearman** (ranking for score→relax→hull), with
    MAE still reported. Override via ``promote_metric`` or
    ``configs/mace.yaml`` → ``promote.metric``.
    """
    if split != "test":
        raise ValueError("compare_lightgbm_mace currently supports split='test' only")

    cfg = config if config is not None else load_mace_config()
    root = project_root or PROJECT_ROOT
    models_dir = _models_root(cfg, root)
    resolved_promote = str(
        promote_metric
        or (cfg.get("promote") or {}).get("metric")
        or "spearman"
    )

    pack = load_mace_pack_frame(target, train_ok_only=True, config=cfg)
    pack_split = pack.loc[pack["split"].astype(str) == split, ["material_id", "label"]].copy()
    pack_split = pack_split.rename(columns={"label": "y_true"})

    lgbm = _latest_lightgbm_predictions(target, project_root=root, models_dir=models_dir)
    mace = _latest_mace_predictions(
        target, model_family=mace_family, project_root=root, config=cfg
    )

    merged = (
        pack_split.merge(lgbm, on="material_id", how="inner")
        .merge(mace, on="material_id", how="inner")
    )
    if limit is not None:
        merged = merged.head(int(limit)).copy()
    n = int(len(merged))
    if n < 2:
        raise RuntimeError(
            "Need ≥2 overlapping test IDs with LightGBM + MACE predictions "
            f"on the structure pack (got {n})."
        )

    y_true = merged["y_true"].to_numpy(dtype=np.float64)
    y_lgbm = merged["y_pred_lightgbm"].to_numpy(dtype=np.float64)
    y_mace = merged["y_pred_mace"].to_numpy(dtype=np.float64)

    metrics = {
        "lightgbm": {
            **regression_metrics(y_true, y_lgbm),
            **correlation_metrics(y_true, y_lgbm),
        },
        "mace": {
            **regression_metrics(y_true, y_mace),
            **correlation_metrics(y_true, y_mace),
        },
    }
    promote_winner, mae_winner = _promote_winner(
        metrics, promote_metric=resolved_promote
    )

    dest = out_dir or (
        mace_pack_dir(cfg) / "comparisons" / target / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    dest.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(dest / "overlap_predictions.parquet", index=False)

    payload = {
        "target": target,
        "split": split,
        "created_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_overlap": n,
        "lightgbm_run_id": str(merged["lightgbm_run_id"].iloc[0]),
        "mace_run_id": str(merged["mace_run_id"].iloc[0]),
        "mace_model_family": str(merged["mace_model_family"].iloc[0]),
        "metrics": metrics,
        "promote_metric": resolved_promote,
        "promote_winner": promote_winner,
        "mae_winner": mae_winner,
        "note": (
            "Overlap = structure-gated MACE pack ∩ LightGBM test predictions ∩ "
            "MACE test predictions (same material_id). "
            "Promote gate defaults to Spearman (ranking-first for score→hull)."
        ),
    }
    report_json = dest / "comparison.json"
    write_json(report_json, payload)

    lines = [
        f"# LightGBM vs MACE — `{target}` ({split})",
        "",
        f"- Created (UTC): `{payload['created_at_utc']}`",
        f"- Overlap N: **{n}**",
        f"- LightGBM run: `{payload['lightgbm_run_id']}`",
        f"- MACE run: `{payload['mace_run_id']}` ({payload['mace_model_family']})",
        f"- Promote metric: **{resolved_promote}** → winner **{promote_winner}**",
        f"- MAE winner: **{mae_winner}**",
        "",
        "| Family | MAE | RMSE | R² | Pearson | Spearman |",
        "|--------|----:|-----:|---:|--------:|---------:|",
        (
            f"| lightgbm | {metrics['lightgbm']['mae']:.6g} | "
            f"{metrics['lightgbm']['rmse']:.6g} | {metrics['lightgbm']['r2']:.6g} | "
            f"{metrics['lightgbm']['pearson_r']:.6g} | {metrics['lightgbm']['spearman_r']:.6g} |"
        ),
        (
            f"| mace | {metrics['mace']['mae']:.6g} | "
            f"{metrics['mace']['rmse']:.6g} | {metrics['mace']['r2']:.6g} | "
            f"{metrics['mace']['pearson_r']:.6g} | {metrics['mace']['spearman_r']:.6g} |"
        ),
        "",
        f"Predictions: `{dest / 'overlap_predictions.parquet'}`",
        "",
    ]
    report_md = dest / "COMPARISON.md"
    report_md.write_text("\n".join(lines), encoding="utf-8")

    return CompareResult(
        target=target,
        out_dir=dest,
        report_md=report_md,
        report_json=report_json,
        n_overlap=n,
        lightgbm_mae=float(metrics["lightgbm"]["mae"]),
        mace_mae=float(metrics["mace"]["mae"]),
        lightgbm_spearman=float(metrics["lightgbm"]["spearman_r"]),
        mace_spearman=float(metrics["mace"]["spearman_r"]),
        promote_metric=resolved_promote,
        promote_winner=promote_winner,
    )
