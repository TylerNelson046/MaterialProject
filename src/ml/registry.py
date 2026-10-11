"""Persist training run metadata under ``data/models/``.

``registry.json`` stores the full history of runs per target. ``LEADERBOARD.md``
summarizes the latest run for quick comparison (local only; gitignored).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ml.io_util import read_json, write_json

REGISTRY_NAME = "registry.json"
LEADERBOARD_NAME = "LEADERBOARD.md"


def load_registry(models_dir: Path) -> dict[str, Any]:
    """Return registry document ``{version, runs: {target: [records...]}}``."""
    raw = read_json(models_dir / REGISTRY_NAME)
    if raw is None:
        return {"version": 1, "runs": {}}
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid registry at {models_dir / REGISTRY_NAME}")
    raw.setdefault("runs", {})
    return raw


def register_run(models_dir: Path, record: dict[str, Any]) -> Path:
    """Append a run record and rewrite ``registry.json`` + leaderboard."""
    target = str(record["target"])
    registry = load_registry(models_dir)
    runs = registry.setdefault("runs", {})
    target_runs = list(runs.get(target) or [])
    target_runs.append(record)
    runs[target] = target_runs
    registry_path = models_dir / REGISTRY_NAME
    write_json(registry_path, registry)
    write_leaderboard(models_dir)
    return registry_path


def latest_summary_rows(models_dir: Path) -> list[dict[str, Any]]:
    """One summary row per ``(target, model_family)`` (most recent run each)."""
    registry = load_registry(models_dir)
    latest_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for target, run_list in (registry.get("runs") or {}).items():
        for record in run_list or []:
            family = str(record.get("model_family") or "lightgbm")
            latest_by_key[(str(target), family)] = record

    rows: list[dict[str, Any]] = []
    for (target, family), latest in sorted(latest_by_key.items()):
        metrics = latest.get("metrics") or {}
        test = metrics.get("test") or {}
        counts = latest.get("counts") or {}
        rows.append(
            {
                "target": target,
                "tier": latest.get("tier"),
                "model_family": family,
                "run_id": latest.get("run_id"),
                "test_mae": test.get("mae"),
                "test_rmse": test.get("rmse"),
                "test_r2": test.get("r2"),
                "test_spearman": test.get("spearman_r"),
                "n_train": counts.get("train"),
                "n_test": counts.get("test"),
                "run_dir": latest.get("run_dir"),
            }
        )
    return rows


def write_leaderboard(models_dir: Path) -> Path:
    """Write a compact markdown leaderboard from the latest runs."""
    rows = latest_summary_rows(models_dir)
    lines = [
        "# Model leaderboard",
        "",
        "Latest registered run per target **and** model family (local; not committed).",
        "",
        "| Target | Family | Tier | Test MAE | Test Spearman | Test RMSE | Test R² | N | Run |",
        "|--------|--------|------|---------:|--------------:|----------:|--------:|--:|-----|",
    ]
    for row in rows:
        # Prefer test N so LightGBM (train-heavy) and MACE eval rows compare fairly.
        n = row.get("n_test")
        if n is None:
            n = row.get("n_train")
        n_s = "" if n is None else str(n)
        row_fmt = (
            "| {target} | {family} | {tier} | {mae} | {spearman} | "
            "{rmse} | {r2} | {n} | `{run}` |"
        )
        lines.append(
            row_fmt.format(
                target=row["target"],
                family=row.get("model_family") or "",
                tier=row.get("tier") or "",
                mae=_fmt(row.get("test_mae")),
                spearman=_fmt(row.get("test_spearman")),
                rmse=_fmt(row.get("test_rmse")),
                r2=_fmt(row.get("test_r2")),
                n=n_s,
                run=row.get("run_id") or "",
            )
        )

    # Dual-family snapshot: targets that have both LightGBM and any MACE family.
    by_target: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_target.setdefault(str(row["target"]), {})[str(row["model_family"])] = row
    dual_lines: list[str] = []
    for target in sorted(by_target):
        families = by_target[target]
        if "lightgbm" not in families:
            continue
        mace_family = None
        # Frozen-first: prefer pretrained / embedding heads over fine-tunes.
        for name in ("mace_pretrained", "mace_embedding_head", "mace_finetuned"):
            if name in families:
                mace_family = name
                break
        if mace_family is None:
            continue
        lgbm = families["lightgbm"]
        mace = families[mace_family]
        dual_fmt = (
            "| {target} | {mace_fam} | {lgbm_sp} | {mace_sp} | {d_sp} | "
            "{lgbm_mae} | {mace_mae} | {d_mae} |"
        )
        dual_lines.append(
            dual_fmt.format(
                target=target,
                mace_fam=mace_family,
                lgbm_sp=_fmt(lgbm.get("test_spearman")),
                mace_sp=_fmt(mace.get("test_spearman")),
                d_sp=_fmt_delta_higher_better(
                    lgbm.get("test_spearman"), mace.get("test_spearman")
                ),
                lgbm_mae=_fmt(lgbm.get("test_mae")),
                mace_mae=_fmt(mace.get("test_mae")),
                d_mae=_fmt_delta(lgbm.get("test_mae"), mace.get("test_mae")),
            )
        )
    if dual_lines:
        dual_header = (
            "| Target | MACE family | LGBM Spearman | MACE Spearman | "
            "ΔSpearman | LGBM MAE | MACE MAE | ΔMAE |"
        )
        dual_sep = (
            "|--------|-------------|--------------:|--------------:|"
            "----------:|---------:|---------:|-----:|"
        )
        lines.extend(
            [
                "",
                "## Dual-family (LightGBM vs best MACE)",
                "",
                "Promote gate for structure Ready loop: **Spearman** "
                "(higher is better). MAE is reported; ΔMAE negative means "
                "MACE lower error.",
                "",
                dual_header,
                dual_sep,
                *dual_lines,
                "",
            ]
        )
    else:
        lines.append("")

    path = models_dir / LEADERBOARD_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_delta(lightgbm_mae: Any, mace_mae: Any) -> str:
    """MAE delta: MACE − LightGBM (negative means MACE wins on MAE)."""
    try:
        return f"{float(mace_mae) - float(lightgbm_mae):+.6g}"
    except (TypeError, ValueError):
        return ""


def _fmt_delta_higher_better(lightgbm_val: Any, mace_val: Any) -> str:
    """Correlation delta: MACE − LightGBM (positive means MACE wins)."""
    try:
        return f"{float(mace_val) - float(lightgbm_val):+.6g}"
    except (TypeError, ValueError):
        return ""
