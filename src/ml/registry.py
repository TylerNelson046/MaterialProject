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
    """One summary row per target using the most recently registered run."""
    registry = load_registry(models_dir)
    rows: list[dict[str, Any]] = []
    for target, run_list in sorted((registry.get("runs") or {}).items()):
        if not run_list:
            continue
        latest = run_list[-1]
        metrics = latest.get("metrics") or {}
        test = metrics.get("test") or {}
        rows.append(
            {
                "target": target,
                "tier": latest.get("tier"),
                "run_id": latest.get("run_id"),
                "test_mae": test.get("mae"),
                "test_rmse": test.get("rmse"),
                "test_r2": test.get("r2"),
                "n_train": (latest.get("counts") or {}).get("train"),
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
        "Latest registered LightGBM run per target (local artifacts; not committed).",
        "",
        "| Target | Tier | Test MAE | Test RMSE | Test R² | N train | Run |",
        "|--------|------|---------:|----------:|--------:|--------:|-----|",
    ]
    for row in rows:
        n_train = row.get("n_train")
        n_train_s = "" if n_train is None else str(n_train)
        lines.append(
            "| {target} | {tier} | {mae} | {rmse} | {r2} | {n} | `{run}` |".format(
                target=row["target"],
                tier=row.get("tier") or "",
                mae=_fmt(row.get("test_mae")),
                rmse=_fmt(row.get("test_rmse")),
                r2=_fmt(row.get("test_r2")),
                n=n_train_s,
                run=row.get("run_id") or "",
            )
        )
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
