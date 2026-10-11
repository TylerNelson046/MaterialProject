#!/usr/bin/env python3
"""CLI: train LightGBM property models from ``training_exports``.

Examples
--------
python scripts/train_models.py --all-ready
python scripts/train_models.py --all-niche
python scripts/train_models.py --target band_gap --target density
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config.settings import clear_settings_cache  # noqa: E402
from ml.targets import (  # noqa: E402
    known_targets,
    load_models_config,
    niche_targets,
    ready_targets,
)
from ml.train import default_train_targets, train_many  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LightGBM regressors on the training export package."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--target",
        action="append",
        dest="targets",
        help="Target name (repeatable). Example: --target band_gap",
    )
    group.add_argument(
        "--all-ready",
        action="store_true",
        help="Train all Ready-tier targets (default scope).",
    )
    group.add_argument(
        "--all-niche",
        action="store_true",
        help="Train all Niche-tier targets.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Train every target in configs/models.yaml.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional models.yaml override (default: configs/models.yaml).",
    )
    parser.add_argument(
        "--no-predictions",
        action="store_true",
        help="Skip writing predictions_*.parquet artifacts.",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional subsample of the train split (debug / smoke).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help="Override LightGBM n_estimators (tree cap).",
    )
    return parser.parse_args()


def _resolve_targets(args: argparse.Namespace, config: dict) -> list[str]:
    if args.all:
        return list(known_targets(config))
    if args.all_ready:
        return list(ready_targets(config))
    if args.all_niche:
        return list(niche_targets(config))
    if args.targets:
        return list(args.targets)
    return list(default_train_targets(config))


def main() -> int:
    args = parse_args()
    clear_settings_cache()
    config = load_models_config(args.config)
    names = _resolve_targets(args, config)

    overrides: dict = {}
    if args.n_estimators is not None:
        overrides["n_estimators"] = args.n_estimators

    print(f"Training {len(names)} target(s): {', '.join(names)}")
    results = train_many(
        names,
        config,
        project_root=ROOT,
        write_predictions=not args.no_predictions,
        max_train_rows=args.max_train_rows,
        lgb_overrides=overrides or None,
    )

    summary = []
    for result in results:
        test = result.metrics.get("test") or {}
        row = {
            "target": result.target,
            "tier": result.tier,
            "run_id": result.run_id,
            "test_mae": test.get("mae"),
            "test_rmse": test.get("rmse"),
            "test_r2": test.get("r2"),
            "best_iteration": result.best_iteration,
            "run_dir": str(result.run_dir),
        }
        summary.append(row)
        print(
            f"  {result.target}: test MAE={test.get('mae'):.6g} "
            f"RMSE={test.get('rmse'):.6g} R2={test.get('r2'):.4f} "
            f"-> {result.run_dir}"
        )

    print(json.dumps({"runs": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
