#!/usr/bin/env python3
"""CLI: predict properties from formulas using trained LightGBM runs.

Examples
--------
python scripts/predict_models.py --target band_gap --formula SiO2 --formula GaN
python scripts/predict_models.py --target density --formula Au --run-dir data/models/density/<run>
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
from ml.predict import load_model, predict_with_model  # noqa: E402
from ml.targets import load_models_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict properties with the latest trained model for a target."
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Target head, e.g. band_gap or density",
    )
    parser.add_argument(
        "--formula",
        action="append",
        dest="formulas",
        required=True,
        help="Chemical formula (repeatable). Example: --formula SiO2",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional models.yaml override.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional specific run directory (default: latest registered).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clear_settings_cache()
    config = load_models_config(args.config)

    loaded = load_model(
        args.target,
        config,
        project_root=ROOT,
        run_dir=args.run_dir,
    )
    frame = predict_with_model(loaded, args.formulas)

    test_r2 = (loaded.metrics.get("test") or {}).get("r2")
    print(f"Using {loaded.target} run {loaded.run_id} (test R2={test_r2})")
    print(frame.to_string(index=False))
    print(json.dumps(frame.to_dict(orient="records"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
