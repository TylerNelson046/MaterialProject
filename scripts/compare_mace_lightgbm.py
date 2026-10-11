#!/usr/bin/env python3
"""Phase 4.6: compare LightGBM vs MACE on overlapping structure-gated test IDs.

Example::

    python scripts/compare_mace_lightgbm.py --target formation_energy_per_atom
    python scripts/compare_mace_lightgbm.py --family mace_pretrained --limit 500
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="formation_energy_per_atom")
    parser.add_argument(
        "--family",
        default="mace_pretrained",
        choices=["mace_pretrained", "mace_embedding_head", "mace_finetuned"],
        help="Frozen-first default: mace_pretrained.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--promote-metric",
        default=None,
        choices=["spearman", "mae"],
        help="Promote gate (default: configs/mace.yaml → promote.metric = spearman).",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from ml.mace.compare import compare_lightgbm_mace

    result = compare_lightgbm_mace(
        args.target,
        mace_family=args.family,
        limit=args.limit,
        promote_metric=args.promote_metric,
        out_dir=args.out_dir,
    )
    payload = {
        "target": result.target,
        "n_overlap": result.n_overlap,
        "lightgbm_mae": result.lightgbm_mae,
        "mace_mae": result.mace_mae,
        "lightgbm_spearman": result.lightgbm_spearman,
        "mace_spearman": result.mace_spearman,
        "promote_metric": result.promote_metric,
        "promote_winner": result.promote_winner,
        "report_md": str(result.report_md),
        "report_json": str(result.report_json),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"Compare OK | target={result.target} n={result.n_overlap} "
            f"promote={result.promote_metric}:{result.promote_winner} "
            f"Spearman lgbm={result.lightgbm_spearman:.4f} mace={result.mace_spearman:.4f} "
            f"MAE lgbm={result.lightgbm_mae:.6g} mace={result.mace_mae:.6g}"
        )
        print(f"Wrote {result.report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
