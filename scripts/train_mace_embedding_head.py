#!/usr/bin/env python3
"""Phase 4.8: train LightGBM readout on MACE embeddings; register + compare.

Examples::

    python scripts/train_mace_embedding_head.py --target band_gap \\
        --limit-per-split 200 --device cpu
    python scripts/train_mace_embedding_head.py --target bulk_modulus_vrh \\
        --limit-per-split 150 --device cpu
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

from ml.mace.runtime import configure_mace_runtime  # noqa: E402

configure_mace_runtime()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="band_gap")
    parser.add_argument(
        "--protocol",
        default="mace_desc_meanstd_v1",
        choices=["mace_desc_meanstd_v1"],
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="If cache missing, extract at most N rows per split.",
    )
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Skip LightGBM intersection compare / promotion flag.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from ml.mace.train_embedding_head import train_embedding_head

    result = train_embedding_head(
        args.target,
        protocol=args.protocol,
        cache_path=args.cache_path,
        limit_per_split=args.limit_per_split,
        max_train_rows=args.max_train_rows,
        device=args.device,
        compare=not args.no_compare,
    )
    test = (result.metrics.get("registry_metrics") or {}).get("test") or {}
    payload = {
        "target": result.target,
        "run_id": result.run_id,
        "run_dir": str(result.run_dir),
        "protocol": result.protocol,
        "counts": result.counts,
        "test_mae": test.get("mae"),
        "promoted_vs_lightgbm": result.promoted,
        "compare_report": str(result.compare_report) if result.compare_report else None,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        promo = (
            "PROMOTE"
            if result.promoted is True
            else "keep LightGBM"
            if result.promoted is False
            else "compare skipped/failed"
        )
        print(
            f"Embedding head OK | target={result.target} "
            f"test_MAE={test.get('mae')} ({promo}) run={result.run_id}"
        )
        print(f"Wrote {result.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
