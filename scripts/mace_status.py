#!/usr/bin/env python3
"""Phase 4.9: print local MACE stack status (read-only).

Example::

    python scripts/mace_status.py
    python scripts/mace_status.py --json
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
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--no-devices",
        action="store_true",
        help="Skip torch device probe (faster; no [mace] import).",
    )
    args = parser.parse_args(argv)

    from ml.mace.status import collect_mace_status

    status = collect_mace_status(include_devices=not args.no_devices)
    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0

    print(f"MACE status @ {status['generated_at_utc']}")
    print(f"  python: {status['python']}")
    print(f"  foundation: {status['foundation_model_id']}")
    print(f"  presets: {', '.join(status.get('runs_presets') or []) or '(defaults)'}")
    devices = status.get("devices")
    if devices is None:
        print("  devices: (skipped)")
    elif "error" in devices:
        print(f"  devices: unavailable ({devices['error']})")
    else:
        print(
            "  devices: "
            f"cuda={devices.get('cuda')} mps={devices.get('mps')} "
            f"cpu={devices.get('cpu')} torch={devices.get('torch_version')}"
        )
    ck = status.get("checkpoint_cache") or {}
    print(f"  checkpoints: {len(ck.get('files') or [])} in {ck.get('dir')}")
    pack = status.get("mace_pack") or {}
    print(f"  pack dir: {pack.get('dir')}")
    for item in pack.get("targets") or []:
        print(f"    - {item['target']}")
    emb = status.get("embedding_caches") or []
    print(f"  embedding caches: {len(emb)}")
    print("  latest MACE runs:")
    for row in status.get("latest_mace_runs") or []:
        promo = row.get("promoted_vs_lightgbm")
        promo_s = (
            " PROMOTE"
            if promo is True
            else " keep-LGBM"
            if promo is False
            else ""
        )
        print(
            f"    - {row['target']} / {row['model_family']}: "
            f"MAE={row.get('test_mae')}{promo_s}"
        )
    if status.get("leaderboard"):
        print(f"  leaderboard: {status['leaderboard']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
