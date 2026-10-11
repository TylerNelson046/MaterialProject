#!/usr/bin/env python3
"""Phase 4.2: audit structure_path → ASE loadability for a Ready target.

Writes::

    data/processed/mace_exports/structure_audit/<target>/
      audit.parquet
      summary.json
      SUMMARY.md

Usage::

    python scripts/audit_mace_structures.py
    python scripts/audit_mace_structures.py --target formation_energy_per_atom --limit 1000
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
    parser.add_argument(
        "--target",
        default="formation_energy_per_atom",
        help="Training-export target name (default: formation_energy_per_atom).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Audit only the first N rows (debug / smoke).",
    )
    parser.add_argument(
        "--split",
        default=None,
        choices=["train", "val", "test"],
        help="Optional split filter.",
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=None,
        help="Fail rows with more than N atoms (REASON_TOO_MANY_ATOMS).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override output directory.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Process pool size (default: 4; use 1 to disable).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print summary JSON to stdout.",
    )
    args = parser.parse_args(argv)

    from ml.mace.audit import audit_structures

    result = audit_structures(
        args.target,
        limit=args.limit,
        split=args.split,
        max_atoms=args.max_atoms,
        out_dir=args.out_dir,
        workers=args.workers,
    )
    payload = {
        "target": result.target,
        "out_dir": str(result.out_dir),
        "audit_parquet": str(result.audit_parquet),
        "n_rows": result.n_rows,
        "n_ok": result.n_ok,
        "n_fail": result.n_fail,
        "ok_fraction": (result.n_ok / result.n_rows) if result.n_rows else None,
        "reason_counts": result.reason_counts,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        frac = 100.0 * (payload["ok_fraction"] or 0.0)
        print(
            f"Structure audit OK | target={result.target} "
            f"rows={result.n_rows} load_ok={result.n_ok} ({frac:.2f}%) "
            f"fail={result.n_fail}"
        )
        print(f"Wrote {result.summary_md}")
        top = sorted(result.reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
        for reason, n in top:
            print(f"  {reason}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
