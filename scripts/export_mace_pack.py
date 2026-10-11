#!/usr/bin/env python3
"""Phase 4.3: export structure-gated MACE training pack.

Writes under ``data/processed/mace_exports/`` (local / gitignored)::

    targets/<target>.parquet
    export_manifest.json
    DATASET_CARD.md
    db_freeze.json

Requires a Phase 4.2 audit by default::

    python scripts/audit_mace_structures.py --target formation_energy_per_atom
    python scripts/export_mace_pack.py
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
        help="Training-export target (default: formation_energy_per_atom).",
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=None,
        help="Mark rows with more atoms as mace_train_ok=false (still kept).",
    )
    parser.add_argument(
        "--allow-without-audit",
        action="store_true",
        help="If audit.parquet is missing, load structures inline (slow).",
    )
    parser.add_argument(
        "--reuse-audit-from",
        default=None,
        help=(
            "If this target has no audit, reuse another target's audit by "
            "material_id (e.g. formation_energy_per_atom for Phase 4.7)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override pack directory.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print result summary as JSON.",
    )
    args = parser.parse_args(argv)

    from ml.mace.pack import export_mace_pack

    result = export_mace_pack(
        args.target,
        max_atoms=args.max_atoms,
        require_audit=not args.allow_without_audit,
        reuse_audit_from=args.reuse_audit_from,
        out_dir=args.out_dir,
    )
    payload = {
        "target": result.target,
        "pack_dir": str(result.pack_dir),
        "target_parquet": str(result.target_parquet),
        "n_source_rows": result.n_source_rows,
        "n_pack_rows": result.n_pack_rows,
        "n_train": result.n_train,
        "n_val": result.n_val,
        "n_test": result.n_test,
        "n_excluded_missing_or_bad": result.n_excluded_missing_or_bad,
        "n_oversized": result.n_oversized,
        "max_atoms": result.max_atoms,
        "manifest": str(result.manifest_path),
        "dataset_card": str(result.dataset_card_path),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"MACE pack OK | target={result.target} "
            f"pack_rows={result.n_pack_rows}/{result.n_source_rows} "
            f"train/val/test={result.n_train}/{result.n_val}/{result.n_test} "
            f"excluded={result.n_excluded_missing_or_bad} "
            f"oversized={result.n_oversized}"
        )
        print(f"Wrote {result.dataset_card_path}")
        print(f"Wrote {result.target_parquet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
