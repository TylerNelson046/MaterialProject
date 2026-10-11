#!/usr/bin/env python3
"""Diagnose frozen MACE formation-energy harness (reference protocol check).

Compares ``elemental_e0_v1`` (isolated-atom, legacy) vs ``elemental_bulk_v1``
(standard-state μ) on a slice of the structure pack. Prefer MP-source rows when
available.

Example::

    python scripts/diagnose_mace_harness.py --limit 200 --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.mace.runtime import configure_mace_runtime  # noqa: E402

configure_mace_runtime()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--source", default="mp", help="Prefer this source (default: mp).")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from ml.mace.calculator import load_foundation_calculator
    from ml.mace.elemental_refs import ensure_elemental_bulk_refs
    from ml.mace.pack import load_mace_pack_frame
    from ml.mace.reference import (
        PROTOCOL_ELEMENTAL_E0_V1,
        formation_energy_per_atom,
        load_elemental_e0s,
    )
    from ml.mace.structures import load_structure
    from ml.metrics import correlation_metrics, regression_metrics

    pack = load_mace_pack_frame("formation_energy_per_atom", train_ok_only=True)
    prefer = pack.loc[pack["source"].astype(str) == args.source].copy()
    if len(prefer) < max(20, args.limit // 4):
        prefer = pack.copy()
        note = f"few {args.source} rows; using mixed sources"
    else:
        note = f"filtered source={args.source}"
    # Prefer test split for promote-shaped diagnostics.
    test = prefer.loc[prefer["split"].astype(str) == "test"]
    frame = test if len(test) >= args.limit else prefer
    frame = frame.head(int(args.limit))

    calc, model_id, _, device = load_foundation_calculator(device=args.device)
    print("Building elemental_bulk_v1 μ table (cached after first run) …", flush=True)
    bulk = ensure_elemental_bulk_refs(
        model_id, calc=calc, force=False, progress=lambda m: print(f"  {m}", flush=True)
    )
    isol = load_elemental_e0s(model_id)

    rows_isol: list[float] = []
    rows_bulk: list[float] = []
    y_true: list[float] = []
    n_fail = 0
    for i, rec in enumerate(frame.itertuples(index=False), start=1):
        loaded = load_structure(rec.structure_path, attach_atoms=True)
        if not loaded.ok or loaded.atoms is None:
            n_fail += 1
            continue
        atoms = loaded.atoms
        try:
            atoms.calc = calc
            e_tot = float(atoms.get_potential_energy())
            zs = [int(z) for z in atoms.get_atomic_numbers()]
            # Skip if bulk table missing any Z
            if any(z not in bulk for z in zs):
                n_fail += 1
                continue
            rows_isol.append(formation_energy_per_atom(e_tot, zs, isol))
            rows_bulk.append(formation_energy_per_atom(e_tot, zs, bulk))
            y_true.append(float(rec.label))
        except Exception:  # noqa: BLE001
            n_fail += 1
            continue
        if i % 50 == 0:
            print(f"  diagnosed {i}/{len(frame)} …", flush=True)

    yt = np.asarray(y_true, dtype=np.float64)
    yi = np.asarray(rows_isol, dtype=np.float64)
    yb = np.asarray(rows_bulk, dtype=np.float64)
    payload = {
        "n": int(len(yt)),
        "n_failed": n_fail,
        "note": note,
        "model_id": model_id,
        "device": device,
        "n_bulk_elements": len(bulk),
        "legacy_elemental_e0_v1": {
            **regression_metrics(yt, yi),
            **correlation_metrics(yt, yi),
            "mean_pred": float(yi.mean()),
            "mean_true": float(yt.mean()),
            "mean_bias": float((yi - yt).mean()),
        },
        "elemental_bulk_v1": {
            **regression_metrics(yt, yb),
            **correlation_metrics(yt, yb),
            "mean_pred": float(yb.mean()),
            "mean_true": float(yt.mean()),
            "mean_bias": float((yb - yt).mean()),
        },
        "verdict": (
            "harness_ok_bulk"
            if regression_metrics(yt, yb)["mae"] < 0.25
            else "harness_improved_but_check_labels"
            if regression_metrics(yt, yb)["mae"]
            < regression_metrics(yt, yi)["mae"] * 0.5
            else "still_broken"
        ),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"\nDiagnose n={payload['n']} ({note})")
        print(
            f"  LEGACY {PROTOCOL_ELEMENTAL_E0_V1}: "
            f"MAE={payload['legacy_elemental_e0_v1']['mae']:.4f} "
            f"bias={payload['legacy_elemental_e0_v1']['mean_bias']:+.4f}"
        )
        print(
            f"  FIXED  elemental_bulk_v1: "
            f"MAE={payload['elemental_bulk_v1']['mae']:.4f} "
            f"bias={payload['elemental_bulk_v1']['mean_bias']:+.4f} "
            f"Spearman={payload['elemental_bulk_v1']['spearman_r']:.3f}"
        )
        print(f"  verdict: {payload['verdict']}")
        print(
            "  Promote bar (LightGBM E_f): ~0.16 eV/atom on structure-gated intersection."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
