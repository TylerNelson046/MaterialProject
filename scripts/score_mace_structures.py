#!/usr/bin/env python3
"""Score candidate structures: optional relax → E_form → E_hull (frozen MACE).

Examples::

    python scripts/score_mace_structures.py --structure path/to/POSCAR --no-hull
    python scripts/score_mace_structures.py --material-id MAT-1 --material-id MAT-2
    python scripts/score_mace_structures.py --structure a.json --out scores.parquet
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
    parser.add_argument(
        "--structure",
        action="append",
        dest="structures",
        default=None,
        help="Structure file path (repeatable).",
    )
    parser.add_argument(
        "--material-id",
        action="append",
        dest="material_ids",
        default=None,
        help="Pack material_id (repeatable).",
    )
    parser.add_argument("--target", default="formation_energy_per_atom")
    parser.add_argument(
        "--no-relax",
        action="store_true",
        help="Skip FIRE relax (as-loaded geometry only).",
    )
    parser.add_argument("--relax-fmax", type=float, default=0.05)
    parser.add_argument("--relax-steps", type=int, default=50)
    parser.add_argument(
        "--no-hull",
        action="store_true",
        help="Skip E_hull (requires E_form cache or --hull-reference).",
    )
    parser.add_argument(
        "--hull-reference",
        type=Path,
        default=None,
        help="Parquet with e_form_per_atom (+ formula) for the hull pool.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=None, help="Write parquet/CSV.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from ml.mace.score import score_structures

    frame = score_structures(
        structure_paths=args.structures,
        material_ids=args.material_ids,
        target=args.target,
        relax=not args.no_relax,
        relax_fmax=args.relax_fmax,
        relax_steps=args.relax_steps,
        compute_hull=not args.no_hull,
        hull_reference_parquet=args.hull_reference,
        device=args.device,
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.out.suffix.lower() == ".csv":
            frame.to_csv(args.out, index=False)
        else:
            frame.to_parquet(args.out, index=False)

    if args.json:
        print(json.dumps(frame.to_dict(orient="records"), indent=2, default=str))
    else:
        cols = [
            c
            for c in (
                "material_id",
                "formula",
                "formation_energy_per_atom",
                "e_above_hull",
                "energy_per_atom_eV",
                "relaxed",
                "relax_converged",
                "ok",
                "reason",
            )
            if c in frame.columns
        ]
        print(frame.loc[:, cols].to_string(index=False))
        if args.out is not None:
            print(f"Wrote {args.out}")
    if frame.empty:
        return 1
    if "hull_note" in frame.columns:
        notes = [
            str(n)
            for n in frame["hull_note"].tolist()
            if n is not None and str(n) not in {"", "None", "nan"}
        ]
        if notes and not args.json:
            extra = f" (+{len(notes) - 1} more)" if len(notes) > 1 else ""
            print(f"hull notes: {notes[0]}{extra}")
    if not bool(frame["ok"].all()):
        return 1
    # Hull requested but every row missing e_above_hull → incomplete score.
    if not args.no_hull and "e_above_hull" in frame.columns:
        hull_ok = frame["e_above_hull"].notna()
        if bool(frame["ok"].any()) and not bool(hull_ok.any()):
            if not args.json:
                print(
                    "No E_hull values computed (missing covering refs or cache). "
                    "Use --no-hull or expand scripts/cache_mace_eform.py.",
                    file=sys.stderr,
                )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
