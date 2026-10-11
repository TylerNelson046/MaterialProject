#!/usr/bin/env python3
"""Phase 4.9: plan or execute a named MACE pipeline preset.

Examples::

    python scripts/run_mace_pipeline.py --preset smoke --dry-run
    python scripts/run_mace_pipeline.py --preset smoke --stages pack density
    python scripts/run_mace_pipeline.py --preset full --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        default="smoke",
        choices=["smoke", "full"],
        help="Named run preset from configs/mace.yaml → runs.*",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=None,
        help=(
            "Subset of stages: audit pack eval_pretrained train_finetune "
            "density hull embeddings embedding_head"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without executing.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from ml.mace.pipeline import format_pipeline_plan, get_run_preset, plan_pipeline

    preset = get_run_preset(args.preset)
    steps = plan_pipeline(
        args.preset,
        stages=tuple(args.stages) if args.stages else None,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "preset": args.preset,
                    "preset_resolved": preset,
                    "dry_run": args.dry_run,
                    "steps": [
                        {"name": s.name, "command": s.command, "note": s.note}
                        for s in steps
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(f"Preset `{args.preset}`: {preset.get('description', '')}")
        print(format_pipeline_plan(steps))

    if args.dry_run:
        return 0

    for step in steps:
        print(f"\n>>> {step.name}", flush=True)
        print(" ".join(step.command), flush=True)
        proc = subprocess.run(step.command, cwd=ROOT, check=False)
        if proc.returncode != 0:
            print(
                f"Pipeline stopped at {step.name} (exit={proc.returncode})",
                file=sys.stderr,
            )
            return int(proc.returncode)
    print("\nPipeline OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
