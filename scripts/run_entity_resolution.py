"""Run entity resolution against the unified DuckDB store."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config.settings import clear_settings_cache, get_settings  # noqa: E402
from database.entity_resolution import resolve_entities  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Link cross-database materials onto shared MAT-* IDs."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate merge groups without writing.",
    )
    parser.add_argument(
        "--allow-same-source",
        action="store_true",
        help="Also merge duplicates within a single database (off by default).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clear_settings_cache()
    settings = get_settings()
    settings.ensure_directories()

    stats = resolve_entities(
        settings=settings,
        dry_run=args.dry_run,
        require_cross_source=not args.allow_same_source,
    )
    print(json.dumps(stats.to_dict(), indent=2))
    if args.dry_run:
        print("Dry run only — database unchanged.")
    else:
        print("Entity resolution complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
