"""Download / refresh external datasets into data/raw/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config.settings import clear_settings_cache, get_settings  # noqa: E402
from ingestion.ingest_aflow import ingest_aflow  # noqa: E402
from ingestion.ingest_cod import ingest_cod  # noqa: E402
from ingestion.ingest_jarvis import ingest_jarvis  # noqa: E402
from ingestion.ingest_mp import ingest_materials_project  # noqa: E402
from ingestion.ingest_oqmd import ingest_oqmd  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download external materials datasets.")
    parser.add_argument(
        "--source",
        choices=["materials_project", "jarvis", "oqmd", "aflow", "cod", "all"],
        default="materials_project",
        help="Which source to download.",
    )
    parser.add_argument(
        "--max-materials",
        type=int,
        default=None,
        help="Override max materials (default: configs/database.yaml).",
    )
    parser.add_argument(
        "--no-structures",
        action="store_true",
        help="Skip crystal structure export where applicable.",
    )
    parser.add_argument(
        "--with-structures",
        action="store_true",
        help="Export crystal structures where applicable.",
    )
    parser.add_argument(
        "--force-renew",
        action="store_true",
        help="Force refresh of the local MP Delta dataset via mp-api.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clear_settings_cache()
    settings = get_settings()
    settings.ensure_directories()

    if args.source in {"materials_project", "all"}:
        download_structures = settings.mp_download_structures
        if args.no_structures:
            download_structures = False
        if args.with_structures:
            download_structures = True
        ingest_materials_project(
            settings=settings,
            max_materials=args.max_materials,
            download_structures=download_structures,
            force_renew=args.force_renew,
        )

    if args.source in {"jarvis", "all"}:
        download_structures = True
        if args.no_structures:
            download_structures = False
        if args.with_structures:
            download_structures = True
        ingest_jarvis(
            settings=settings,
            max_materials=args.max_materials,
            download_structures=download_structures,
        )

    if args.source in {"oqmd", "all"}:
        ingest_oqmd(settings=settings, max_materials=args.max_materials)

    if args.source in {"aflow", "all"}:
        ingest_aflow(settings=settings, max_materials=args.max_materials)

    if args.source in {"cod", "all"}:
        download_structures = True
        if args.no_structures:
            download_structures = False
        if args.with_structures:
            download_structures = True
        ingest_cod(
            settings=settings,
            max_materials=args.max_materials,
            download_structures=download_structures,
        )


if __name__ == "__main__":
    main()
