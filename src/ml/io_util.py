"""Small JSON helpers shared by training artifact writers/readers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    """Load JSON from ``path``. Returns ``None`` if the file is missing."""
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    """Write pretty-printed JSON (sorted keys) with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def relpath_under(path: Path, root: Path) -> str:
    """Return ``path`` relative to ``root`` when possible, else absolute string."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
