"""Cross-process exclusive lock for DuckDB writes."""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from config.settings import Settings, get_settings


@contextmanager
def duckdb_write_lock(
    settings: Settings | None = None,
    *,
    poll_seconds: float = 2.0,
    label: str = "duckdb",
) -> Iterator[None]:
    """Serialize DuckDB writers across pipelines (fcntl flock).

    Download/ingest of raw JSONL does not need this lock — only DB load steps.
    """
    import fcntl

    settings = settings or get_settings()
    lock_path = Path(settings.duckdb_path).with_suffix(".duckdb.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    handle = lock_path.open("a+", encoding="utf-8")
    waited = False
    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not waited:
                    print(
                        f"[{label}] waiting for DuckDB write lock "
                        f"({lock_path.name})...",
                        flush=True,
                    )
                    waited = True
                time.sleep(poll_seconds)
        if waited:
            print(f"[{label}] acquired DuckDB write lock", flush=True)
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
