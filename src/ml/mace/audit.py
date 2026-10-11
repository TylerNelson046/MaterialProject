"""Audit ``structure_path`` loadability for a training-export target (Phase 4.2)."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import PROJECT_ROOT
from ml.mace.config import load_mace_config
from ml.mace.structures import load_structure, structure_row_dict


@dataclass(frozen=True)
class StructureAuditResult:
    """Paths and summary stats from one audit run."""

    target: str
    out_dir: Path
    audit_parquet: Path
    summary_md: Path
    summary_json: Path
    n_rows: int
    n_ok: int
    n_fail: int
    reason_counts: dict[str, int]


def _export_target_parquet(target: str, config: dict[str, Any] | None = None) -> Path:
    cfg = config if config is not None else load_mace_config()
    export = cfg.get("export") or {}
    export_dir = PROJECT_ROOT / str(export.get("dir") or "data/processed/training_exports")
    sub = str(export.get("targets_subdir") or "targets")
    targets = cfg.get("targets") or {}
    spec = targets.get(target) or {}
    parquet_name = str(spec.get("parquet") or f"{target}.parquet")
    path = export_dir / sub / parquet_name
    if not path.is_file():
        raise FileNotFoundError(f"Target parquet not found: {path}")
    return path


def default_audit_dir(target: str, config: dict[str, Any] | None = None) -> Path:
    """``data/processed/mace_exports/structure_audit/<target>/``."""
    cfg = config if config is not None else load_mace_config()
    export = cfg.get("export") or {}
    pack = PROJECT_ROOT / str(export.get("mace_pack_dir") or "data/processed/mace_exports")
    return pack / "structure_audit" / target


def _audit_one_row(
    payload: tuple[str, str, str, str | None, int | None],
) -> dict[str, Any]:
    """Worker-friendly single-row audit (picklable args)."""
    material_id, split, source, structure_path, max_atoms = payload
    result = load_structure(
        structure_path,
        attach_atoms=False,
        max_atoms=max_atoms,
    )
    return structure_row_dict(
        result,
        material_id=material_id,
        split=split,
        source=source,
    )


def audit_structures(
    target: str = "formation_energy_per_atom",
    *,
    limit: int | None = None,
    split: str | None = None,
    max_atoms: int | None = None,
    out_dir: Path | None = None,
    config: dict[str, Any] | None = None,
    attach_atoms: bool = False,
    progress_every: int = 5000,
    workers: int = 1,
) -> StructureAuditResult:
    """Load each row's structure and write audit parquet + summary.

    Failed loads are quarantined with reason codes; they are never marked OK.
    """
    cfg = config if config is not None else load_mace_config()
    structure_cfg = cfg.get("structure") or {}
    path_col = str(structure_cfg.get("path_column") or "structure_path")
    if max_atoms is None:
        raw = structure_cfg.get("max_atoms_eval")
        max_atoms = int(raw) if raw is not None else None

    parquet_path = _export_target_parquet(target, cfg)
    columns = ["material_id", "split", "source", path_col]
    frame = pd.read_parquet(parquet_path, columns=columns)
    if split is not None:
        frame = frame.loc[frame["split"].astype(str) == str(split)].copy()
    if limit is not None:
        frame = frame.head(int(limit)).copy()

    jobs: list[tuple[str, str, str, str | None, int | None]] = []
    for rec in frame.itertuples(index=False):
        raw_path = getattr(rec, path_col, None)
        if raw_path is None or pd.isna(raw_path):
            structure_path: str | None = None
        else:
            text = str(raw_path).strip()
            structure_path = text if text and text.lower() not in {"none", "null", "nan"} else None
        jobs.append(
            (
                str(getattr(rec, "material_id", "")),
                str(getattr(rec, "split", "")),
                str(getattr(rec, "source", "")),
                structure_path,
                max_atoms,
            )
        )

    rows: list[dict[str, Any]] = []
    n_jobs = len(jobs)
    if int(workers) <= 1 or attach_atoms:
        for i, job in enumerate(jobs, start=1):
            if attach_atoms:
                material_id, split_v, source, structure_path, max_a = job
                result = load_structure(
                    structure_path,
                    attach_atoms=True,
                    max_atoms=max_a,
                )
                rows.append(
                    structure_row_dict(
                        result,
                        material_id=material_id,
                        split=split_v,
                        source=source,
                    )
                )
            else:
                rows.append(_audit_one_row(job))
            if progress_every and i % int(progress_every) == 0:
                print(f"audited {i}/{n_jobs} …", flush=True)
    else:
        done = 0
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futures = [pool.submit(_audit_one_row, job) for job in jobs]
            for fut in as_completed(futures):
                rows.append(fut.result())
                done += 1
                if progress_every and done % int(progress_every) == 0:
                    print(f"audited {done}/{n_jobs} …", flush=True)

    audit = pd.DataFrame(rows)
    dest = out_dir or default_audit_dir(target, cfg)
    dest.mkdir(parents=True, exist_ok=True)
    audit_path = dest / "audit.parquet"
    audit.to_parquet(audit_path, index=False)

    n_rows = int(len(audit))
    n_ok = int(audit["load_ok"].sum()) if n_rows else 0
    n_fail = n_rows - n_ok
    reason_counts = dict(Counter(audit["reason"].astype(str).tolist()))

    summary = {
        "target": target,
        "created_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_parquet": str(parquet_path.relative_to(PROJECT_ROOT)),
        "n_rows": n_rows,
        "n_ok": n_ok,
        "n_fail": n_fail,
        "ok_fraction": (n_ok / n_rows) if n_rows else None,
        "limit": limit,
        "split_filter": split,
        "max_atoms": max_atoms,
        "reason_counts": reason_counts,
    }
    summary_json = dest / "summary.json"
    summary_md = dest / "SUMMARY.md"
    summary_json.write_text(
        __import__("json").dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_md.write_text(_format_summary_md(summary), encoding="utf-8")

    return StructureAuditResult(
        target=target,
        out_dir=dest,
        audit_parquet=audit_path,
        summary_md=summary_md,
        summary_json=summary_json,
        n_rows=n_rows,
        n_ok=n_ok,
        n_fail=n_fail,
        reason_counts=reason_counts,
    )


def _format_summary_md(summary: dict[str, Any]) -> str:
    lines = [
        f"# Structure audit — `{summary['target']}`",
        "",
        f"- Created (UTC): `{summary['created_at_utc']}`",
        f"- Source: `{summary['source_parquet']}`",
        f"- Rows: **{summary['n_rows']}**",
        f"- Load OK: **{summary['n_ok']}** ({100.0 * (summary['ok_fraction'] or 0):.2f}%)",
        f"- Load fail: **{summary['n_fail']}**",
        f"- Limit: `{summary['limit']}`",
        f"- Split filter: `{summary['split_filter']}`",
        f"- max_atoms: `{summary['max_atoms']}`",
        "",
        "## Reason counts",
        "",
        "| Reason | N |",
        "|--------|--:|",
    ]
    counts: dict[str, int] = summary.get("reason_counts") or {}
    for reason, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| `{reason}` | {n} |")
    lines.append("")
    return "\n".join(lines)
