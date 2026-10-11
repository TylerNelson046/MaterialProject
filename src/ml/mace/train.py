"""Energy-only MACE fine-tuning on the structure pack (Phase 4.5)."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from config.settings import PROJECT_ROOT
from ml.io_util import relpath_under, write_json
from ml.mace.checkpoints import ensure_foundation_checkpoint
from ml.mace.config import foundation_model_id, load_mace_config
from ml.mace.device import resolve_device
from ml.mace.eval_pretrained import eval_pretrained
from ml.mace.pack import load_mace_pack_frame
from ml.mace.reference import (
    load_formation_references,
    reference_table_payload,
    write_reference_energies,
)
from ml.mace.runtime import configure_mace_runtime, warn_if_unstable_python
from ml.mace.xyz_export import e0s_cli_string, export_pack_xyz


@dataclass(frozen=True)
class FineTuneResult:
    """Artifacts from one fine-tune + eval run."""

    target: str
    run_id: str
    run_dir: Path
    checkpoint: Path
    train_log: Path
    metrics: dict[str, Any]
    counts: dict[str, int]


def _run_id(target: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{target}_mace_finetuned_{stamp}"


def _models_root(config: dict[str, Any], project_root: Path) -> Path:
    output = config.get("output") or {}
    return project_root / str(output.get("models_dir") or "data/models")


def _mace_run_dir(
    config: dict[str, Any], target: str, run_id: str, project_root: Path
) -> Path:
    output = config.get("output") or {}
    sub = str(output.get("mace_subdir") or "mace")
    return _models_root(config, project_root) / sub / target / run_id


def _training_defaults(config: dict[str, Any]) -> dict[str, Any]:
    train = dict(config.get("training") or {})
    # Phase 4.5 energy-only fine-tune defaults (CPU-friendly; override in yaml/CLI).
    train.setdefault("energy_only", True)
    train.setdefault("seed", 42)
    train.setdefault("batch_size", 2)
    train.setdefault("valid_batch_size", 2)
    train.setdefault("lr", 0.0001)
    train.setdefault("max_num_epochs", 20)
    train.setdefault("patience", 8)
    train.setdefault("energy_weight", 1.0)
    train.setdefault("forces_weight", 0.0)
    train.setdefault("ema", True)
    train.setdefault("scaling", "std_scaling")
    train.setdefault("num_workers", 0)
    train.setdefault("eval_limit_per_split", None)
    return train


def _find_trained_model(work_dir: Path, name: str) -> Path:
    """Locate the ``.model`` artifact written by ``mace_run_train``."""
    candidates = [
        work_dir / f"{name}.model",
        work_dir / "checkpoints" / f"{name}_compiled.model",
        work_dir / "checkpoints" / f"{name}.model",
    ]
    # Prefer newest .model under work_dir.
    found = sorted(work_dir.rglob("*.model"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates + found:
        if path.is_file() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError(
        f"No fine-tuned .model found under {work_dir}. See train.log for errors."
    )


def _build_mace_train_cmd(
    *,
    name: str,
    work_dir: Path,
    train_xyz: Path,
    valid_xyz: Path,
    foundation_ckpt: Path,
    e0s: dict[int, float],
    device: str,
    dtype: str,
    train_cfg: dict[str, Any],
) -> list[str]:
    exe = shutil.which("mace_run_train")
    if exe is None:
        # Fallback: module entry via python -m is not always available; try console script path.
        raise RuntimeError(
            "mace_run_train not found on PATH. Ensure `[mace]` is installed "
            '(`pip install -e ".[mace]"`) and the venv is activated.'
        )
    cmd = [
        exe,
        f"--name={name}",
        f"--work_dir={work_dir}",
        f"--device={device}",
        f"--default_dtype={dtype}",
        f"--foundation_model={foundation_ckpt}",
        "--multiheads_finetuning=False",
        "--foundation_model_elements=True",
        f"--train_file={train_xyz}",
        f"--valid_file={valid_xyz}",
        f"--E0s={e0s_cli_string(e0s)}",
        "--energy_key=REF_energy",
        "--forces_key=REF_forces",
        f"--energy_weight={train_cfg['energy_weight']}",
        f"--forces_weight={train_cfg['forces_weight']}",
        "--compute_forces=True",
        f"--scaling={train_cfg['scaling']}",
        f"--batch_size={train_cfg['batch_size']}",
        f"--valid_batch_size={train_cfg['valid_batch_size']}",
        f"--lr={train_cfg['lr']}",
        f"--max_num_epochs={train_cfg['max_num_epochs']}",
        f"--patience={train_cfg['patience']}",
        f"--seed={train_cfg['seed']}",
        f"--num_workers={train_cfg['num_workers']}",
        "--error_table=PerAtomMAE",
    ]
    if train_cfg.get("ema", True):
        cmd.append("--ema")
    return cmd


def train_mace(
    target: str = "formation_energy_per_atom",
    *,
    device: str | None = "cpu",
    model_id: str | None = None,
    max_train_rows: int | None = None,
    max_val_rows: int | None = None,
    max_test_rows: int | None = None,
    max_num_epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    config: dict[str, Any] | None = None,
    project_root: Path | None = None,
    skip_eval: bool = False,
    eval_limit_per_split: int | None = None,
    register: bool = True,
) -> FineTuneResult:
    """Fine-tune foundation MACE on pack labels (energy-only) and evaluate."""
    configure_mace_runtime()
    warn_if_unstable_python()

    cfg = config if config is not None else load_mace_config()
    root = project_root or PROJECT_ROOT
    train_cfg = _training_defaults(cfg)
    if max_train_rows is not None:
        train_cfg["max_train_rows"] = max_train_rows
    else:
        max_train_rows = train_cfg.get("max_train_rows")
    if max_num_epochs is not None:
        train_cfg["max_num_epochs"] = int(max_num_epochs)
    if batch_size is not None:
        train_cfg["batch_size"] = int(batch_size)
        train_cfg["valid_batch_size"] = int(batch_size)
    if lr is not None:
        train_cfg["lr"] = float(lr)
    if eval_limit_per_split is not None:
        train_cfg["eval_limit_per_split"] = eval_limit_per_split

    if not train_cfg.get("energy_only", True):
        raise ValueError("Phase 4.5 supports energy_only=true only")
    # Last-resort path only — frozen-first strategy keeps FT out of the default plan.
    if (cfg.get("energy") or {}).get("frozen_first", True) and not (
        cfg.get("energy") or {}
    ).get("allow_finetune_default", False):
        import warnings

        warnings.warn(
            "train_mace is a last-resort path under frozen-first strategy. "
            "Verify elemental_bulk_v1 harness + promote gate before fine-tuning.",
            UserWarning,
            stacklevel=2,
        )

    from ml.mace.runtime import coerce_stable_dtype

    resolved_model = model_id or foundation_model_id(cfg)
    resolved_device = resolve_device(device)
    foundation = cfg.get("foundation") or {}
    dtype = coerce_stable_dtype(str(foundation.get("default_dtype") or "float32"))
    foundation_ckpt = ensure_foundation_checkpoint(resolved_model, config=cfg)
    # Load calculator once so bulk elemental μ can be computed if cache missing.
    from ml.mace.calculator import load_foundation_calculator

    calc, _, _, _ = load_foundation_calculator(
        model_id=resolved_model,
        device=resolved_device,
        config=cfg,
        checkpoint=foundation_ckpt,
    )
    e0s, protocol = load_formation_references(
        resolved_model, calc=calc, config=cfg
    )

    run_id = _run_id(target)
    run_dir = _mace_run_dir(cfg, target, run_id, root)
    data_dir = run_dir / "xyz"
    work_dir = run_dir / "mace_work"
    ckpt_dir = run_dir / "checkpoint"
    run_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    write_reference_energies(
        run_dir / "reference_energies.json",
        reference_table_payload(e0s, protocol=protocol, model_id=resolved_model),
    )
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump({"mace": cfg, "training_resolved": train_cfg}, sort_keys=True),
        encoding="utf-8",
    )

    pack = load_mace_pack_frame(target, train_ok_only=True, config=cfg)
    xyz = export_pack_xyz(
        pack,
        out_dir=data_dir,
        e0s=e0s,
        max_train_rows=max_train_rows,
        max_val_rows=max_val_rows,
        max_test_rows=max_test_rows,
    )
    if xyz["train"].n_written < 1 or xyz["val"].n_written < 1:
        raise RuntimeError(
            f"Need non-empty train/val xyz; got train={xyz['train'].n_written} "
            f"val={xyz['val'].n_written}"
        )

    name = "mace_ft"
    cmd = _build_mace_train_cmd(
        name=name,
        work_dir=work_dir,
        train_xyz=xyz["train"].path,
        valid_xyz=xyz["val"].path,
        foundation_ckpt=foundation_ckpt,
        e0s=e0s,
        device=resolved_device,
        dtype=dtype,
        train_cfg=train_cfg,
    )
    log_path = run_dir / "train.log"
    write_json(
        run_dir / "train_command.json",
        {"cmd": cmd, "cwd": str(work_dir), "xyz": {k: str(v.path) for k, v in xyz.items()}},
    )

    env = {
        **os.environ,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "1"),
        "TORCH_NUM_THREADS": os.environ.get("TORCH_NUM_THREADS", "1"),
    }
    print("Running:", " ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(work_dir),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"mace_run_train failed (exit={proc.returncode}). See {log_path}"
        )

    trained = _find_trained_model(work_dir, name)
    dest_ckpt = ckpt_dir / "model.model"
    shutil.copy2(trained, dest_ckpt)

    metrics: dict[str, Any] = {}
    counts = {k: v.n_written for k, v in xyz.items()}
    if not skip_eval:
        eval_result = eval_pretrained(
            target,
            splits=("val", "test"),
            limit_per_split=train_cfg.get("eval_limit_per_split"),
            device=resolved_device,
            model_id=resolved_model,
            checkpoint=dest_ckpt,
            model_family="mace_finetuned",
            run_id=run_id,
            run_dir=run_dir,
            config=cfg,
            project_root=root,
            write_predictions=True,
            register=register,
        )
        metrics = eval_result.metrics
        # Preserve xyz counts alongside eval counts.
        counts = {**counts, **{f"eval_{k}": v for k, v in eval_result.counts.items()}}
    else:
        record = {
            "target": target,
            "tier": ((cfg.get("targets") or {}).get(target) or {}).get("tier") or "Ready",
            "model_family": "mace_finetuned",
            "run_id": run_id,
            "run_dir": relpath_under(run_dir, root),
            "created_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "foundation_model": resolved_model,
            "checkpoint": relpath_under(dest_ckpt, root),
            "device": resolved_device,
            "reference_protocol": protocol,
            "counts": counts,
            "metrics": {},
            "train_cfg": train_cfg,
        }
        write_json(run_dir / "run_record.json", record)
        if register:
            from ml.registry import register_run

            register_run(_models_root(cfg, root), record)

    return FineTuneResult(
        target=target,
        run_id=run_id,
        run_dir=run_dir,
        checkpoint=dest_ckpt,
        train_log=log_path,
        metrics=metrics,
        counts=counts,
    )
