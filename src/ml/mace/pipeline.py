"""Named MACE run presets and pipeline orchestration (Phase 4.9)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ml.mace.config import load_mace_config

DEFAULT_SMOKE: dict[str, Any] = {
    "description": "CPU-friendly frozen-first smoke (no fine-tune).",
    "device": "cpu",
    "audit_limit": 1000,
    "eval_limit_per_split": 200,
    "hull_limit_reference": 128,
    "hull_limit_per_split": 64,
    "density_limit_per_split": 200,
    "density_cod_limit": 100,
    "pack_targets": [
        "formation_energy_per_atom",
        "density",
        "energy_above_hull",
    ],
}

DEFAULT_FULL: dict[str, Any] = {
    "description": "Full frozen eval (CPU or auto device); no fine-tune.",
    "device": "auto",
    "audit_limit": None,
    "eval_limit_per_split": None,
    "hull_limit_reference": None,
    "hull_limit_per_split": None,
    "density_limit_per_split": None,
    "density_cod_limit": None,
    "pack_targets": [
        "formation_energy_per_atom",
        "density",
        "energy_above_hull",
    ],
}


@dataclass(frozen=True)
class PipelineStep:
    """One planned or executed pipeline step."""

    name: str
    command: list[str]
    note: str = ""


def get_run_preset(
    name: str = "smoke",
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a merged run preset from ``configs/mace.yaml`` → ``runs``."""
    cfg = config if config is not None else load_mace_config()
    defaults = {"smoke": DEFAULT_SMOKE, "full": DEFAULT_FULL}
    base = dict(defaults.get(name) or {})
    if not base and name not in (cfg.get("runs") or {}):
        raise KeyError(
            f"Unknown run preset {name!r}; expected one of "
            f"{sorted(set(defaults) | set(cfg.get('runs') or {}))}"
        )
    if not base:
        base = {}
    overlay = dict((cfg.get("runs") or {}).get(name) or {})
    merged = {**base, **overlay}
    if not merged:
        raise KeyError(f"Empty run preset {name!r}")
    return merged


def plan_pipeline(
    preset: str = "smoke",
    *,
    stages: tuple[str, ...] | None = None,
    config: dict[str, Any] | None = None,
) -> list[PipelineStep]:
    """Build an ordered list of CLI steps for a named preset.

    Stages (default all)::

        audit, pack, eval_pretrained, train_finetune, density, eform_cache,
        hull, embeddings, embedding_head
    """
    cfg = config if config is not None else load_mace_config()
    run = get_run_preset(preset, config=cfg)
    device = str(run.get("device") or "cpu")
    # Frozen-first default: no train_finetune unless explicitly requested.
    all_stages = stages or (
        "audit",
        "pack",
        "eval_pretrained",
        "density",
        "eform_cache",
        "hull",
    )
    steps: list[PipelineStep] = []
    pack_targets = list(run.get("pack_targets") or ["formation_energy_per_atom"])

    if "audit" in all_stages:
        cmd = [
            "python",
            "scripts/audit_mace_structures.py",
            "--target",
            "formation_energy_per_atom",
            "--workers",
            "4",
        ]
        if run.get("audit_limit") is not None:
            cmd.extend(["--limit", str(int(run["audit_limit"]))])
        steps.append(
            PipelineStep(
                "audit",
                cmd,
                "Primary structure audit (other packs reuse via reuse_audit_from).",
            )
        )

    if "pack" in all_stages:
        for target in pack_targets:
            steps.append(
                PipelineStep(
                    f"pack:{target}",
                    ["python", "scripts/export_mace_pack.py", "--target", target],
                    "Structure-gated pack export.",
                )
            )

    if "eval_pretrained" in all_stages:
        cmd = [
            "python",
            "scripts/eval_mace_pretrained.py",
            "--device",
            device,
            "--splits",
            "val",
            "test",
        ]
        if run.get("eval_limit_per_split") is not None:
            cmd.extend(["--limit-per-split", str(int(run["eval_limit_per_split"]))])
        steps.append(PipelineStep("eval_pretrained", cmd, "Zero-shot E_form eval."))

    if "train_finetune" in all_stages:
        cmd = [
            "python",
            "scripts/train_mace.py",
            "--device",
            device,
            "--epochs",
            str(int(run.get("train_epochs") or 20)),
        ]
        if run.get("train_max_train_rows") is not None:
            cmd.extend(["--max-train-rows", str(int(run["train_max_train_rows"]))])
        if run.get("train_eval_limit_per_split") is not None:
            cmd.extend(
                [
                    "--eval-limit-per-split",
                    str(int(run["train_eval_limit_per_split"])),
                ]
            )
        steps.append(PipelineStep("train_finetune", cmd, "Energy-only fine-tune."))

    if "density" in all_stages:
        cmd = [
            "python",
            "scripts/eval_mace_density.py",
            "--protocol",
            "geometry_v1",
        ]
        if run.get("density_limit_per_split") is not None:
            cmd.extend(
                ["--limit-per-split", str(int(run["density_limit_per_split"]))]
            )
        if run.get("density_cod_limit") is not None:
            cmd.extend(["--cod-limit", str(int(run["density_cod_limit"]))])
        steps.append(PipelineStep("density", cmd, "Geometry density protocol."))

    if "eform_cache" in all_stages:
        cmd = [
            "python",
            "scripts/cache_mace_eform.py",
            "--device",
            device,
        ]
        # Prefer hull reference limit for the train-heavy cache smoke.
        cache_limit = run.get("eform_cache_limit_per_split")
        if cache_limit is None:
            cache_limit = run.get("hull_limit_reference")
        if cache_limit is not None:
            cmd.extend(["--limit-per-split", str(int(cache_limit))])
        steps.append(
            PipelineStep(
                "eform_cache",
                cmd,
                "Resumable frozen E_form cache (hull + score).",
            )
        )

    if "hull" in all_stages:
        cmd = ["python", "scripts/eval_mace_hull.py", "--device", device]
        if run.get("hull_limit_reference") is not None:
            cmd.extend(
                ["--limit-reference", str(int(run["hull_limit_reference"]))]
            )
        if run.get("hull_limit_per_split") is not None:
            cmd.extend(
                ["--limit-per-split", str(int(run["hull_limit_per_split"]))]
            )
        steps.append(PipelineStep("hull", cmd, "Named E_hull protocol."))

    embedding_targets = {
        "band_gap",
        "bulk_modulus_vrh",
        "shear_modulus_vrh",
        "thermal_conductivity_agl",
        "debye_temperature",
        "heat_capacity_cv",
        "battery_capacity_grav",
    }
    selected_emb = [t for t in pack_targets if t in embedding_targets]
    if not selected_emb and "embeddings" in all_stages:
        selected_emb = ["band_gap"]

    if "embeddings" in all_stages:
        for target in selected_emb:
            cmd = [
                "python",
                "scripts/extract_mace_embeddings.py",
                "--target",
                target,
                "--device",
                device,
            ]
            if run.get("embedding_limit_per_split") is not None:
                cmd.extend(
                    [
                        "--limit-per-split",
                        str(int(run["embedding_limit_per_split"])),
                    ]
                )
            steps.append(
                PipelineStep(
                    f"embeddings:{target}",
                    cmd,
                    "Descriptor cache (resumable).",
                )
            )

    if "embedding_head" in all_stages:
        for target in selected_emb:
            cmd = [
                "python",
                "scripts/train_mace_embedding_head.py",
                "--target",
                target,
                "--device",
                device,
            ]
            if run.get("embedding_limit_per_split") is not None:
                cmd.extend(
                    [
                        "--limit-per-split",
                        str(int(run["embedding_limit_per_split"])),
                    ]
                )
            steps.append(
                PipelineStep(
                    f"embedding_head:{target}",
                    cmd,
                    "LightGBM readout + intersection compare.",
                )
            )

    return steps


def format_pipeline_plan(steps: list[PipelineStep]) -> str:
    """Human-readable plan for ``--dry-run``."""
    lines = ["# MACE pipeline plan", ""]
    for i, step in enumerate(steps, start=1):
        lines.append(f"{i}. **{step.name}** — {step.note}")
        lines.append("   `" + " ".join(step.command) + "`")
        lines.append("")
    return "\n".join(lines)
