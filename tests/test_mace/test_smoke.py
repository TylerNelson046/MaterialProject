"""Optional end-to-end MACE smoke (downloads foundation weights).

Skipped unless ``[mace]`` is installed **and** ``MACE_SMOKE=1`` is set, so
default ``pytest`` stays offline / lightweight.

Runs the smoke CLI in a **subprocess** — some Python 3.14 + torch wheels can
segfault inside the pytest process after a MACE forward.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ml.mace import mace_deps_available

ROOT = Path(__file__).resolve().parents[2]
SMOKE_SCRIPT = ROOT / "scripts" / "smoke_mace.py"

pytestmark = [
    pytest.mark.skipif(
        not mace_deps_available(),
        reason='MACE optional extra not installed (pip install -e ".[mace]")',
    ),
    pytest.mark.skipif(
        os.environ.get("MACE_SMOKE", "").strip() not in {"1", "true", "yes"},
        reason="Set MACE_SMOKE=1 to run foundation download + energy smoke",
    ),
]


def test_run_smoke_cpu_finite_energy() -> None:
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    proc = subprocess.run(
        [sys.executable, str(SMOKE_SCRIPT), "--device", "cpu", "--json"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
    )
    assert proc.returncode == 0, (
        f"smoke_mace.py failed (code={proc.returncode})\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    # CLI may print torch warnings on stderr; JSON is the last non-empty stdout block.
    stdout = proc.stdout.strip()
    payload = json.loads(stdout[stdout.rfind("{") :])
    assert payload["model_id"] == "medium-mpa-0"
    assert payload["device"] == "cpu"
    assert payload["n_atoms"] >= 1
    assert abs(float(payload["energy_eV"])) < 1e6
    assert payload["energy_per_atom_eV"] == pytest.approx(payload["energy_eV"] / payload["n_atoms"])
