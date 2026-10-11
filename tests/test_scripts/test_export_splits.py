"""Tests for stratified leak-safe split assignment."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from export_training_package import assign_splits  # noqa: E402


def test_assign_splits_keeps_groups_together() -> None:
    si = pd.DataFrame(
        {
            "material_id": ["A", "B", "C", "D", "E", "F"],
            "nelements": [1, 1, 2, 2, 3, 3],
            "source": ["oqmd"] * 6,
            "chemsys": ["Li", "Li", "Fe-O", "Fe-O", "Al-O", "Al-O"],
        }
    )
    mid_to_group = {
        "A": "g1",
        "B": "g1",
        "C": "g2",
        "D": "g3",
        "E": "g4",
        "F": "g4",
    }
    splits = assign_splits(si, mid_to_group, seed=0)
    assert set(splits["split"]) <= {"train", "val", "test"}
    assert (
        splits.loc[splits["material_id"] == "A", "split"].iloc[0]
        == splits.loc[splits["material_id"] == "B", "split"].iloc[0]
    )
    assert (
        splits.loc[splits["material_id"] == "E", "split"].iloc[0]
        == splits.loc[splits["material_id"] == "F", "split"].iloc[0]
    )
