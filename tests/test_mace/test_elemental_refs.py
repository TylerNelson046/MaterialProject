"""elemental_bulk_v1 helpers (no torch required for formula unit tests)."""

from __future__ import annotations

from ml.mace.elemental_refs import _ASE_FALLBACK, PROTOCOL_ELEMENTAL_BULK_V1
from ml.mace.reference import PROTOCOL_ELEMENTAL_BULK_V1 as REF_PROTOCOL


def test_protocol_name_stable() -> None:
    assert PROTOCOL_ELEMENTAL_BULK_V1 == "elemental_bulk_v1"
    assert REF_PROTOCOL == PROTOCOL_ELEMENTAL_BULK_V1


def test_ase_fallback_covers_common_elements() -> None:
    for sym in ("H", "O", "Si", "Al", "Fe", "Cu"):
        assert sym in _ASE_FALLBACK
