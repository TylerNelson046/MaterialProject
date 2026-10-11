"""Foundation-model element coverage (MACE-MPA-0 / MACE-MP family).

Pinned from the ``medium-mpa-0`` checkpoint ``atomic_numbers`` tensor:
Z = 1–83 and 89–94 (89 elements; Po–Ra gap).
"""

from __future__ import annotations

# Z=1..83, 89..94 — matches mace-mpa-0-medium.model
MACE_MPA0_ATOMIC_NUMBERS: frozenset[int] = frozenset([*range(1, 84), *range(89, 95)])

assert len(MACE_MPA0_ATOMIC_NUMBERS) == 89
