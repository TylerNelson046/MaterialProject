"""Dependency gate helpers (safe without [mace] installed)."""

from __future__ import annotations

import pytest

from ml.mace import mace_deps_available, require_mace_deps


def test_mace_deps_available_is_bool() -> None:
    assert isinstance(mace_deps_available(), bool)


def test_require_mace_deps_message_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ml.mace.mace_deps_available", lambda: False)
    with pytest.raises(ImportError, match=r'pip install -e "\.\[mace\]"'):
        require_mace_deps()
