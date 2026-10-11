"""Device resolution tests (skip when [mace] extra missing)."""

from __future__ import annotations

import pytest

from ml.mace import mace_deps_available

pytestmark = pytest.mark.skipif(
    not mace_deps_available(),
    reason='MACE optional extra not installed (pip install -e ".[mace]")',
)


def test_resolve_device_cpu_explicit() -> None:
    from ml.mace.device import resolve_device

    assert resolve_device("cpu") == "cpu"


def test_resolve_device_invalid() -> None:
    from ml.mace.device import resolve_device

    with pytest.raises(ValueError, match="Unknown device"):
        resolve_device("tpu")


def test_resolve_device_cuda_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from ml.mace import device as device_mod

    monkeypatch.setattr(device_mod, "cuda_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA requested"):
        device_mod.resolve_device("cuda")


def test_resolve_device_preference_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from ml.mace import device as device_mod

    monkeypatch.setattr(device_mod, "cuda_available", lambda: False)
    monkeypatch.setattr(device_mod, "mps_available", lambda: False)
    assert device_mod.resolve_device(preference=("cuda", "mps", "cpu")) == "cpu"


def test_resolve_device_auto_uses_preference(monkeypatch: pytest.MonkeyPatch) -> None:
    from ml.mace import device as device_mod

    monkeypatch.setattr(device_mod, "cuda_available", lambda: False)
    monkeypatch.setattr(device_mod, "mps_available", lambda: False)
    assert device_mod.resolve_device("auto", preference=("cuda", "cpu")) == "cpu"


def test_describe_devices_keys() -> None:
    from ml.mace.device import describe_devices

    info = describe_devices()
    assert info["cpu"] is True
    assert "torch_version" in info
    assert "cuda" in info
    assert "mps" in info
