"""Lock in the TF32 / memory-cap escape hatches (gpu_meanrev.gpu.tune_backend).

These knobs exist so an older number can be recomputed under the previous
precision setting. Nothing else in the repo references them, so without a test
they are one refactor away from silently disappearing.
"""
from __future__ import annotations

import pytest

from gpu_meanrev.gpu import get_device, tune_backend

torch = pytest.importorskip("torch")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")


def test_reports_state_without_raising():
    state = tune_backend()
    assert set(state) >= {"tf32", "mem_fraction"}
    assert "error" not in state, state


@cuda
def test_default_enables_tf32():
    torch.backends.cuda.matmul.allow_tf32 = False
    assert tune_backend()["tf32"] is True
    assert torch.backends.cuda.matmul.allow_tf32 is True


@cuda
def test_gpu_tf32_zero_leaves_matmul_precision_untouched(monkeypatch):
    monkeypatch.setenv("GPU_TF32", "0")
    torch.backends.cuda.matmul.allow_tf32 = False
    assert tune_backend()["tf32"] is False
    assert torch.backends.cuda.matmul.allow_tf32 is False


@cuda
def test_gpu_mem_fraction_zero_lifts_the_cap(monkeypatch):
    monkeypatch.setenv("GPU_MEM_FRACTION", "0")
    assert tune_backend()["mem_fraction"] is None


@cuda
def test_get_device_applies_tuning():
    torch.backends.cuda.matmul.allow_tf32 = False
    assert get_device().type == "cuda"
    assert torch.backends.cuda.matmul.allow_tf32 is True


def test_malformed_fraction_is_recorded_not_raised(monkeypatch):
    monkeypatch.setenv("GPU_MEM_FRACTION", "not-a-float")
    state = tune_backend()
    if torch.cuda.is_available():
        assert "error" in state
