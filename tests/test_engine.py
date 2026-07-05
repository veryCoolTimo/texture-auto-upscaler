import numpy as np
import pytest
import torch

from texup.engine import Upscaler, pick_device


class Fake4x(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=4, mode="nearest")


def _rgba(w, h):
    rng = np.random.default_rng(0)
    px = rng.integers(0, 255, (h, w, 4), dtype=np.uint8)
    px[..., 3] = 255
    return px


def test_basic_upscale_4x():
    up = Upscaler(Fake4x(), scale=4, device="cpu")
    out = up.run(_rgba(32, 24))
    assert out.shape == (96, 128, 4)
    assert out.dtype == np.uint8


def test_alpha_channel_preserved():
    up = Upscaler(Fake4x(), scale=4, device="cpu")
    px = _rgba(16, 16)
    px[..., 3] = 77
    out = up.run(px)
    assert np.all(out[..., 3] == 77)


def test_varying_alpha_uses_lanczos_not_second_neural_pass(monkeypatch):
    up = Upscaler(Fake4x(), scale=4, device="cpu")
    calls = {"n": 0}
    orig_run_rgb = up._run_rgb

    def counting(rgb):
        calls["n"] += 1
        return orig_run_rgb(rgb)

    monkeypatch.setattr(up, "_run_rgb", counting)
    px = _rgba(16, 16)
    px[..., 3] = np.linspace(0, 255, 256).reshape(16, 16).astype(np.uint8)
    out = up.run(px)
    assert calls["n"] == 1  # RGB pass only, alpha handled by Lanczos
    assert out.shape == (64, 64, 4)
    # gradient direction preserved by the resize
    assert out[0, 0, 3] < out[-1, -1, 3]


def test_tiling_matches_whole_image():
    up_whole = Upscaler(Fake4x(), scale=4, device="cpu")
    up_tiled = Upscaler(Fake4x(), scale=4, device="cpu")
    up_tiled.tile_size = 16
    px = _rgba(64, 48)
    assert np.array_equal(up_whole.run(px), up_tiled.run(px))


def test_max_size_downscale():
    up = Upscaler(Fake4x(), scale=4, device="cpu")
    out = up.run(_rgba(64, 64), max_size=128)
    assert out.shape == (128, 128, 4)


def test_pick_device_returns_valid():
    assert pick_device() in ("mps", "cuda", "cpu")


def test_cpu_device_stays_fp32():
    up = Upscaler(Fake4x(), scale=4, device="cpu")
    assert up.use_fp16 is False
    out = up.run(_rgba(16, 16))
    assert out.dtype == np.uint8
    assert out.shape == (64, 64, 4)


class _NanOnFp16(torch.nn.Module):
    """Simulates a model whose fp16 forward pass degenerates to NaN on this hardware."""

    def forward(self, x):
        if x.dtype == torch.float16:
            return torch.full(
                (x.shape[0], 3, x.shape[2] * 4, x.shape[3] * 4), float("nan"), dtype=x.dtype
            )
        return torch.nn.functional.interpolate(x, scale_factor=4, mode="nearest")


class _ZeroOnFp16(torch.nn.Module):
    """Simulates a model whose fp16 forward pass silently degenerates to all-black output."""

    def forward(self, x):
        if x.dtype == torch.float16:
            return torch.zeros((x.shape[0], 3, x.shape[2] * 4, x.shape[3] * 4), dtype=x.dtype)
        return torch.nn.functional.interpolate(x, scale_factor=4, mode="nearest")


def test_fp16_nan_output_triggers_permanent_fp32_fallback():
    up = Upscaler(_NanOnFp16(), scale=4, device="cpu")
    up.use_fp16 = True  # force fp16 path (real detection only kicks in on device=="mps")
    out = up.run(_rgba(16, 16))
    assert up.use_fp16 is False  # permanently switched back
    assert not np.isnan(out.astype(np.float32)).any()
    assert out.dtype == np.uint8


def test_fp16_black_output_triggers_permanent_fp32_fallback():
    up = Upscaler(_ZeroOnFp16(), scale=4, device="cpu")
    up.use_fp16 = True
    px = _rgba(16, 16)  # non-zero input
    out = up.run(px)
    assert up.use_fp16 is False
    assert out[..., :3].sum() > 0  # not degenerate black


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
def test_mps_fp16_smoke_no_nan():
    up = Upscaler(Fake4x(), scale=4, device="mps")
    assert up.use_fp16 is True
    out = up.run(_rgba(16, 16))
    assert out.dtype == np.uint8
    assert not np.isnan(out.astype(np.float32)).any()
