import numpy as np
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
