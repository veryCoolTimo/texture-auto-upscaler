import json

import numpy as np
import torch

from texup import bench
from texup.engine import Upscaler


class Fake4x(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=4, mode="nearest")


def fake_factory(model_name: str) -> Upscaler:
    return Upscaler(Fake4x(), scale=4, device="cpu")


def test_run_bench_writes_rates(tmp_path):
    data = bench.run_bench(engine_factory=fake_factory,
                           models=["remacri"], cache_dir=tmp_path)
    assert data["rates"]["remacri"] > 0
    assert data["device"] in ("mps", "cuda", "cpu")
    on_disk = json.loads((tmp_path / "bench.json").read_text())
    assert on_disk == data


def test_load_bench_missing_returns_none(tmp_path):
    assert bench.load_bench(cache_dir=tmp_path) is None


def test_default_models_cover_presets():
    ms = bench.default_models()
    assert "remacri" in ms and "realesrgan-x4plus" in ms and "normal-rg0-bc1" in ms
