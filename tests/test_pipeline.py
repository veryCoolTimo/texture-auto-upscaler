import numpy as np
import torch
from PIL import Image

from texup.engine import Upscaler
from texup.pipeline import process
from texup.scan import scan_game


class Fake4x(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=4, mode="nearest")


def fake_factory(model_name: str) -> Upscaler:
    return Upscaler(Fake4x(), scale=4, device="cpu")


def _game(tmp_path, n_diffuse=3):
    game = tmp_path / "game"
    game.mkdir()
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    for i in range(n_diffuse):
        Image.fromarray(rgba, "RGBA").save(game / f"tex{i}_d.png")
    Image.fromarray(rgba, "RGBA").save(game / "wall_n.png")
    return game


def test_process_writes_output_tree(tmp_path):
    game = _game(tmp_path)
    out = tmp_path / "out"
    prj = scan_game(game, out)
    stats = process(prj, out, engine_factory=fake_factory)
    assert stats["done"] == 4
    up = np.asarray(Image.open(out / "tex0_d.png"))
    assert up.shape == (32, 32, 4)
    assert not prj.records(status="pending")


def test_sample_limits_per_class(tmp_path):
    game = _game(tmp_path, n_diffuse=5)
    out = tmp_path / "out"
    prj = scan_game(game, out)
    stats = process(prj, out, sample=2, engine_factory=fake_factory, compare=True)
    assert stats["done"] == 3  # 2 diffuse + 1 normal
    assert len(prj.records(status="pending")) == 3
    compare_files = list((out / "_compare").rglob("*.png"))
    assert len(compare_files) == 3


def test_only_filter(tmp_path):
    game = _game(tmp_path)
    out = tmp_path / "out"
    prj = scan_game(game, out)
    stats = process(prj, out, only=["normal"], engine_factory=fake_factory)
    assert stats["done"] == 1


def test_error_marks_failed_and_continues(tmp_path):
    game = _game(tmp_path)
    out = tmp_path / "out"
    prj = scan_game(game, out)
    # испортить один исходник после скана
    (game / "tex0_d.png").write_bytes(b"garbage now")
    stats = process(prj, out, engine_factory=fake_factory)
    assert stats["failed"] == 1
    assert stats["done"] == 3
