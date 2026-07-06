import numpy as np
import torch
from PIL import Image

from texup.engine import Upscaler
from texup.project import Project
from texup.wizard import run_remaster


class Fake4x(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=4, mode="nearest")


def fake_factory(name):
    return Upscaler(Fake4x(), scale=4, device="cpu")


def fake_bench(**kw):
    return {"device": "cpu", "rates": {"remacri": 5.0, "realesrgan-x4plus": 5.0,
                                       "normal-rg0-bc1": 5.0}, "measured_at": "t"}


def _game(tmp_path):
    game = tmp_path / "game"
    game.mkdir()
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    Image.fromarray(rgba, "RGBA").save(game / "wall_d.png")
    return game


def test_remaster_folder_mode(tmp_path):
    game = _game(tmp_path)
    out = tmp_path / "out"
    answers = iter(["detailed", "folder"])
    code = run_remaster(game, out, ask=lambda q, c, d: next(answers),
                        engine_factory=fake_factory, bench_runner=fake_bench)
    assert code == 0
    assert (out / "wall_d.png").exists()
    prj = Project.load(out)
    assert prj.wizard == {"preset": "detailed", "apply_mode": "folder"}
    assert not prj.records(status="pending")
    # игра не тронута
    assert (game / "wall_d.png").stat().st_size < 500


def test_remaster_game_mode_applies_with_backup(tmp_path):
    game = _game(tmp_path)
    out = tmp_path / "out"
    original = (game / "wall_d.png").read_bytes()
    answers = iter(["detailed", "game"])
    run_remaster(game, out, ask=lambda q, c, d: next(answers),
                 engine_factory=fake_factory, bench_runner=fake_bench)
    assert (game / ".texup-backup" / "wall_d.png").read_bytes() == original
    assert (game / "wall_d.png").read_bytes() != original


def test_second_run_skips_questions(tmp_path):
    game = _game(tmp_path)
    out = tmp_path / "out"
    answers = iter(["detailed", "folder"])
    run_remaster(game, out, ask=lambda q, c, d: next(answers),
                 engine_factory=fake_factory, bench_runner=fake_bench)

    def boom(q, c, d):
        raise AssertionError("questions must not be asked again")
    code = run_remaster(game, out, ask=boom,
                        engine_factory=fake_factory, bench_runner=fake_bench)
    assert code == 0
