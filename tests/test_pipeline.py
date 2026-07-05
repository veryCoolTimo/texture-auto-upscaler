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


def test_dedupe_cache_skips_duplicate_content(tmp_path):
    game = tmp_path / "game"
    game.mkdir()
    rgba = np.random.default_rng(0).integers(0, 255, (8, 8, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    Image.fromarray(rgba, "RGBA").save(game / "tex0_d.png")
    Image.fromarray(rgba, "RGBA").save(game / "tex1_d.png")  # same content, different name
    out = tmp_path / "out"
    prj = scan_game(game, out)

    run_calls = {"n": 0}
    real = fake_factory("remacri")

    class CountingUpscaler:
        def run(self, rgba, max_size=4096):
            run_calls["n"] += 1
            return real.run(rgba, max_size=max_size)

    def counting_factory(model_name: str):
        return CountingUpscaler()

    stats = process(prj, out, engine_factory=counting_factory)
    assert stats["done"] == 2
    assert run_calls["n"] == 1  # only ONE of the two duplicates actually ran inference

    a = np.asarray(Image.open(out / "tex0_d.png"))
    b = np.asarray(Image.open(out / "tex1_d.png"))
    assert a.shape == (32, 32, 4)
    assert np.array_equal(a, b)
    assert (out / "_upcache").is_dir()
    assert len(list((out / "_upcache").glob("*.png"))) == 1


def test_encode_failure_rolls_back_done_records(tmp_path, monkeypatch):
    game = _game(tmp_path, n_diffuse=2)
    out = tmp_path / "out"
    prj = scan_game(game, out)

    from texup.codecs import get_codec
    codec = get_codec("standard")
    monkeypatch.setattr(
        type(codec), "encode_file",
        lambda self, path, repl: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    stats = process(prj, out, engine_factory=fake_factory)
    assert stats["done"] == 0
    assert stats["failed"] == 3
    assert all(r["status"] == "failed" for r in prj.records())
