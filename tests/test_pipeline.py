import numpy as np
import torch
from PIL import Image

from texup.codecs.mtframework import ARC_TEXTURE_HASH, MtfArcCodec, TexInfo, build_arc, build_tex
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
    # Only 2 compare files: 1 fresh diffuse + 1 normal
    # (second sampled diffuse is cache hit, so skipped from compare sheets)
    assert len(compare_files) == 2


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
        tile_size = 512

        def run(self, rgba, max_size=4096):
            run_calls["n"] += 1
            return real.run(rgba, max_size=max_size)

        def run_batch(self, rgbas, max_size=4096):
            run_calls["n"] += 1
            return real.run_batch(rgbas, max_size=max_size)

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


def test_encode_failure_then_next_source_still_processes(tmp_path, monkeypatch):
    # With the overlapped pipeline, a finalize (encode) failure for one source
    # must roll back only that source's records while the NEXT source (whose
    # GPU work overlapped with the failing finalize) still completes fine.
    game = _game(tmp_path, n_diffuse=2)
    out = tmp_path / "out"
    prj = scan_game(game, out)

    from texup.codecs import get_codec
    codec = get_codec("standard")
    original_encode_file = type(codec).encode_file
    failing_src = game / "tex0_d.png"

    def flaky_encode_file(self, path, repl):
        if path == failing_src:
            raise RuntimeError("boom")
        return original_encode_file(self, path, repl)

    monkeypatch.setattr(type(codec), "encode_file", flaky_encode_file)
    stats = process(prj, out, engine_factory=fake_factory)

    assert stats["failed"] == 1
    assert stats["done"] == 2
    failed = prj.records(status="failed")
    assert len(failed) == 1
    assert failed[0]["key"] == str(failing_src)
    done_keys = {r["key"] for r in prj.records(status="done")}
    assert done_keys == {str(game / "tex1_d.png"), str(game / "wall_n.png")}
    assert (out / "tex1_d.png").exists()
    assert (out / "wall_n.png").exists()
    assert not (out / "tex0_d.png").exists()


def _tex_info() -> TexInfo:
    return TexInfo(112, 2, 1, 1, 8, 8, 0, "RGBA8", b"\x00" * 16)


def test_batches_same_size_textures_in_one_source(tmp_path):
    # Five distinct-content, same-size textures packed into a single ARC "source"
    # (the real-world shape of a game archive with many small mask/UI textures).
    rng = np.random.default_rng(1)
    entries = []
    for i in range(5):
        rgba = rng.integers(0, 255, (8, 8, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        entries.append((f"tex{i}", ARC_TEXTURE_HASH, build_tex(_tex_info(), rgba)))
    blob = build_arc(7, entries)
    game = tmp_path / "game"
    game.mkdir()
    (game / "pack.arc").write_bytes(blob)
    out = tmp_path / "out"
    prj = scan_game(game, out)

    calls = {"n": 0}

    class CountingFake4x(Fake4x):
        def forward(self, x):
            calls["n"] += 1
            return super().forward(x)

    def counting_factory(model_name: str) -> Upscaler:
        return Upscaler(CountingFake4x(), scale=4, device="cpu")

    stats = process(prj, out, engine_factory=counting_factory)
    assert stats["done"] == 5
    assert calls["n"] < 5  # batched into fewer forward passes than textures

    out_items = MtfArcCodec().decode(out / "pack.arc")
    assert len(out_items) == 5
    for it in out_items:
        assert it.pixels.shape == (32, 32, 4)


def test_batch_post_hook_failure_isolated_from_siblings(tmp_path, monkeypatch):
    base = np.zeros((8, 8, 4), dtype=np.uint8)
    base[..., 2] = 255  # flat "pointing up" normal map (z=255)
    base[..., 3] = 255
    bad = base.copy()
    bad[0, 0, 0] = 123  # sentinel: post-hook raises when it sees this
    other = base.copy()
    other[7, 7, 1] = 5  # distinct content_sha, but no sentinel

    entries = [
        ("tex0_n", ARC_TEXTURE_HASH, build_tex(_tex_info(), base)),
        ("tex1_n", ARC_TEXTURE_HASH, build_tex(_tex_info(), bad)),
        ("tex2_n", ARC_TEXTURE_HASH, build_tex(_tex_info(), other)),
    ]
    blob = build_arc(7, entries)
    game = tmp_path / "game"
    game.mkdir()
    (game / "pack.arc").write_bytes(blob)
    out = tmp_path / "out"
    prj = scan_game(game, out)

    import texup.router as router_mod
    original_renormalize = router_mod.renormalize

    def flaky_renormalize(rgba):
        if rgba[0, 0, 0] == 123:
            raise RuntimeError("boom")
        return original_renormalize(rgba)

    monkeypatch.setattr(router_mod, "renormalize", flaky_renormalize)

    stats = process(prj, out, engine_factory=fake_factory)
    assert stats["failed"] == 1
    assert stats["done"] == 2

    failed = prj.records(status="failed")
    assert len(failed) == 1
    assert failed[0]["key"].endswith("tex1_n")


def test_compare_sheets_skip_cache_hits(tmp_path):
    game = tmp_path / "game"
    game.mkdir()
    rgba = np.random.default_rng(0).integers(0, 255, (8, 8, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    Image.fromarray(rgba, "RGBA").save(game / "tex0_d.png")
    Image.fromarray(rgba, "RGBA").save(game / "tex1_d.png")  # same content
    out = tmp_path / "out"
    prj = scan_game(game, out)

    stats = process(prj, out, engine_factory=fake_factory, compare=True, compare_limit=5)
    assert stats["done"] == 2
    compare_files = list((out / "_compare").rglob("*.png"))
    assert len(compare_files) == 1  # only the fresh one, not the cache hit
