import hashlib

import numpy as np
from PIL import Image

from texup.apply import apply_to_game, rollback_game
from texup.pipeline import process
from texup.scan import scan_game
from tests.test_pipeline import fake_factory


def _setup(tmp_path):
    game = tmp_path / "game"
    game.mkdir()
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    Image.fromarray(rgba, "RGBA").save(game / "a_d.png")
    out = tmp_path / "out"
    prj = scan_game(game, out)
    process(prj, out, engine_factory=fake_factory)
    return game, out


def test_apply_and_rollback(tmp_path):
    game, out = _setup(tmp_path)
    original = (game / "a_d.png").read_bytes()

    stats = apply_to_game(out)
    assert stats["applied"] == 1
    assert (game / ".texup-backup" / "a_d.png").read_bytes() == original
    assert (game / "a_d.png").read_bytes() != original

    n = rollback_game(game)
    assert n == 1
    assert (game / "a_d.png").read_bytes() == original


def test_apply_skips_modified_game_file(tmp_path):
    game, out = _setup(tmp_path)
    (game / "a_d.png").write_bytes(b"someone changed me")
    stats = apply_to_game(out)
    assert stats["applied"] == 0 and stats["skipped"] == 1


def test_double_apply_keeps_first_backup(tmp_path):
    game, out = _setup(tmp_path)
    original = (game / "a_d.png").read_bytes()
    apply_to_game(out)
    apply_to_game(out, force=True)  # второй раз
    assert (game / ".texup-backup" / "a_d.png").read_bytes() == original
