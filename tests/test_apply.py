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


def test_reapply_after_external_modification_skips(tmp_path):
    game, out = _setup(tmp_path)
    apply_to_game(out)
    (game / "a_d.png").write_bytes(b"modified by another mod")
    stats = apply_to_game(out)
    assert stats["applied"] == 0 and stats["skipped"] >= 1
    assert (game / "a_d.png").read_bytes() == b"modified by another mod"


def test_reapply_own_output_is_allowed(tmp_path):
    game, out = _setup(tmp_path)
    original = (game / "a_d.png").read_bytes()
    apply_to_game(out)
    stats = apply_to_game(out)  # файл в игре == наш применённый → можно
    assert stats["applied"] == 1
    assert (game / ".texup-backup" / "a_d.png").read_bytes() == original


def test_rescan_bak_never_copied_to_game(tmp_path):
    game, out = _setup(tmp_path)
    (out / "texup-project.json.bak").write_text("{}")
    apply_to_game(out)
    assert not (game / "texup-project.json.bak").exists()


def test_rollback_recreates_missing_dirs(tmp_path):
    game, out = _setup(tmp_path)
    sub = game / "textures"
    sub.mkdir()
    import shutil as _sh
    _sh.copy2(game / "a_d.png", sub / "b_d.png")
    # пересканируем и применим, затем удалим папку и откатим
    from texup.scan import scan_game
    from texup.pipeline import process
    out2 = tmp_path / "out2"
    prj = scan_game(game, out2)
    from tests.test_pipeline import fake_factory
    process(prj, out2, engine_factory=fake_factory)
    apply_to_game(out2)
    _sh.rmtree(sub)
    n = rollback_game(game)
    assert (sub / "b_d.png").exists()
