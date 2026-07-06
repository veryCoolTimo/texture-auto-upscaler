from __future__ import annotations

import io
import json
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest
import vpk as _vpk

from texup.apply import apply_to_game, rollback_game
from texup.codecs import find_codec
from texup.codecs.base import UnsupportedTexture
from texup.codecs.vpkarc import VpkCodec
from texup.codecs.vtf import VtfCodec
from texup.pipeline import process
from texup.scan import scan_game
from tests.test_pipeline import fake_factory


def _vtf_bytes(w=16, h=16, color=(200, 120, 60, 255)) -> bytes:
    from srctools.vtf import VTF, ImageFormats

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:] = color
    v = VTF(w, h, version=(7, 4), fmt=ImageFormats.DXT5)
    frame = v.get(mipmap=0)
    frame.copy_from(rgba.tobytes(), ImageFormats.RGBA8888)
    buf = io.BytesIO()
    v.save(buf)
    return buf.getvalue()


def _make_vpk(dest_dir: Path, name="pak01_dir.vpk") -> Path:
    """Build a real VPK (via the `vpk` library) containing 2 VTFs + 1
    non-texture entry, and return the path to the resulting `_dir.vpk`.

    The staging tree used to build the archive lives in a throwaway system
    temp dir (NOT under `dest_dir`) so it never gets picked up a second time
    as loose .vtf files if `dest_dir` is later scanned/rglob'd.
    """
    src = Path(tempfile.mkdtemp(prefix="texup-vpk-src-"))
    (src / "materials" / "wall").mkdir(parents=True)
    (src / "scripts").mkdir()
    (src / "materials" / "wall" / "brick_d.vtf").write_bytes(_vtf_bytes(color=(200, 120, 60, 255)))
    (src / "materials" / "wall" / "wall_d.vtf").write_bytes(_vtf_bytes(color=(10, 220, 40, 255)))
    (src / "scripts" / "map.cfg").write_bytes(b"not a texture")

    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / name
    _vpk.new(str(src)).save(str(out_path))
    return out_path


def test_detect(tmp_path):
    p = _make_vpk(tmp_path)
    codec = find_codec(p)
    assert codec is not None and codec.name == "vpk"

    # wrong name suffix -> no detect, even with the right magic bytes
    wrong_name = tmp_path / "pak01.vpk"
    wrong_name.write_bytes(p.read_bytes())
    assert not VpkCodec().detect(wrong_name)

    # right name, wrong magic -> no detect
    fake = tmp_path / "fake_dir.vpk"
    fake.write_bytes(b"\x00\x00\x00\x00 not a vpk")
    assert not VpkCodec().detect(fake)


def test_decode_lists_vtf_entries_only(tmp_path):
    p = _make_vpk(tmp_path)
    codec = VpkCodec()
    items = codec.decode(p)
    names = sorted(it.inner_path for it in items)
    assert names == ["materials/wall/brick_d.vtf", "materials/wall/wall_d.vtf"]
    for it in items:
        assert it.pixels.shape == (16, 16, 4)
        assert len(it.meta["content_sha"]) == 64
        assert it.codec == "vpk"


def test_encode_file_raises(tmp_path):
    p = _make_vpk(tmp_path)
    with pytest.raises(UnsupportedTexture):
        VpkCodec().encode_file(p, {})


def _game_with_vpk(tmp_path):
    game = tmp_path / "game"
    game.mkdir()
    _make_vpk(game)
    return game


def test_pipeline_writes_loose_files(tmp_path):
    game = _game_with_vpk(tmp_path)
    out = tmp_path / "out"
    prj = scan_game(game, out)
    stats = process(prj, out, engine_factory=fake_factory)
    assert stats["done"] == 2

    brick = out / "materials" / "wall" / "brick_d.vtf"
    wall = out / "materials" / "wall" / "wall_d.vtf"
    assert brick.is_file()
    assert wall.is_file()

    rgba, meta = VtfCodec().decode_bytes(brick.read_bytes())
    assert rgba.shape == (64, 64, 4)  # 16x16 source, fake engine upscales x4

    # No repacked VPK anywhere under the output tree — container is read-only.
    assert list(out.rglob("*.vpk")) == []


def test_apply_creates_loose_files_and_rollback_deletes(tmp_path):
    game = _game_with_vpk(tmp_path)
    vpk_path = game / "pak01_dir.vpk"
    original_vpk_bytes = vpk_path.read_bytes()
    out = tmp_path / "out"
    prj = scan_game(game, out)
    process(prj, out, engine_factory=fake_factory)

    brick_target = game / "materials" / "wall" / "brick_d.vtf"
    wall_target = game / "materials" / "wall" / "wall_d.vtf"
    assert not brick_target.exists()
    assert not wall_target.exists()

    stats = apply_to_game(out)
    assert stats["applied"] == 2
    assert brick_target.is_file()
    assert wall_target.is_file()

    ledger = json.loads((game / ".texup-backup" / "applied.json").read_text())
    assert sorted(ledger["created"]) == [
        "materials/wall/brick_d.vtf",
        "materials/wall/wall_d.vtf",
    ]
    # No backup was taken for these — there was nothing to protect.
    assert not (game / ".texup-backup" / "materials").exists()

    n = rollback_game(game)
    assert not brick_target.exists()
    assert not wall_target.exists()
    # Directories we created for the loose files are cleaned up too.
    assert not (game / "materials").exists()
    # The VPK container itself was never touched.
    assert vpk_path.read_bytes() == original_vpk_bytes
