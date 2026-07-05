from pathlib import Path

import numpy as np
from PIL import Image

from texup.codecs import find_codec


def _write_png(path: Path) -> np.ndarray:
    rgba = np.random.default_rng(0).integers(0, 255, (8, 6, 4), dtype=np.uint8)
    Image.fromarray(rgba, "RGBA").save(path)
    return rgba


def test_decode_png(tmp_path):
    p = tmp_path / "t.png"
    rgba = _write_png(p)
    codec = find_codec(p)
    assert codec is not None and codec.name == "standard"
    items = codec.decode(p)
    assert len(items) == 1
    item = items[0]
    assert item.pixels.shape == (8, 6, 4)
    assert np.array_equal(item.pixels, rgba)
    assert item.meta["format"] == "PNG"
    assert item.inner_path is None


def test_encode_roundtrip_png(tmp_path):
    p = tmp_path / "t.png"
    _write_png(p)
    codec = find_codec(p)
    new = np.full((16, 12, 4), 200, dtype=np.uint8)
    out = codec.encode_file(p, {"": new})
    (tmp_path / "out.png").write_bytes(out)
    back = np.asarray(Image.open(tmp_path / "out.png").convert("RGBA"))
    assert np.array_equal(back, new)


def test_encode_jpeg_no_alpha(tmp_path):
    p = tmp_path / "t.jpg"
    Image.new("RGB", (6, 6), (10, 20, 30)).save(p)
    codec = find_codec(p)
    items = codec.decode(p)
    assert items[0].pixels.shape == (6, 6, 4)
    out = codec.encode_file(p, {"": items[0].pixels})
    assert out[:2] == b"\xff\xd8"  # JPEG SOI
