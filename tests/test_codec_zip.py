import io
import zipfile

import numpy as np
from PIL import Image

from texup.codecs import find_codec
from texup.codecs.ziparc import ZipCodec


def _png_bytes(color, w=8, h=8):
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:] = color
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _make_pk3(tmp_path, name="test.pk3"):
    p = tmp_path / name
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("textures/wall.png", _png_bytes((10, 20, 30, 255)), zipfile.ZIP_DEFLATED)
        zf.writestr("textures/glow.tga", _tga_bytes(), zipfile.ZIP_DEFLATED)
        zf.writestr("scripts/map.cfg", b"not a texture", zipfile.ZIP_DEFLATED)
        zf.writestr("stored.png", _png_bytes((1, 2, 3, 255)), zipfile.ZIP_STORED)
    return p


def _tga_bytes():
    rgba = np.full((4, 4, 4), 77, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="TGA")
    return buf.getvalue()


def test_detect_and_decode(tmp_path):
    p = _make_pk3(tmp_path)
    codec = find_codec(p)
    assert codec is not None and codec.name == "zip"
    items = codec.decode(p)
    names = sorted(it.inner_path for it in items)
    assert names == ["stored.png", "textures/glow.tga", "textures/wall.png"]
    wall = next(it for it in items if it.inner_path == "textures/wall.png")
    assert wall.pixels.shape == (8, 8, 4)
    assert wall.meta["format"] == "PNG"
    assert len(wall.meta["content_sha"]) == 64


def test_repack_no_changes_entry_identical(tmp_path):
    p = _make_pk3(tmp_path)
    out = ZipCodec().encode_file(p, {})
    with zipfile.ZipFile(p) as a, zipfile.ZipFile(io.BytesIO(out)) as b:
        assert [i.filename for i in a.infolist()] == [i.filename for i in b.infolist()]
        for name in a.namelist():
            assert a.read(name) == b.read(name)
            assert a.getinfo(name).compress_type == b.getinfo(name).compress_type


def test_repack_with_replacement(tmp_path):
    p = _make_pk3(tmp_path)
    new = np.full((16, 16, 4), 200, dtype=np.uint8)
    out = ZipCodec().encode_file(p, {"textures/wall.png": new})
    with zipfile.ZipFile(io.BytesIO(out)) as zf:
        img = Image.open(io.BytesIO(zf.read("textures/wall.png"))).convert("RGBA")
        assert np.array_equal(np.asarray(img), new)
        assert zf.read("scripts/map.cfg") == b"not a texture"


def test_corrupt_inner_image_skipped(tmp_path):
    p = tmp_path / "bad.pk3"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("ok.png", _png_bytes((5, 5, 5, 255)))
        zf.writestr("broken.png", b"garbage")
    items = ZipCodec().decode(p)
    assert [it.inner_path for it in items] == ["ok.png"]
