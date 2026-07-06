from __future__ import annotations

import io
import zipfile

import pytest

from texup.codecs.base import is_safe_inner_path
from texup.codecs.vpkarc import VpkCodec
from texup.codecs.ziparc import ZipCodec


@pytest.mark.parametrize("bad", [
    "../../etc/passwd.vtf", "/abs/path.vtf", "\\\\unc\\share\\x.vtf",
    "C:/windows/x.vtf", "a/../../b.vtf", "", "/leading.vtf",
])
def test_rejects_unsafe(bad):
    assert not is_safe_inner_path(bad)


@pytest.mark.parametrize("ok", [
    "materials/wall/brick.vtf", "textures/x.png", "a/b/c/d.dds",
    "single.vtf", "with space/tex.vtf",
])
def test_accepts_safe(ok):
    assert is_safe_inner_path(ok)


def _tga_bytes():
    import numpy as np
    from PIL import Image

    rgba = np.full((4, 4, 4), 77, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="TGA")
    return buf.getvalue()


def test_zip_decode_skips_path_traversal_entry(tmp_path):
    p = tmp_path / "evil.pk3"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("../../../tmp/evil.tga", _tga_bytes())
        zf.writestr("ok.tga", _tga_bytes())
    items = ZipCodec().decode(p)
    names = sorted(it.inner_path for it in items)
    assert names == ["ok.tga"]


def test_zip_decode_skips_absolute_entry(tmp_path):
    p = tmp_path / "evil2.pk3"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("/etc/evil.tga", _tga_bytes())
        zf.writestr("ok.tga", _tga_bytes())
    items = ZipCodec().decode(p)
    names = sorted(it.inner_path for it in items)
    assert names == ["ok.tga"]


def test_vpk_decode_filters_unsafe_inner_paths(monkeypatch, tmp_path):
    """The `vpk` library builds archives from a real directory tree, so it
    won't let us stage a literal `../` entry on disk. Instead we verify the
    decode-side guard directly: monkeypatch the opened package to yield an
    unsafe name alongside a safe one, and confirm only the safe entry survives
    (this exercises the exact `is_safe_inner_path` gate wired into
    VpkCodec.decode, not just the helper in isolation)."""
    codec = VpkCodec()

    class FakeEntry:
        def __init__(self, data: bytes):
            self._data = data

        def read(self) -> bytes:
            return self._data

    class FakePkg:
        def __iter__(self):
            return iter(["../../../evil.vtf", "materials/wall/brick.vtf"])

        def get_file(self, inner):
            if inner == "materials/wall/brick.vtf":
                from tests.test_codec_vpk import _vtf_bytes
                return FakeEntry(_vtf_bytes())
            raise AssertionError(f"unsafe entry {inner!r} must never be read")

    monkeypatch.setattr(codec, "_open", lambda path: FakePkg())
    items = codec.decode(tmp_path / "fake_dir.vpk")
    assert [it.inner_path for it in items] == ["materials/wall/brick.vtf"]


def test_pipeline_finalize_rejects_escaping_loose_target(tmp_path, monkeypatch):
    import numpy as np

    from texup.pipeline import _finalize_source

    game_dir = tmp_path / "game"
    game_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    src = game_dir / "pak01_dir.vpk"
    src.write_bytes(b"stub")

    class EvilCodec:
        loose_output = True

        def read_inner(self, path, inner):
            return b"orig"

        def encode_inner(self, inner, orig_bytes, rgba):
            return b"blob"

        def loose_target(self, inner):
            return "../../../escaped.vtf"

    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="unsafe loose target"):
        _finalize_source(EvilCodec(), src, game_dir, out_dir, {"x": rgba}, [])

    assert not (tmp_path / "escaped.vtf").exists()
