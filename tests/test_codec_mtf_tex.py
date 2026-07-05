import os
from pathlib import Path

import numpy as np
import pytest

from texup.codecs.mtframework import MtfTexCodec, TexInfo, build_tex, parse_tex

RE5 = os.environ.get("TEXUP_RE5_DIR")


def _info(w, h, fmt, mips):
    return TexInfo(
        version=112, unk1=2, mip_count=mips, image_count=1,
        width=w, height=h, unk2=0, fmt=fmt, unk_floats=b"\x00\x00\x80?" * 4,
    )


def test_synthetic_roundtrip():
    rgba = np.random.default_rng(1).integers(0, 255, (16, 32, 4), dtype=np.uint8)
    blob = build_tex(_info(32, 16, "DXT5", 5), rgba)
    info = parse_tex(blob)
    assert (info.width, info.height, info.fmt, info.mip_count) == (32, 16, "DXT5", 5)


def test_rgba8_lossless_roundtrip(tmp_path):
    rgba = np.random.default_rng(2).integers(0, 255, (8, 8, 4), dtype=np.uint8)
    blob = build_tex(_info(8, 8, "RGBA8", 1), rgba)
    p = tmp_path / "t.tex"
    p.write_bytes(blob)
    items = MtfTexCodec().decode(p)
    assert np.array_equal(items[0].pixels, rgba)


@pytest.mark.skipif(not RE5, reason="TEXUP_RE5_DIR not set")
def test_real_re5_files_parse():
    codec = MtfTexCodec()
    texes = sorted(Path(RE5).rglob("*.tex"))[:50]
    assert texes, "no .tex files found"
    parsed = skipped = 0
    for p in texes:
        try:
            items = codec.decode(p)
        except Exception as e:
            from texup.codecs.base import UnsupportedTexture
            assert isinstance(e, UnsupportedTexture), f"{p.name}: {e}"
            skipped += 1
            continue
        parsed += 1
        assert items[0].pixels.dtype == np.uint8
    assert parsed > 0
