import numpy as np
import pytest

from texup.codecs import find_codec
from texup.codecs.bcn import build_mip_chain, decode_bcn, encode_bcn, mip_levels_for


def _gradient(w, h):
    x = np.linspace(0, 255, w, dtype=np.uint8)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = x[None, :]
    rgba[..., 1] = 128
    rgba[..., 3] = 255
    return rgba


@pytest.mark.parametrize("fmt", ["DXT1", "DXT3", "DXT5", "BC5"])
def test_bcn_roundtrip_close(fmt):
    rgba = _gradient(16, 8)
    blob = encode_bcn(rgba, fmt)
    back = decode_bcn(blob, 16, 8, fmt)
    assert back.shape == (8, 16, 4)
    # BCn с потерями: RG-каналы близки
    assert np.abs(back[..., :2].astype(int) - rgba[..., :2].astype(int)).mean() < 12


def test_mip_helpers():
    assert mip_levels_for(16, 8) == 5  # 16,8,4,2,1
    chain = build_mip_chain(_gradient(16, 8), 5)
    assert [m.shape[1] for m in chain] == [16, 8, 4, 2, 1]
    assert [m.shape[0] for m in chain] == [8, 4, 2, 1, 1]


def test_dds_roundtrip(tmp_path):
    from texup.codecs.dds import DdsCodec

    rgba = _gradient(32, 16)
    codec = DdsCodec()
    p = tmp_path / "t.dds"
    p.write_bytes(codec.build_dds(rgba, "DXT5", mip_count=6))
    assert codec.detect(p)
    items = codec.decode(p)
    assert len(items) == 1
    it = items[0]
    assert (it.width, it.height) == (32, 16)
    assert it.meta == {"format": "DXT5", "mip_count": 6}

    new = _gradient(64, 32)
    out = codec.encode_file(p, {"": new})
    p2 = tmp_path / "t2.dds"
    p2.write_bytes(out)
    it2 = codec.decode(p2)[0]
    assert (it2.width, it2.height) == (64, 32)
    assert it2.meta["format"] == "DXT5"
    assert it2.meta["mip_count"] == 7  # полная цепочка для 64x32
