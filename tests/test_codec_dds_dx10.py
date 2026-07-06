import numpy as np
import pytest

from texup.codecs.dds import DdsCodec
from texup.codecs.bcn import bcn_size, decode_bcn, encode_bcn


def _grad(w=16, h=16):
    x = np.linspace(0, 255, w, dtype=np.uint8)
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., 0] = x[None, :]; rgba[..., 1] = 128; rgba[..., 3] = 255
    return rgba


def test_bc7_encode_decode_roundtrip():
    rgba = _grad()
    blob = encode_bcn(rgba, "BC7")
    assert len(blob) == bcn_size(16, 16, "BC7")
    back = decode_bcn(blob, 16, 16, "BC7")
    assert np.abs(back[..., :3].astype(int) - rgba[..., :3].astype(int)).mean() < 10


def test_dx10_bc7_dds_roundtrip():
    codec = DdsCodec()
    rgba = _grad(32, 32)
    dds = codec.build_dds(rgba, "BC7", mip_count=1)  # BC7 -> DX10 header
    assert dds[84:88] == b"DX10"
    import struct
    assert struct.unpack_from("<I", dds, 128)[0] in (98, 99)  # BC7_UNORM(_SRGB)
    out, meta = codec.decode_bytes(dds)
    assert out.shape == (32, 32, 4) and meta["format"] == "BC7"


def test_dx10_dxt5_preserved():
    codec = DdsCodec()
    rgba = _grad()
    # смоделировать DX10-DXT5 (BC3_UNORM=77): построить и проверить что _parse читает
    dds = codec.build_dds(rgba, "DXT5", mip_count=1, force_dx10=True)
    assert dds[84:88] == b"DX10"
    out, meta = codec.decode_bytes(dds)
    assert meta["format"] == "DXT5"
    # энкод обратно сохраняет DX10-контейнер
    blob = codec.encode_bytes(dds, _grad(64, 64))
    assert blob[84:88] == b"DX10"
