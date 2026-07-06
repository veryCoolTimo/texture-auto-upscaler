import io

import numpy as np
from PIL import Image

from texup.codecs.dds import DdsCodec
from texup.codecs.standard import StandardCodec


def _rgba(w=8, h=6):
    rgba = np.random.default_rng(0).integers(0, 255, (h, w, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    return rgba


def test_standard_decode_encode_bytes_png():
    rgba = _rgba()
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    codec = StandardCodec()
    out, meta = codec.decode_bytes(buf.getvalue(), ".png")
    assert np.array_equal(out, rgba)
    assert meta["format"] == "PNG"
    blob = codec.encode_bytes(rgba, ".png")
    out2, _ = codec.decode_bytes(blob, ".png")
    assert np.array_equal(out2, rgba)


def test_standard_encode_bytes_jpeg_drops_alpha():
    codec = StandardCodec()
    blob = codec.encode_bytes(_rgba(), ".jpg")
    assert blob[:2] == b"\xff\xd8"


def test_dds_decode_encode_bytes_roundtrip():
    codec = DdsCodec()
    rgba = _rgba(16, 16)
    dds = codec.build_dds(rgba, "DXT5", mip_count=1)
    out, meta = codec.decode_bytes(dds)
    assert out.shape == (16, 16, 4)
    assert meta["format"] == "DXT5"
    up = _rgba(32, 32)
    blob = codec.encode_bytes(dds, up)
    out2, meta2 = codec.decode_bytes(blob)
    assert out2.shape == (32, 32, 4) and meta2["format"] == "DXT5"


def test_existing_path_api_still_works(tmp_path):
    rgba = _rgba()
    p = tmp_path / "t.png"
    Image.fromarray(rgba, "RGBA").save(p)
    codec = StandardCodec()
    items = codec.decode(p)
    assert np.array_equal(items[0].pixels, rgba)
    assert codec.encode_file(p, {"": rgba})[:8] == b"\x89PNG\r\n\x1a\n"
