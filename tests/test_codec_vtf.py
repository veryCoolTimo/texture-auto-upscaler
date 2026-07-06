import io

import numpy as np
import pytest

from texup.codecs import find_codec
from texup.codecs.base import UnsupportedTexture
from texup.codecs.vtf import VtfCodec


def _rgba(w=16, h=16, seed=0):
    # Smooth gradient, not full random noise: BC1/BC3 are inherently very lossy on
    # per-block-random content (verified ~40+ mean abs error even from a "perfect"
    # encoder), which would make any reasonable fidelity assertion meaningless. A
    # gradient exercises the same decode/encode paths while staying compressible,
    # matching the convention used by the DDS/BCn tests (see test_codec_dds.py).
    x = np.linspace(0, 255, w, dtype=np.uint8)
    y = np.linspace(0, 255, h, dtype=np.uint8)
    r = np.zeros((h, w, 4), dtype=np.uint8)
    r[..., 0] = x[None, :]
    r[..., 1] = y[:, None]
    r[..., 2] = (seed * 40) % 256
    r[..., 3] = 255
    return r


def _srctools_vtf_bytes(rgba, fmt_name="DXT5", version=(7, 4)) -> bytes:
    """Referenc VTF, built by an INDEPENDENT implementation (srctools)."""
    from srctools.vtf import VTF, ImageFormats

    vtf = VTF(
        rgba.shape[1], rgba.shape[0], version=version, fmt=getattr(ImageFormats, fmt_name)
    )
    frame = vtf.get(mipmap=0)
    frame.copy_from(rgba.tobytes(), ImageFormats.RGBA8888)
    buf = io.BytesIO()
    vtf.save(buf)
    return buf.getvalue()


@pytest.mark.parametrize("fmt", ["DXT1", "DXT5", "RGBA8888"])
def test_decode_srctools_reference(tmp_path, fmt):
    rgba = _rgba()
    p = tmp_path / "t.vtf"
    p.write_bytes(_srctools_vtf_bytes(rgba, fmt))
    codec = find_codec(p)
    assert codec is not None and codec.name == "vtf"
    items = codec.decode(p)
    assert len(items) == 1
    it = items[0]
    assert (it.width, it.height) == (16, 16)
    assert len(it.meta["content_sha"]) == 64
    # DXT lossy -> close match; RGBA8888 -> exact match
    diff = np.abs(it.pixels[..., :3].astype(int) - rgba[..., :3].astype(int)).mean()
    assert diff < (1 if fmt == "RGBA8888" else 12)


def _srctools_decode_mip0(data: bytes) -> np.ndarray:
    """Decode mip 0 with the independent (srctools) reader, for cross-validation."""
    from srctools.vtf import VTF

    v = VTF.read(io.BytesIO(data))
    frame = v.get(mipmap=0)
    frame.load()
    return np.frombuffer(bytes(frame._data), dtype=np.uint8).reshape(v.height, v.width, 4)


def test_encode_roundtrip_readable_by_srctools(tmp_path):
    from srctools.vtf import VTF

    rgba = _rgba()
    p = tmp_path / "t.vtf"
    p.write_bytes(_srctools_vtf_bytes(rgba, "DXT5"))
    codec = VtfCodec()
    up = _rgba(32, 32, seed=1)
    out = codec.encode_file(p, {"": up})
    v = VTF.read(io.BytesIO(out))
    assert (v.width, v.height) == (32, 32)
    assert v.format.name == "DXT5"
    dec = _srctools_decode_mip0(out)
    diff = np.abs(dec[..., :3].astype(int) - up[..., :3].astype(int)).mean()
    assert diff < 12


def test_multi_mip_encode_full_chain(tmp_path):
    rgba = _rgba(32, 32)
    p = tmp_path / "t.vtf"
    p.write_bytes(_srctools_vtf_bytes(rgba, "DXT1"))
    codec = VtfCodec()
    items = codec.decode(p)
    orig_mips = items[0].meta["mip_count"]
    up = _rgba(128, 128, seed=2)
    out = codec.encode_file(p, {"": up})
    rgba2, meta2 = codec.decode_bytes(out)
    assert rgba2.shape == (128, 128, 4)
    if orig_mips > 1:
        assert meta2["mip_count"] == 8  # full chain 128 -> 1
    # Cross-validate against the independent (srctools) reader, not just our own decode:
    # this is exactly the sort of multi-mip-with-a-lossy-format file where a mip-chain
    # offset bug (garbage bytes spliced between levels) would otherwise go unnoticed.
    dec = _srctools_decode_mip0(out)
    diff = np.abs(dec[..., :3].astype(int) - up[..., :3].astype(int)).mean()
    assert diff < 12


def test_unsupported_faces_raises(tmp_path):
    codec = VtfCodec()
    data = _srctools_vtf_bytes(_rgba(), "DXT1")
    with pytest.raises(UnsupportedTexture):
        codec.decode_bytes(data[:60])  # truncated header


def test_unsupported_cubemap_raises():
    from srctools.vtf import VTF, ImageFormats, VTFFlags

    vtf = VTF(16, 16, version=(7, 4), fmt=ImageFormats.DXT1, flags=VTFFlags.ENVMAP)
    buf = io.BytesIO()
    vtf.save(buf)
    codec = VtfCodec()
    with pytest.raises(UnsupportedTexture):
        codec.decode_bytes(buf.getvalue())


def test_decode_legacy_version_no_resource_table(tmp_path):
    # srctools.VTF() only constructs 7.2-7.5; 7.2 is the oldest version exercising the
    # pre-7.3 "no resource table, fixed thumbnail-then-mips layout" code path.
    version = (7, 2)
    rgba = _rgba()
    p = tmp_path / "t.vtf"
    p.write_bytes(_srctools_vtf_bytes(rgba, "DXT1", version=version))
    codec = VtfCodec()
    items = codec.decode(p)
    it = items[0]
    assert (it.width, it.height) == (16, 16)
    assert it.meta["vtf_version"] == version
    diff = np.abs(it.pixels[..., :3].astype(int) - rgba[..., :3].astype(int)).mean()
    assert diff < 12


def test_encode_legacy_version_readable_by_srctools(tmp_path):
    # Pre-7.3 layout has an extra fixed pad after the header core/depth fields (see
    # srctools.vtf.VTF.save: "file.write(bytes(15))  # Pad to 80 bytes"), which is not
    # reflected in any field we parse directly -- must come from trusting header_size.
    from srctools.vtf import VTF

    version = (7, 2)
    rgba = _rgba()
    p = tmp_path / "t.vtf"
    p.write_bytes(_srctools_vtf_bytes(rgba, "DXT1", version=version))
    codec = VtfCodec()
    up = _rgba(32, 32, seed=1)
    out = codec.encode_file(p, {"": up})
    v = VTF.read(io.BytesIO(out))
    assert (v.width, v.height) == (32, 32)
    assert v.version == version
    frame = v.get(mipmap=0)
    frame.load()
    dec = np.frombuffer(bytes(frame._data), dtype=np.uint8).reshape(32, 32, 4)
    diff = np.abs(dec[..., :3].astype(int) - up[..., :3].astype(int)).mean()
    assert diff < 12
