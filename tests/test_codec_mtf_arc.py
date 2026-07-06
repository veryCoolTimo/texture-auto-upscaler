import hashlib
import os
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from texup.codecs.mtframework import (
    ARC_TEXTURE_HASH, MtfArcCodec, TexInfo, build_arc, build_tex, parse_arc,
)

RE5 = os.environ.get("TEXUP_RE5_DIR")


def _make_arc(tmp_path) -> Path:
    rgba = np.random.default_rng(3).integers(0, 255, (8, 8, 4), dtype=np.uint8)
    info = TexInfo(112, 2, 1, 1, 8, 8, 0, "RGBA8", b"\x00" * 16)
    tex = build_tex(info, rgba)
    entries = [
        ("model\\body", ARC_TEXTURE_HASH, tex),
        ("model\\meta", 0x22222222, b"not a texture"),
    ]
    blob = build_arc(7, entries)
    p = tmp_path / "test.arc"
    p.write_bytes(blob)
    return p


def test_arc_decode_finds_textures(tmp_path):
    p = _make_arc(tmp_path)
    codec = MtfArcCodec()
    assert codec.detect(p)
    items = codec.decode(p)
    assert len(items) == 1  # только rTexture
    assert items[0].inner_path == "model\\body"
    assert items[0].pixels.shape == (8, 8, 4)
    # content_sha is over the compressed entry blob, not the raw decompressed tex
    _, entries = parse_arc(p.read_bytes())
    entry = [e for e in entries if e.name == "model\\body"][0]
    data = p.read_bytes()
    comp_blob = data[entry.offset : entry.offset + entry.comp_size]
    assert items[0].meta["content_sha"] == hashlib.sha256(comp_blob).hexdigest()


def test_arc_decode_content_sha_differs_for_different_content(tmp_path):
    rgba_a = np.random.default_rng(10).integers(0, 255, (8, 8, 4), dtype=np.uint8)
    rgba_b = np.random.default_rng(11).integers(0, 255, (8, 8, 4), dtype=np.uint8)
    info = TexInfo(112, 2, 1, 1, 8, 8, 0, "RGBA8", b"\x00" * 16)
    entries = [
        ("a", ARC_TEXTURE_HASH, build_tex(info, rgba_a)),
        ("b", ARC_TEXTURE_HASH, build_tex(info, rgba_b)),
    ]
    blob = build_arc(7, entries)
    p = tmp_path / "two.arc"
    p.write_bytes(blob)
    items = MtfArcCodec().decode(p)
    assert items[0].meta["content_sha"] != items[1].meta["content_sha"]


def test_arc_repack_no_changes_is_identical(tmp_path):
    p = _make_arc(tmp_path)
    codec = MtfArcCodec()
    out = codec.encode_file(p, {})
    assert out == p.read_bytes()


def test_arc_repack_with_replacement(tmp_path):
    p = _make_arc(tmp_path)
    codec = MtfArcCodec()
    new = np.full((16, 16, 4), 42, dtype=np.uint8)
    out = codec.encode_file(p, {"model\\body": new})
    p2 = tmp_path / "new.arc"
    p2.write_bytes(out)
    items = codec.decode(p2)
    assert items[0].pixels.shape == (16, 16, 4)
    assert np.array_equal(items[0].pixels, new)
    # нетронутый энтри остался
    version, entries = parse_arc(out)
    other = [e for e in entries if e.name == "model\\meta"][0]
    assert zlib.decompress(out[other.offset : other.offset + other.comp_size]) == b"not a texture"


def test_arc_data_start_gap(tmp_path):
    entries = [("a", 0x11111111, b"hello")]
    blob = build_arc(7, entries, data_start=1024)
    version, parsed = parse_arc(blob)
    assert parsed[0].offset == 1024
    assert blob[8 + 80 : 1024] == b"\x00" * (1024 - 88)
    assert zlib.decompress(blob[1024 : 1024 + parsed[0].comp_size]) == b"hello"
    # и репак без замен байт-идентичен
    p = tmp_path / "g.arc"
    p.write_bytes(blob)
    from texup.codecs.mtframework import MtfArcCodec
    assert MtfArcCodec().encode_file(p, {}) == blob


def test_arc_repack_with_duplicate_names(tmp_path):
    entries = [
        ("shared\\name", ARC_TEXTURE_HASH, b"texture-payload-AAAA"),
        ("shared\\name", 0x60DD1B16, b"message-payload-BB"),
    ]
    blob = build_arc(7, entries)
    p = tmp_path / "dup.arc"
    p.write_bytes(blob)
    codec = MtfArcCodec()
    assert codec.encode_file(p, {}) == blob


def test_arc_repack_preserves_zero_length_entry(tmp_path):
    entries = [
        ("empty\\entry", 0x33333333, b""),
        ("normal\\entry", ARC_TEXTURE_HASH, b"payload"),
    ]
    blob = build_arc(7, entries)
    p = tmp_path / "z.arc"
    p.write_bytes(blob)
    assert MtfArcCodec().encode_file(p, {}) == blob


@pytest.mark.skipif(not RE5, reason="TEXUP_RE5_DIR not set")
def test_real_arc_repack_identical():
    import glob
    arcs = sorted(glob.glob(os.path.join(RE5, "nativePC_MT", "**", "*.arc"), recursive=True))[:5]
    codec = MtfArcCodec()
    for ap in arcs:
        p = Path(ap)
        assert codec.encode_file(p, {}) == p.read_bytes(), f"repack changed {p.name}"
