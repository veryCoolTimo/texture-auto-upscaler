import glob
import os
import struct
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from texup.codecs.base import UnsupportedTexture
from texup.codecs.mtframework import (
    ARC_TEXTURE_HASH, MtfArcCodec, MtfTexCodec, Tex2Info, build_arc, build_tex2, parse_arc,
    parse_tex2,
)

RE0 = os.environ.get("TEXUP_RE0_DIR")

TEX2_VERSION = 157  # 0x09D


def _info(w, h, fmt, fmt_id, mips, version=TEX2_VERSION, image_count=1, packed2_upper=1):
    return Tex2Info(
        version=version, mip_count=mips, width=w, height=h, image_count=image_count,
        format_id=fmt_id, packed2_upper=packed2_upper, fmt=fmt,
    )


def test_synthetic_roundtrip_dxt1():
    rgba = np.random.default_rng(1).integers(0, 255, (16, 32, 4), dtype=np.uint8)
    blob = build_tex2(_info(32, 16, "DXT1", 20, 5), rgba)
    info = parse_tex2(blob)
    assert (info.width, info.height, info.fmt, info.mip_count) == (32, 16, "DXT1", 6)
    assert info.format_id == 20


def test_synthetic_roundtrip_bc7():
    rgba = np.random.default_rng(2).integers(0, 255, (16, 16, 4), dtype=np.uint8)
    blob = build_tex2(_info(16, 16, "BC7", 43, 1), rgba)
    info = parse_tex2(blob)
    assert (info.width, info.height, info.fmt, info.format_id) == (16, 16, "BC7", 43)


def test_synthetic_roundtrip_bc5():
    rgba = np.random.default_rng(3).integers(0, 255, (16, 16, 4), dtype=np.uint8)
    blob = build_tex2(_info(16, 16, "BC5", 31, 1), rgba)
    info = parse_tex2(blob)
    assert (info.width, info.height, info.fmt, info.format_id) == (16, 16, "BC5", 31)


def test_rgba8_lossless_roundtrip(tmp_path):
    rgba = np.random.default_rng(4).integers(0, 255, (8, 8, 4), dtype=np.uint8)
    blob = build_tex2(_info(8, 8, "RGBA8", 40, 1), rgba)
    p = tmp_path / "t.tex"
    p.write_bytes(blob)
    items = MtfTexCodec().decode(p)
    assert np.array_equal(items[0].pixels, rgba)


def test_version_flags_preserved_verbatim():
    # RE0 real files carry flag bits above the low 12-bit version number.
    version = (0x60 << 12) | TEX2_VERSION
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    blob = build_tex2(_info(8, 8, "DXT1", 20, 1, version=version), rgba)
    info = parse_tex2(blob)
    assert info.version == version


def test_format_id_and_upper16_preserved_on_rebuild(tmp_path):
    rgba = np.random.default_rng(5).integers(0, 255, (8, 8, 4), dtype=np.uint8)
    blob = build_tex2(_info(8, 8, "DXT1", 25, 1, packed2_upper=0x1234), rgba)
    p = tmp_path / "t.tex"
    p.write_bytes(blob)
    out = MtfTexCodec().encode_file(p, {"": np.zeros((8, 8, 4), dtype=np.uint8)})
    info = parse_tex2(out)
    assert info.format_id == 25
    assert info.packed2_upper == 0x1234


def test_encode_file_upscale_recomputes_mips(tmp_path):
    rgba = np.random.default_rng(6).integers(0, 255, (16, 32, 4), dtype=np.uint8)
    blob = build_tex2(_info(32, 16, "DXT5", 24, 6), rgba)
    p = tmp_path / "t.tex"
    p.write_bytes(blob)
    codec = MtfTexCodec()
    up = np.random.default_rng(7).integers(0, 255, (64, 128, 4), dtype=np.uint8)
    out = parse_tex2(codec.encode_file(p, {"": up}))
    assert (out.width, out.height) == (128, 64)
    assert out.mip_count == 8
    assert out.fmt == "DXT5"


def test_encode_file_single_mip_stays_single(tmp_path):
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    blob = build_tex2(_info(8, 8, "RGBA8", 40, 1), rgba)
    p = tmp_path / "t.tex"
    p.write_bytes(blob)
    out = parse_tex2(MtfTexCodec().encode_file(p, {"": np.zeros((32, 32, 4), dtype=np.uint8)}))
    assert out.mip_count == 1


def test_cubemap_unsupported():
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    blob = build_tex2(_info(8, 8, "DXT1", 20, 1, image_count=6), rgba)
    with pytest.raises(UnsupportedTexture):
        parse_tex2(blob)


def test_unknown_format_id_unsupported():
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    blob = build_tex2(_info(8, 8, "DXT1", 20, 1), rgba)
    # corrupt format_id field (byte at 0xD is format_id, since packed2 = imgs | fmt<<8 | upper<<16)
    corrupted = bytearray(blob)
    packed2, = struct.unpack_from("<I", corrupted, 0xC)
    packed2 = (packed2 & ~0xFF00) | (0xEE << 8)
    struct.pack_into("<I", corrupted, 0xC, packed2)
    with pytest.raises(UnsupportedTexture):
        parse_tex2(bytes(corrupted))


def test_v1_v2_dispatch_by_version(tmp_path):
    # v1 file (version=112) still goes through the old TexInfo/parse_tex path.
    from texup.codecs.mtframework import TexInfo, build_tex

    rgba = np.random.default_rng(8).integers(0, 255, (8, 8, 4), dtype=np.uint8)
    v1_info = TexInfo(112, 2, 1, 1, 8, 8, 0, "RGBA8", b"\x00" * 16)
    v1_blob = build_tex(v1_info, rgba)
    p1 = tmp_path / "v1.tex"
    p1.write_bytes(v1_blob)
    assert np.array_equal(MtfTexCodec().decode(p1)[0].pixels, rgba)

    v2_blob = build_tex2(_info(8, 8, "RGBA8", 40, 1), rgba)
    p2 = tmp_path / "v2.tex"
    p2.write_bytes(v2_blob)
    assert np.array_equal(MtfTexCodec().decode(p2)[0].pixels, rgba)


def test_arc_dispatch_v2(tmp_path):
    rgba = np.random.default_rng(9).integers(0, 255, (8, 8, 4), dtype=np.uint8)
    tex = build_tex2(_info(8, 8, "DXT1", 20, 1), rgba)
    entries = [("model\\v2body", ARC_TEXTURE_HASH, tex)]
    blob = build_arc(7, entries)
    p = tmp_path / "v2.arc"
    p.write_bytes(blob)
    codec = MtfArcCodec()
    items = codec.decode(p)
    assert len(items) == 1
    assert items[0].meta["format"] == "DXT1"

    new = np.full((16, 16, 4), 7, dtype=np.uint8)
    out = codec.encode_file(p, {"model\\v2body": new})
    p2 = tmp_path / "v2new.arc"
    p2.write_bytes(out)
    items2 = codec.decode(p2)
    assert items2[0].pixels.shape == (16, 16, 4)


@pytest.mark.skipif(not RE0, reason="TEXUP_RE0_DIR not set")
def test_real_re0_sweep():
    """Full sweep over every loose .tex and every rTexture entry in every .arc under
    TEXUP_RE0_DIR: parse must either succeed with an exact payload-size match, or raise
    UnsupportedTexture. Never silently produce a wrong size."""
    ARC_MAGIC = 0x00435241

    def iter_blobs():
        for p in sorted(glob.glob(os.path.join(RE0, "**", "*.tex"), recursive=True)):
            data = Path(p).read_bytes()
            if data[:4] == b"TEX\x00":
                yield p, data
        for ap in sorted(glob.glob(os.path.join(RE0, "**", "*.arc"), recursive=True)):
            data = Path(ap).read_bytes()
            if len(data) < 8:
                continue
            magic, = struct.unpack_from("<I", data, 0)
            if magic != ARC_MAGIC:
                continue
            _, entries = parse_arc(data)
            for e in entries:
                if e.type_hash != ARC_TEXTURE_HASH:
                    continue
                comp = data[e.offset : e.offset + e.comp_size]
                try:
                    raw = zlib.decompress(comp)
                except Exception:
                    continue
                if raw[:4] == b"TEX\x00":
                    yield f"{ap}::{e.name}", raw

    total = parsed = skipped = 0
    size_mismatch = []
    fmt_counts = Counter()
    for name, data in iter_blobs():
        total += 1
        try:
            info = parse_tex2(data)
        except UnsupportedTexture:
            skipped += 1
            continue
        parsed += 1
        fmt_counts[(info.format_id, info.fmt)] += 1
        offs = struct.unpack_from(f"<{info.mip_count}I", data, 0x10)
        from texup.codecs.bcn import bcn_size

        expected_end = offs[-1] + bcn_size(
            max(1, info.width >> (info.mip_count - 1)), max(1, info.height >> (info.mip_count - 1)), info.fmt
        )
        if expected_end != len(data):
            size_mismatch.append((name, expected_end, len(data)))

    print(f"\nRE0 tex2 sweep: total={total} parsed={parsed} skipped={skipped} "
          f"size_mismatch={len(size_mismatch)}")
    for k, v in sorted(fmt_counts.items()):
        print(f"  format_id={k[0]:>3} fmt={k[1]:<6} count={v}")
    for name, exp, act in size_mismatch[:10]:
        print(f"  MISMATCH {name}: expected_end={exp} actual={act}")

    assert total > 0
    assert not size_mismatch
    assert parsed / total > 0.95


@pytest.mark.skipif(not RE0, reason="TEXUP_RE0_DIR not set")
def test_real_re0_roundtrip_sample():
    """decode -> build_tex2 with the same pixels -> parse again: dims/fmt/mip_count must
    stay consistent, for one real sample per format id."""
    from texup.codecs.mtframework import build_tex2, tex2_pixels

    seen_fmt_ids: set[int] = set()
    checked = 0
    texes = sorted(glob.glob(os.path.join(RE0, "**", "*.tex"), recursive=True))
    for p in texes:
        data = Path(p).read_bytes()
        if data[:4] != b"TEX\x00":
            continue
        try:
            info = parse_tex2(data)
        except UnsupportedTexture:
            continue
        if info.format_id in seen_fmt_ids:
            continue
        seen_fmt_ids.add(info.format_id)
        rgba = tex2_pixels(data, info)
        rebuilt = build_tex2(info, rgba)
        info2 = parse_tex2(rebuilt)
        assert (info2.width, info2.height, info2.fmt) == (info.width, info.height, info.fmt)
        assert info2.mip_count == (info.mip_count if info.mip_count > 1 else 1)
        assert info2.format_id == info.format_id
        checked += 1
        if len(seen_fmt_ids) >= 8:
            break
    print(f"\nRE0 roundtrip sample: checked {checked} format ids: {sorted(seen_fmt_ids)}")
    assert checked >= 5
