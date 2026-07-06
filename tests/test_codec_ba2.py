"""Coverage note (Task 3, v2D): like BSA, `bethesda-structs` exposes no
archive-creation API for BTDX (BA2) either, so fixtures are hand-assembled
byte-for-byte from `bethesda_structs.archive.btdx.BTDXArchive`'s own
construct structs (`header_struct`/`file_struct` for GNRL,
`tex_header_struct`/`tex_chunk_struct` for DX10).

Both sub-types are covered end-to-end and genuinely cross-validated against
the real third-party parser (`test_synthetic_ba2_gnrl_low_level_records_match_library_parser`
and `test_synthetic_ba2_dx10_matches_library_parser`), independent of
`Ba2Codec`:

- GNRL: straightforward — entries are complete files, `.dds` ones decode
  directly, same as BSA.
- DX10: the hard case. `BTDXArchive.iter_files()` (a **third-party,
  independent implementation**) already synthesizes a complete in-memory DDS
  from the raw chunk + index metadata for DX10 archives, so `Ba2Codec` uses
  it directly rather than re-deriving chunk offsets — this doubles as
  cross-validation of our own DX10 DDS-header logic from Task 1, since the
  bytes `DdsCodec.decode_bytes` consumes here were built by someone else's
  code, not ours.

One genuine, verified gap: `BTDXArchive.iter_files()` cannot be used for
GNRL name resolution — its `_iter_gnrl_files` has an off-by-one bug (see
`texup/codecs/ba2.py` module docstring and
`test_gnrl_library_iter_files_has_known_name_truncation_bug` below, which
demonstrates it directly against the untouched library) that truncates the
last character of every filename. `Ba2Codec._iter_entries` works around it
by re-parsing the (differently-shaped, but not reinvented — same
`uint16-length-prefixed` convention the library itself uses correctly for
DX10) name table directly. This is a documented, deliberate deviation from
"just call `iter_files()`", not a re-implementation of the whole format.

What's structurally-supported-but-unverified: BA2 version numbers other
than 1 (only Fallout 4's base version has been exercised), zlib-compressed
DX10 chunks (`packed_size > 0` — the fixtures below use uncompressed chunks
since building a valid zlib stream by hand adds nothing to the DX10 header-
synthesis logic under test), multi-chunk/multi-mip DX10 textures (fixtures
use a single chunk spanning all mips, which is what the format allows for
small textures and is enough to exercise `_build_dds_headers`), and cubemap
DX10 entries (rejected by `DdsCodec._parse`, out of scope per the plan).
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from bethesda_structs.archive import BTDXArchive

from texup.apply import apply_to_game, rollback_game
from texup.codecs import find_codec
from texup.codecs.ba2 import Ba2Codec
from texup.codecs.base import UnsupportedTexture
from texup.codecs.bcn import encode_bcn
from texup.codecs.dds import DdsCodec
from texup.pipeline import process
from texup.scan import scan_game
from tests.test_pipeline import fake_factory

_HEADER_LEN = 24
_GNRL_FILE_STRUCT_LEN = 36
_TEX_HEADER_LEN = 24
_TEX_CHUNK_LEN = 24

_DXGI = {"DXT1": 71, "DXT3": 74, "DXT5": 77, "BC5": 83, "BC7": 98}


def _pascal_2byte(name: str) -> bytes:
    """`uint16 length` + name bytes (no terminator) — the real BA2 name-table
    encoding, matching what `BTDXArchive` correctly parses for DX10 names."""
    raw = name.encode("utf-8")
    return struct.pack("<H", len(raw)) + raw


def _make_ba2_gnrl(entries: list[tuple[str, bytes]]) -> bytes:
    """Hand-assemble a minimal, valid BTDX/GNRL (BA2) archive from `entries`
    of (inner_path, raw_bytes), matching `BTDXArchive.header_struct` /
    `file_struct` byte-for-byte."""
    cur = _HEADER_LEN + _GNRL_FILE_STRUCT_LEN * len(entries)
    offsets = []
    for _name, data in entries:
        offsets.append(cur)
        cur += len(data)
    names_offset = cur

    file_structs = b""
    file_data = b""
    for (name, data), off in zip(entries, offsets):
        ext = Path(name).suffix.lstrip(".")[:4].encode("ascii").ljust(4, b"\0")
        file_structs += struct.pack("<I4sIIQIII", 0, ext, 0, 0, off, 0, len(data), 0)
        file_data += data

    names_blob = b"".join(_pascal_2byte(n) for n, _ in entries)
    header = struct.pack("<4sI4sIQ", b"BTDX", 1, b"GNRL", len(entries), names_offset)
    return header + file_structs + file_data + names_blob


def _make_ba2_gnrl_with_bad_middle(
    good1: bytes,
    bad_payload: bytes,
    good2: bytes,
    name1: str,
    name2: str,
    name3: str,
) -> bytes:
    """Three-entry GNRL BA2 where the SECOND (middle) entry is stored with
    `packed_size > 0` (claims to be zlib-compressed) but `bad_payload` is
    NOT a valid zlib stream — a realistic malformed record, not a
    monkeypatch. `good1`/`good2` are stored uncompressed and must survive."""
    entries_data = [good1, bad_payload, good2]
    names = [name1, name2, name3]
    unpacked_sizes = [len(good1), len(bad_payload), len(good2)]
    packed_sizes = [0, len(bad_payload), 0]

    cur = _HEADER_LEN + _GNRL_FILE_STRUCT_LEN * 3
    offsets = []
    for d in entries_data:
        offsets.append(cur)
        cur += len(d)
    names_offset = cur

    file_structs = b""
    for name, off, packed, unpacked in zip(names, offsets, packed_sizes, unpacked_sizes):
        ext = Path(name).suffix.lstrip(".")[:4].encode("ascii").ljust(4, b"\0")
        file_structs += struct.pack("<I4sIIQIII", 0, ext, 0, 0, off, packed, unpacked, 0)

    file_data = b"".join(entries_data)
    names_blob = b"".join(_pascal_2byte(n) for n in names)
    header = struct.pack("<4sI4sIQ", b"BTDX", 1, b"GNRL", 3, names_offset)
    return header + file_structs + file_data + names_blob


def _make_ba2_dx10(entries: list[tuple[str, np.ndarray, str]]) -> bytes:
    """Hand-assemble a minimal, valid BTDX/DX10 (BA2 texture) archive from
    `entries` of (inner_path, rgba, fmt). Each texture gets exactly one
    uncompressed chunk spanning its single mip level, matching
    `BTDXArchive.tex_header_struct` / `tex_chunk_struct` byte-for-byte."""
    blobs = [encode_bcn(rgba, fmt) for _name, rgba, fmt in entries]
    n = len(entries)
    record_len = _TEX_HEADER_LEN + _TEX_CHUNK_LEN
    cur = _HEADER_LEN + record_len * n
    data_offsets = []
    for blob in blobs:
        data_offsets.append(cur)
        cur += len(blob)
    names_offset = cur

    tex_records = b""
    for (name, rgba, fmt), blob, off in zip(entries, blobs, data_offsets):
        h, w = rgba.shape[:2]
        tex_header = struct.pack(
            "<I4sIBBHHHBBH", 0, b"dds\0", 0, 0, 1, _TEX_CHUNK_LEN, h, w, 1, _DXGI[fmt], 0
        )
        tex_chunk = struct.pack("<QIIHHI", off, 0, len(blob), 0, 0, 0)
        tex_records += tex_header + tex_chunk

    data_blob = b"".join(blobs)
    names_blob = b"".join(_pascal_2byte(name) for name, _, _ in entries)
    header = struct.pack("<4sI4sIQ", b"BTDX", 1, b"DX10", n, names_offset)
    return header + tex_records + data_blob + names_blob


def _make_ba2_dx10_with_bad_middle(
    entries: list[tuple[str, np.ndarray, str]], corrupt_index: int
) -> bytes:
    """Like `_make_ba2_dx10`, but the chunk of the entry at `corrupt_index`
    claims to be zlib-compressed (`packed_size > 0`) while actually holding
    garbage — a realistic malformed chunk (not a monkeypatch), forcing
    `zlib.decompress` to fail for that entry only."""
    blobs = []
    for i, (_name, rgba, fmt) in enumerate(entries):
        if i == corrupt_index:
            blobs.append(b"\xffnot a valid zlib stream\xff")
        else:
            blobs.append(encode_bcn(rgba, fmt))
    n = len(entries)
    record_len = _TEX_HEADER_LEN + _TEX_CHUNK_LEN
    cur = _HEADER_LEN + record_len * n
    data_offsets = []
    for blob in blobs:
        data_offsets.append(cur)
        cur += len(blob)
    names_offset = cur

    tex_records = b""
    for i, ((name, rgba, fmt), blob, off) in enumerate(zip(entries, blobs, data_offsets)):
        h, w = rgba.shape[:2]
        tex_header = struct.pack(
            "<I4sIBBHHHBBH", 0, b"dds\0", 0, 0, 1, _TEX_CHUNK_LEN, h, w, 1, _DXGI[fmt], 0
        )
        if i == corrupt_index:
            packed_size, unpacked_size = len(blob), 0
        else:
            packed_size, unpacked_size = 0, len(blob)
        tex_chunk = struct.pack("<QIIHHI", off, packed_size, unpacked_size, 0, 0, 0)
        tex_records += tex_header + tex_chunk

    data_blob = b"".join(blobs)
    names_blob = b"".join(_pascal_2byte(name) for name, _, _ in entries)
    header = struct.pack("<4sI4sIQ", b"BTDX", 1, b"DX10", n, names_offset)
    return header + tex_records + data_blob + names_blob


def _dds_bytes(w=16, h=16, color=(200, 120, 60, 255)) -> bytes:
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:] = color
    return DdsCodec().build_dds(rgba, "DXT5", mip_count=1)


def _grad(w=16, h=16):
    x = np.linspace(0, 255, w, dtype=np.uint8)
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., 0] = x[None, :]
    rgba[..., 1] = 128
    rgba[..., 3] = 255
    return rgba


def _make_ba2_gnrl_file(dest_dir: Path, name="Fallout4 - Textures2.ba2") -> Path:
    entries = [
        ("textures/wall/brick_d.dds", _dds_bytes(color=(200, 120, 60, 255))),
        ("textures/wall/wall_d.dds", _dds_bytes(color=(10, 220, 40, 255))),
        ("meshes/sword.nif", b"not a texture"),
    ]
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / name
    out_path.write_bytes(_make_ba2_gnrl(entries))
    return out_path


def _make_ba2_dx10_file(dest_dir: Path, name="Fallout4 - Textures.ba2") -> Path:
    entries = [
        ("textures/armor/cuirass_d.dds", _grad(32, 32), "BC7"),
        ("textures/armor/cuirass_n.dds", _grad(16, 16), "DXT5"),
    ]
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / name
    out_path.write_bytes(_make_ba2_dx10(entries))
    return out_path


# --- cross-validation against the real (untouched) third-party parser ------


def test_synthetic_ba2_gnrl_low_level_records_match_library_parser(tmp_path):
    """Cross-validate the hand-rolled GNRL fixture's low-level records
    (offset/packed_size/unpacked_size) directly against `BTDXArchive` —
    independent of `Ba2Codec`. Names are checked via `Ba2Codec` elsewhere
    because `iter_files()` itself can't be trusted for GNRL names (see
    `test_gnrl_library_iter_files_has_known_name_truncation_bug`)."""
    entries = [
        ("textures/wall/brick_d.dds", _dds_bytes()),
        ("meshes/sword.nif", b"not a texture"),
    ]
    p = tmp_path / "t.ba2"
    p.write_bytes(_make_ba2_gnrl(entries))
    archive = BTDXArchive.parse_file(str(p))
    assert archive.container.header.type == "GNRL"
    assert archive.container.header.file_count == 2
    recs = archive.container.files
    assert recs[0].unpacked_size == len(entries[0][1])
    assert recs[1].unpacked_size == len(entries[1][1])
    assert archive.content[recs[0].offset : recs[0].offset + recs[0].unpacked_size] == entries[0][1]
    assert archive.content[recs[1].offset : recs[1].offset + recs[1].unpacked_size] == entries[1][1]


def test_gnrl_library_iter_files_has_known_name_truncation_bug(tmp_path):
    """Documents the exact upstream bug `Ba2Codec` works around: BTDXArchive's
    own `iter_files()` truncates the last character of every GNRL filename."""
    p = tmp_path / "t.ba2"
    p.write_bytes(_make_ba2_gnrl([("textures/wall/brick_d.dds", b"x")]))
    archive = BTDXArchive.parse_file(str(p))
    (entry,) = list(archive.iter_files())
    assert entry.filepath.as_posix() == "textures/wall/brick_d.dd"  # missing trailing "s"


def test_synthetic_ba2_dx10_matches_library_parser(tmp_path):
    """Direct cross-validation of the hand-rolled DX10 fixture against the
    real third-party parser (not just against Ba2Codec): confirms the
    synthesized DDS bytes carry the right magic/DX10-tag/fourcc, entirely
    independent of our own DdsCodec."""
    p = _make_ba2_dx10_file(tmp_path)
    archive = BTDXArchive.parse_file(str(p))
    assert archive.container.header.type == "DX10"
    found = {f.filepath.as_posix(): f.data for f in archive.iter_files()}
    assert set(found) == {"textures/armor/cuirass_d.dds", "textures/armor/cuirass_n.dds"}
    bc7_dds = found["textures/armor/cuirass_d.dds"]
    assert bc7_dds[:4] == b"DDS "
    assert bc7_dds[84:88] == b"DX10"  # BC7 has no legacy FourCC
    dxt5_dds = found["textures/armor/cuirass_n.dds"]
    assert dxt5_dds[:4] == b"DDS "
    assert dxt5_dds[84:88] == b"DXT5"  # BC3_UNORM keeps the legacy container


# --- detect ------------------------------------------------------------


def test_detect(tmp_path):
    p = _make_ba2_dx10_file(tmp_path)
    codec = find_codec(p)
    assert codec is not None and codec.name == "ba2"

    wrong_name = tmp_path / "notaba2.dat"
    wrong_name.write_bytes(p.read_bytes())
    assert not Ba2Codec().detect(wrong_name)

    fake = tmp_path / "fake.ba2"
    fake.write_bytes(b"\x00\x00\x00\x00 not a ba2")
    assert not Ba2Codec().detect(fake)


# --- decode --------------------------------------------------------------


def test_decode_gnrl_lists_dds_entries_only(tmp_path):
    p = _make_ba2_gnrl_file(tmp_path)
    codec = Ba2Codec()
    items = codec.decode(p)
    names = sorted(it.inner_path for it in items)
    assert names == ["textures/wall/brick_d.dds", "textures/wall/wall_d.dds"]
    for it in items:
        assert it.pixels.shape == (16, 16, 4)
        assert len(it.meta["content_sha"]) == 64
        assert it.codec == "ba2"


def test_decode_dx10_synthesizes_valid_dds_for_both_entries(tmp_path):
    p = _make_ba2_dx10_file(tmp_path)
    codec = Ba2Codec()
    items = {it.inner_path: it for it in codec.decode(p)}
    assert set(items) == {"textures/armor/cuirass_d.dds", "textures/armor/cuirass_n.dds"}

    bc7_item = items["textures/armor/cuirass_d.dds"]
    assert bc7_item.pixels.shape == (32, 32, 4)
    assert bc7_item.meta["format"] == "BC7"
    assert bc7_item.meta["is_dx10"] is True
    assert len(bc7_item.meta["content_sha"]) == 64

    dxt5_item = items["textures/armor/cuirass_n.dds"]
    assert dxt5_item.pixels.shape == (16, 16, 4)
    assert dxt5_item.meta["format"] == "DXT5"
    assert dxt5_item.meta["is_dx10"] is False


def test_decode_skips_path_traversal_entry_gnrl(tmp_path):
    entries = [
        ("textures/wall/brick_d.dds", _dds_bytes()),
        ("../../evil/hack.dds", _dds_bytes()),
    ]
    p = tmp_path / "evil.ba2"
    p.write_bytes(_make_ba2_gnrl(entries))
    items = Ba2Codec().decode(p)
    assert [it.inner_path for it in items] == ["textures/wall/brick_d.dds"]


def test_decode_skips_path_traversal_entry_dx10(tmp_path):
    entries = [
        ("textures/armor/cuirass_d.dds", _grad(16, 16), "BC7"),
        ("../../evil/hack.dds", _grad(16, 16), "BC7"),
    ]
    p = tmp_path / "evil.ba2"
    p.write_bytes(_make_ba2_dx10(entries))
    items = Ba2Codec().decode(p)
    assert [it.inner_path for it in items] == ["textures/armor/cuirass_d.dds"]


def test_read_inner_matches_decode_content_sha(tmp_path):
    p = _make_ba2_dx10_file(tmp_path)
    codec = Ba2Codec()
    items = {it.inner_path: it for it in codec.decode(p)}
    for inner, item in items.items():
        raw = codec.read_inner(p, inner)
        import hashlib

        assert hashlib.sha256(raw).hexdigest() == item.meta["content_sha"]


def test_encode_file_raises(tmp_path):
    p = _make_ba2_dx10_file(tmp_path)
    with pytest.raises(UnsupportedTexture):
        Ba2Codec().encode_file(p, {})


# --- pipeline / apply / rollback ------------------------------------------


def _game_with_ba2(tmp_path):
    game = tmp_path / "game"
    game.mkdir()
    _make_ba2_dx10_file(game)
    _make_ba2_gnrl_file(game)
    return game


def test_pipeline_writes_loose_files(tmp_path):
    game = _game_with_ba2(tmp_path)
    out = tmp_path / "out"
    prj = scan_game(game, out)
    stats = process(prj, out, engine_factory=fake_factory)
    assert stats["done"] == 4  # 2 DX10 textures + 2 GNRL .dds

    cuirass_d = out / "textures" / "armor" / "cuirass_d.dds"
    cuirass_n = out / "textures" / "armor" / "cuirass_n.dds"
    brick = out / "textures" / "wall" / "brick_d.dds"
    assert cuirass_d.is_file() and cuirass_n.is_file() and brick.is_file()

    rgba, meta = DdsCodec().decode_bytes(cuirass_d.read_bytes())
    assert rgba.shape == (128, 128, 4)  # 32x32 source, fake engine upscales x4
    assert meta["format"] == "BC7"

    rgba, meta = DdsCodec().decode_bytes(brick.read_bytes())
    assert rgba.shape == (64, 64, 4)  # 16x16 source, fake engine upscales x4
    assert meta["format"] == "DXT5"

    # No repacked BA2 anywhere under the output tree — container is read-only.
    assert list(out.rglob("*.ba2")) == []


def test_apply_creates_loose_files_and_rollback_deletes(tmp_path):
    game = _game_with_ba2(tmp_path)
    out = tmp_path / "out"
    prj = scan_game(game, out)
    process(prj, out, engine_factory=fake_factory)

    cuirass_target = game / "textures" / "armor" / "cuirass_d.dds"
    brick_target = game / "textures" / "wall" / "brick_d.dds"
    assert not cuirass_target.exists()
    assert not brick_target.exists()

    stats = apply_to_game(out)
    assert stats["applied"] == 4
    assert cuirass_target.is_file()
    assert brick_target.is_file()

    ledger = json.loads((game / ".texup-backup" / "applied.json").read_text())
    assert "textures/armor/cuirass_d.dds" in ledger["created"]
    assert "textures/wall/brick_d.dds" in ledger["created"]

    n = rollback_game(game)
    assert not cuirass_target.exists()
    assert not brick_target.exists()
    assert not (game / "textures").exists()


# --- per-entry resilience -------------------------------------------------


def test_decode_gnrl_resumes_after_malformed_middle_entry(tmp_path):
    """Real per-entry resilience, not a monkeypatched mock: a THREE-entry
    GNRL archive whose SECOND (middle) entry claims to be zlib-compressed
    but holds garbage must still yield entries 1 AND 3. This is the exact
    scenario the generator-exhaustion bug broke: a `next()`-catching loop
    over a shared iterator would have lost entry 3 too, since a Python
    generator that raises out of its body is permanently exhausted."""
    good1 = _dds_bytes(color=(200, 120, 60, 255))
    good2 = _dds_bytes(color=(10, 220, 40, 255))
    bad_payload = b"\xffnot a valid zlib stream\xff"
    p = tmp_path / "mid_corrupt.ba2"
    p.write_bytes(
        _make_ba2_gnrl_with_bad_middle(
            good1,
            bad_payload,
            good2,
            name1="textures/wall/brick_d.dds",
            name2="textures/wall/corrupt_d.dds",
            name3="textures/wall/wall_d.dds",
        )
    )

    items = Ba2Codec().decode(p)
    names = sorted(it.inner_path for it in items)
    assert names == ["textures/wall/brick_d.dds", "textures/wall/wall_d.dds"]


def test_read_inner_gnrl_resumes_after_malformed_middle_entry(tmp_path):
    """Same three-entry GNRL archive: `read_inner` for the entry AFTER the
    malformed middle one must still succeed, and querying the malformed
    entry itself raises KeyError since it was never yielded."""
    good1 = _dds_bytes(color=(200, 120, 60, 255))
    good2 = _dds_bytes(color=(10, 220, 40, 255))
    bad_payload = b"\xffnot a valid zlib stream\xff"
    p = tmp_path / "mid_corrupt.ba2"
    p.write_bytes(
        _make_ba2_gnrl_with_bad_middle(
            good1,
            bad_payload,
            good2,
            name1="textures/wall/brick_d.dds",
            name2="textures/wall/corrupt_d.dds",
            name3="textures/wall/wall_d.dds",
        )
    )
    codec = Ba2Codec()

    data = codec.read_inner(p, "textures/wall/wall_d.dds")
    assert data == good2

    with pytest.raises(KeyError, match="no such entry"):
        codec.read_inner(p, "textures/wall/corrupt_d.dds")


def test_decode_dx10_resumes_after_malformed_middle_entry(tmp_path):
    """DX10's independent per-record chunk extraction gets the same true
    resilience guarantee as GNRL and BSA: a THREE-entry DX10 archive whose
    SECOND (middle) entry's chunk claims to be zlib-compressed but holds
    garbage must still yield entries 1 AND 3."""
    entries = [
        ("textures/armor/cuirass_d.dds", _grad(16, 16), "BC7"),
        ("textures/armor/corrupt_n.dds", _grad(16, 16), "DXT5"),
        ("textures/armor/cuirass_n.dds", _grad(16, 16), "DXT5"),
    ]
    p = tmp_path / "mid_corrupt.ba2"
    p.write_bytes(_make_ba2_dx10_with_bad_middle(entries, corrupt_index=1))

    items = {it.inner_path: it for it in Ba2Codec().decode(p)}
    assert set(items) == {
        "textures/armor/cuirass_d.dds",
        "textures/armor/cuirass_n.dds",
    }
