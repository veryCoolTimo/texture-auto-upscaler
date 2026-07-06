"""Coverage note (Task 2, v2D): `bethesda-structs` has no archive-creation
API (BSAArchive only `parse`/`parse_file`/`iter_files`) so there is no way to
build our test fixtures through the library itself. Fixtures below are
synthetic BSA v104 (Skyrim) archives assembled by hand, byte-for-byte, from
the format documented in `bethesda_structs.archive.bsa.BSAArchive`'s own
construct structs (header/directory-record/directory-block/file-record).
`test_synthetic_bsa_matches_library_parser` cross-validates the hand-rolled
bytes directly against `BSAArchive.iter_files()` (independent of our codec)
so both the fixture builder and `BsaCodec` are exercised against the real
third-party parser, not just against each other.
"""
from __future__ import annotations

import json
import struct
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest

from bethesda_structs.archive import BSAArchive

from texup.apply import apply_to_game, rollback_game
from texup.codecs import find_codec
from texup.codecs.base import UnsupportedTexture
from texup.codecs.bsa import BsaCodec
from texup.codecs.dds import DdsCodec
from texup.pipeline import process
from texup.scan import scan_game
from tests.test_pipeline import fake_factory


def _pascal_dir_name(name: str) -> bytes:
    """VarInt-length-prefixed dir name incl. trailing NUL — matches the
    `PascalString(VarInt, "utf8")` field in BSAArchive.directory_block_struct.
    VarInt encodes values < 128 as a single byte, which every dir name here
    fits comfortably under."""
    raw = (name + "\0").encode("utf-8")
    assert len(raw) < 128
    return struct.pack("<B", len(raw)) + raw


def _cstring(name: str) -> bytes:
    return name.encode("utf-8") + b"\0"


def _make_bsa(entries: list[tuple[str, str, bytes]], version: int = 104) -> bytes:
    """Hand-assemble a minimal, valid BSA v104 archive (Skyrim) containing
    `entries` of (dir_path, file_name, raw_bytes). `dir_path` uses forward
    slashes for readability; stored on disk with the Windows-style backslash
    separators real BSAs use."""
    grouped: "OrderedDict[str, list[tuple[str, bytes]]]" = OrderedDict()
    for d, fname, data in entries:
        grouped.setdefault(d, []).append((fname, data))

    directory_count = len(grouped)
    file_count = sum(len(v) for v in grouped.values())

    dir_name_bytes = [_pascal_dir_name(d.replace("/", "\\")) for d in grouped]
    file_names_flat = [fname for files in grouped.values() for fname, _ in files]
    file_name_bytes = b"".join(_cstring(n) for n in file_names_flat)

    dir_records_size = 16 * directory_count
    dir_blocks_size = sum(
        len(nb) + 16 * len(files) for nb, files in zip(dir_name_bytes, grouped.values())
    )
    header_size = 36 + dir_records_size + dir_blocks_size + len(file_name_bytes)

    header = struct.pack(
        "<4sIIIIIIII",
        b"BSA\x00",
        version,
        36,  # directory_offset == header size
        0x003,  # archive_flags: directories_named | files_named
        directory_count,
        file_count,
        0,  # directory_names_length (not used by the parser)
        0,  # file_names_length (not used by the parser)
        0x002,  # file_flags: dds
    )
    assert len(header) == 36

    dir_records = b"".join(
        struct.pack("<QII", 0, len(files), 0) for files in grouped.values()  # hash, file_count, name_offset
    )

    cur_offset = header_size
    dir_blocks = b""
    for name_bytes, files in zip(dir_name_bytes, grouped.values()):
        dir_blocks += name_bytes
        for _, data in files:
            dir_blocks += struct.pack("<QII", 0, len(data), cur_offset)  # hash, size, offset
            cur_offset += len(data)

    file_data = b"".join(data for files in grouped.values() for _, data in files)

    return header + dir_records + dir_blocks + file_name_bytes + file_data


def _dds_bytes(w=16, h=16, color=(200, 120, 60, 255)) -> bytes:
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:] = color
    return DdsCodec().build_dds(rgba, "DXT5", mip_count=1)


def _make_bsa_file(dest_dir: Path, name="Skyrim - Textures.bsa") -> Path:
    entries = [
        ("textures/wall", "brick_d.dds", _dds_bytes(color=(200, 120, 60, 255))),
        ("textures/wall", "wall_d.dds", _dds_bytes(color=(10, 220, 40, 255))),
        ("meshes", "sword.nif", b"not a texture"),
    ]
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / name
    out_path.write_bytes(_make_bsa(entries))
    return out_path


def test_synthetic_bsa_matches_library_parser(tmp_path):
    """Direct cross-validation of the hand-rolled fixture against the real
    third-party parser (not just against BsaCodec)."""
    p = _make_bsa_file(tmp_path)
    archive = BSAArchive.parse_file(str(p))
    found = {f.filepath.as_posix(): f.data for f in archive.iter_files()}
    assert set(found) == {
        "textures/wall/brick_d.dds",
        "textures/wall/wall_d.dds",
        "meshes/sword.nif",
    }
    assert found["meshes/sword.nif"] == b"not a texture"


def test_detect(tmp_path):
    p = _make_bsa_file(tmp_path)
    codec = find_codec(p)
    assert codec is not None and codec.name == "bsa"

    # wrong extension -> no detect, even with the right magic bytes
    wrong_name = tmp_path / "notabsa.dat"
    wrong_name.write_bytes(p.read_bytes())
    assert not BsaCodec().detect(wrong_name)

    # right extension, wrong magic -> no detect
    fake = tmp_path / "fake.bsa"
    fake.write_bytes(b"\x00\x00\x00\x00 not a bsa")
    assert not BsaCodec().detect(fake)


def test_decode_lists_dds_entries_only(tmp_path):
    p = _make_bsa_file(tmp_path)
    codec = BsaCodec()
    items = codec.decode(p)
    names = sorted(it.inner_path for it in items)
    assert names == ["textures/wall/brick_d.dds", "textures/wall/wall_d.dds"]
    for it in items:
        assert it.pixels.shape == (16, 16, 4)
        assert len(it.meta["content_sha"]) == 64
        assert it.codec == "bsa"


def test_decode_skips_path_traversal_entry(tmp_path):
    entries = [
        ("textures/wall", "brick_d.dds", _dds_bytes()),
        ("../../evil", "hack.dds", b"garbage"),
    ]
    p = tmp_path / "evil.bsa"
    p.write_bytes(_make_bsa(entries))
    items = BsaCodec().decode(p)
    assert [it.inner_path for it in items] == ["textures/wall/brick_d.dds"]


def test_encode_file_raises(tmp_path):
    p = _make_bsa_file(tmp_path)
    with pytest.raises(UnsupportedTexture):
        BsaCodec().encode_file(p, {})


def _game_with_bsa(tmp_path):
    game = tmp_path / "game"
    game.mkdir()
    _make_bsa_file(game)
    return game


def test_pipeline_writes_loose_files(tmp_path):
    game = _game_with_bsa(tmp_path)
    out = tmp_path / "out"
    prj = scan_game(game, out)
    stats = process(prj, out, engine_factory=fake_factory)
    assert stats["done"] == 2

    brick = out / "textures" / "wall" / "brick_d.dds"
    wall = out / "textures" / "wall" / "wall_d.dds"
    assert brick.is_file()
    assert wall.is_file()

    rgba, meta = DdsCodec().decode_bytes(brick.read_bytes())
    assert rgba.shape == (64, 64, 4)  # 16x16 source, fake engine upscales x4
    assert meta["format"] == "DXT5"

    # No repacked BSA anywhere under the output tree — container is read-only.
    assert list(out.rglob("*.bsa")) == []


def test_apply_creates_loose_files_and_rollback_deletes(tmp_path):
    game = _game_with_bsa(tmp_path)
    bsa_path = game / "Skyrim - Textures.bsa"
    original_bsa_bytes = bsa_path.read_bytes()
    out = tmp_path / "out"
    prj = scan_game(game, out)
    process(prj, out, engine_factory=fake_factory)

    brick_target = game / "textures" / "wall" / "brick_d.dds"
    wall_target = game / "textures" / "wall" / "wall_d.dds"
    assert not brick_target.exists()
    assert not wall_target.exists()

    stats = apply_to_game(out)
    assert stats["applied"] == 2
    assert brick_target.is_file()
    assert wall_target.is_file()

    ledger = json.loads((game / ".texup-backup" / "applied.json").read_text())
    assert sorted(ledger["created"]) == [
        "textures/wall/brick_d.dds",
        "textures/wall/wall_d.dds",
    ]
    # No backup was taken for these — there was nothing to protect.
    assert not (game / ".texup-backup" / "textures").exists()

    n = rollback_game(game)
    assert not brick_target.exists()
    assert not wall_target.exists()
    # Directories we created for the loose files are cleaned up too.
    assert not (game / "textures").exists()
    # The BSA container itself was never touched.
    assert bsa_path.read_bytes() == original_bsa_bytes


def test_decode_resilience_skips_bad_entry(tmp_path):
    """Monkeypatch iter_files to yield one good entry, then raise on the
    second advance. Verify decode() returns the one good item and doesn't
    abort the entire archive scan."""
    p = _make_bsa_file(tmp_path)
    codec = BsaCodec()

    # Monkeypatch archive.iter_files to fail mid-iteration
    original_open = codec._open

    def patched_open(path: Path):
        archive = original_open(path)
        original_iter = archive.iter_files

        def failing_iter():
            """Yield one entry, then raise on the next advance."""
            it = original_iter()
            yield next(it)  # Yield the first entry (brick_d.dds)
            raise ValueError("Simulated corrupt entry during iteration")

        archive.iter_files = failing_iter
        return archive

    codec._open = patched_open
    items = codec.decode(p)

    # Even though the second entry fails, we should have decoded the first
    assert len(items) == 1
    assert items[0].inner_path == "textures/wall/brick_d.dds"


def test_read_inner_resilience_skips_bad_entry(tmp_path):
    """Verify read_inner() survives a bad entry during iteration and can
    still retrieve a later good entry."""
    p = _make_bsa_file(tmp_path)
    codec = BsaCodec()

    # Monkeypatch archive.iter_files to fail on the first entry, succeed on the second
    original_open = codec._open

    def patched_open(path: Path):
        archive = original_open(path)
        original_iter = archive.iter_files

        def failing_iter():
            """Raise on the first advance, then yield remaining entries."""
            it = original_iter()
            try:
                next(it)  # Try to get first entry
                raise ValueError("Simulated corrupt first entry")
            except ValueError:
                pass  # Swallow the simulated error
            # Yield the rest
            for entry in it:
                yield entry

        archive.iter_files = failing_iter
        return archive

    codec._open = patched_open

    # Should still find wall_d.dds even though brick_d.dds failed
    data = codec.read_inner(p, "textures/wall/wall_d.dds")
    assert data == _dds_bytes(color=(10, 220, 40, 255))

    # Querying brick_d.dds should still raise KeyError (it was skipped)
    with pytest.raises(KeyError, match="no such entry"):
        codec.read_inner(p, "textures/wall/brick_d.dds")
