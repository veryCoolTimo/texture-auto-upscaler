"""Bethesda BA2 (BTDX) container (Fallout 4) — read-only.

BA2 archives split their content into two very different sub-formats,
distinguished by the 4-byte `type` field right after the header:

- ``GNRL`` ("general"): ordinary files stored mostly as-is (optionally
  zlib-compressed). Any `.dds` entry is already a complete, valid DDS file —
  decode it the same way `texup.codecs.bsa.BsaCodec` decodes a BSA entry.
- ``DX10`` ("texture"): every entry *is* a texture, but stored as raw BCn/RGBA
  chunks plus an index record (width, height, mip count, DXGI format code,
  per-chunk offset/size) — there is no DDS header anywhere on disk. A valid
  DDS has to be **synthesized in memory** before `texup.codecs.dds.DdsCodec`
  can touch it.

`bethesda_structs.archive.btdx.BTDXArchive.iter_files` already does this
synthesis for DX10 archives (building `DDS ` + the standard header + a DX10
extended header when the DXGI format needs one, per
`BTDXArchive._build_dds_headers`, then concatenating chunk data) — so for
DX10 we lean on the library instead of re-deriving chunk offsets ourselves.
`read_inner` returns this synthesized blob unchanged, so
`encode_inner = DdsCodec.encode_bytes` works uniformly across GNRL and DX10,
and loose output always writes a complete, standalone `.dds` file.

GNRL name-table bug
--------------------
`BTDXArchive.iter_files()` cannot be used for GNRL archives, though: its
`_iter_gnrl_files` derives each filename via
``PascalString(VarInt, "utf8")`` and then slices off the first character
(``filepath[1:]``). The on-disk GNRL name table is actually the same
``uint16 length + name bytes`` layout the library *correctly* parses for
DX10 archives (`PascalString(Int16ul, "utf8")`, no slicing). Running a
2-byte little-endian length through a 1-byte VarInt silently "succeeds" for
any name shorter than 128 characters (i.e. virtually every real filename)
but reads the wrong window of bytes: the returned string is off by one and,
in particular, is missing the *last* character of the true name (verified
empirically — a hand-built ``brick_d.dds`` entry round-trips through
`BTDXArchive.iter_files()` as ``brick_d.dd``). Left alone, that silently
breaks `.dds` suffix filtering for every GNRL archive. `_iter_gnrl_entries`
below bypasses the buggy convenience method and re-parses the name table
itself directly from `archive.content`, reusing the library's own
already-correct per-file records (`hash`/`ext`/`offset`/`packed_size`/
`unpacked_size`) from `archive.container` — i.e. it doesn't invent any
offsets, it just doesn't let the library's off-by-one string slicing
corrupt the names.

Per-entry resilience — real, not just the fragile ``next()`` pattern
---------------------------------------------------------------------
Python generators are permanently exhausted the moment they raise out of
their body: catching the exception around a subsequent `next()` call does
**not** resume the loop — everything after the failure point is silently
lost. An earlier version of this codec drove iteration with exactly that
broken `next()`-in-a-try pattern, which only ever protected the *first*
entry. Both `_iter_gnrl_entries` and `_iter_dx10_entries` below instead
guard each entry's own extraction *inside* their own loop body, so a bad
record makes the generator `continue` to the next one instead of raising —
true resilience for every record, front to back:

- GNRL: `archive.container.files` is a fully pre-parsed record array (each
  record's offset/packed_size/unpacked_size), so each entry's slice +
  optional zlib-decompress is read independently, no shared iterator state.
- DX10: `archive.container.files` is likewise pre-parsed (header + chunk
  table per entry), so each entry's DDS-header synthesis and chunk
  decompression is likewise independent per record. The one sequential
  dependency between entries is the name table itself (each name is a
  `uint16 length`-prefixed string, so the next name's offset is only known
  once the current one is parsed) — if the name table itself is corrupt,
  we can't recover the offset for the records after it and honestly stop
  there (see the `except Exception` around name parsing, which warns and
  returns instead of pretending to continue); this is orthogonal to, and
  does not affect, per-entry resilience against corrupt pixel/chunk data,
  which is the failure mode both DX10 and GNRL guard against.

Repacking a BA2 in place is out of scope for this codec — replaced textures
are written by the pipeline as loose .dds files next to the container
(`loose_output`/`loose_target`/`encode_inner`/`read_inner`, consumed by
`texup.pipeline._finalize_source`), same as BSA/VPK.
"""
from __future__ import annotations

import hashlib
import struct
import warnings
import zlib
from pathlib import Path, PureWindowsPath
from typing import Iterator

import numpy as np
from bethesda_structs.archive import BTDXArchive
from construct import Compressed, GreedyBytes, Int16ul, PascalString

from texup.codecs.base import TextureItem, UnsupportedTexture, is_safe_inner_path
from texup.codecs.dds import DdsCodec

_MAGIC = b"BTDX"


def _iter_gnrl_entries(archive: BTDXArchive) -> Iterator[tuple[str, bytes]]:
    """Yield (inner_path, data) for a GNRL-type BA2, working around the
    upstream name-table parsing bug documented in this module's docstring.
    Each entry's name-table read and data extraction/decompression is
    guarded independently, so one malformed record doesn't stop iteration
    of the records after it (see "Per-entry resilience" above)."""
    content = archive.content
    container = archive.container
    offset = container.header.names_offset
    for file_container in container.files:
        try:
            length = struct.unpack_from("<H", content, offset)[0]
            name = content[offset + 2 : offset + 2 + length].decode("utf-8")
        except Exception:  # noqa: BLE001 — corrupt name table: can't recover subsequent offsets
            warnings.warn(
                f"BA2 GNRL name table corrupt at offset {offset}; stopping "
                "iteration early, entries after this point are skipped",
                stacklevel=2,
            )
            return
        offset += 2 + length
        try:
            data = content[
                file_container.offset : file_container.offset + file_container.unpacked_size
            ]
            if file_container.packed_size > 0:
                data = zlib.decompress(data)
        except Exception:  # noqa: BLE001 — bad/unsupported record: skip just this one
            continue
        yield PureWindowsPath(name).as_posix(), data


def _iter_dx10_entries(archive: BTDXArchive) -> Iterator[tuple[str, bytes]]:
    """Yield (inner_path, synthesized_dds_bytes) for a DX10-type BA2. Each
    entry's DDS-header synthesis and per-chunk decompression is guarded
    independently (see "Per-entry resilience" above); only a corrupt name
    table (rare, and orthogonal to per-entry chunk-data corruption) forces
    early termination, since subsequent name offsets can't be recovered."""
    content = archive.content
    names_offset = archive.container.header.names_offset
    filename_offset = 0
    for file_container in archive.container.files:
        try:
            name = PascalString(Int16ul, "utf8").parse(
                content[names_offset + filename_offset :]
            )
        except Exception:  # noqa: BLE001 — corrupt name table: can't recover subsequent offsets
            warnings.warn(
                f"BA2 DX10 name table corrupt at offset {filename_offset}; "
                "stopping iteration early, entries after this point are skipped",
                stacklevel=2,
            )
            return
        filename_offset += len(name) + 2

        try:
            built = archive._build_dds_headers(file_container)
            if not built or not built[0]:
                continue
            dds_header, dx10_header = built
            dds_content = b"DDS " + dds_header
            if dx10_header:
                dds_content += dx10_header
            for tex_chunk in file_container.chunks:
                if tex_chunk.packed_size > 0:
                    dds_content += Compressed(GreedyBytes, "zlib").parse(
                        content[tex_chunk.offset : tex_chunk.offset + tex_chunk.packed_size]
                    )
                else:
                    dds_content += content[
                        tex_chunk.offset : tex_chunk.offset + tex_chunk.unpacked_size
                    ]
        except Exception:  # noqa: BLE001 — bad/unsupported record: skip just this one
            continue
        yield PureWindowsPath(name).as_posix(), dds_content


class Ba2Codec:
    name = "ba2"
    loose_output = True

    def __init__(self) -> None:
        self._dds = DdsCodec()

    def detect(self, path: Path) -> bool:
        if path.suffix.lower() != ".ba2":
            return False
        try:
            with open(path, "rb") as f:
                return f.read(4) == _MAGIC
        except OSError:
            return False

    def _open(self, path: Path) -> BTDXArchive:
        return BTDXArchive.parse(path.read_bytes(), filepath=str(path))

    def _iter_entries(self, archive: BTDXArchive) -> Iterator[tuple[str, bytes]]:
        """Yield (inner_path, raw_bytes) uniformly for GNRL and DX10 BA2s.
        Both `_iter_gnrl_entries` and `_iter_dx10_entries` (see module
        docstring) guard each entry's own extraction independently, so a
        malformed record can't stop iteration of the records after it —
        callers here can use a plain `for` loop with no `next()`-catching
        (which would NOT resume a generator that already raised out of its
        body; see module docstring's "Per-entry resilience" section)."""
        if archive.container.header.type == "GNRL":
            yield from _iter_gnrl_entries(archive)
        else:
            yield from _iter_dx10_entries(archive)

    def decode(self, path: Path) -> list[TextureItem]:
        items: list[TextureItem] = []
        archive = self._open(path)
        for inner, data in self._iter_entries(archive):
            if not inner.lower().endswith(".dds"):
                continue
            if not is_safe_inner_path(inner):  # untrusted archive: reject traversal/absolute names
                continue
            try:
                rgba, meta = self._dds.decode_bytes(data)
            except Exception:  # noqa: BLE001 — corrupt/unsupported texture: skip
                continue
            meta["content_sha"] = hashlib.sha256(data).hexdigest()
            items.append(TextureItem(path, inner, self.name, rgba, meta))
        return items

    def read_inner(self, path: Path, inner: str) -> bytes:
        """Re-read one entry's bytes (synthesized DDS for DX10, verbatim DDS
        for GNRL) so `encode_inner` can rebuild the DDS around upscaled
        pixels using the same header `decode()` saw."""
        archive = self._open(path)
        for entry_inner, data in self._iter_entries(archive):
            if entry_inner == inner:
                return data
        raise KeyError(f"no such entry {inner!r} in {path}")

    def loose_target(self, inner: str) -> str:
        return inner

    def encode_inner(self, inner: str, orig_bytes: bytes, rgba: np.ndarray) -> bytes:
        return self._dds.encode_bytes(orig_bytes, rgba)

    def encode_file(self, path: Path, replacements: dict[str, np.ndarray]) -> bytes:
        raise UnsupportedTexture("BA2 is read-only; textures are written as loose files")
