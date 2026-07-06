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

Repacking a BA2 in place is out of scope for this codec — replaced textures
are written by the pipeline as loose .dds files next to the container
(`loose_output`/`loose_target`/`encode_inner`/`read_inner`, consumed by
`texup.pipeline._finalize_source`), same as BSA/VPK.
"""
from __future__ import annotations

import hashlib
import struct
import zlib
from pathlib import Path, PureWindowsPath
from typing import Iterator

import numpy as np
from bethesda_structs.archive import BTDXArchive

from texup.codecs.base import TextureItem, UnsupportedTexture, is_safe_inner_path
from texup.codecs.dds import DdsCodec

_MAGIC = b"BTDX"


def _iter_gnrl_entries(archive: BTDXArchive) -> Iterator[tuple[str, bytes]]:
    """Yield (inner_path, data) for a GNRL-type BA2, working around the
    upstream name-table parsing bug documented in this module's docstring."""
    content = archive.content
    container = archive.container
    offset = container.header.names_offset
    for file_container in container.files:
        length = struct.unpack_from("<H", content, offset)[0]
        name = content[offset + 2 : offset + 2 + length].decode("utf-8")
        offset += 2 + length
        data = content[
            file_container.offset : file_container.offset + file_container.unpacked_size
        ]
        if file_container.packed_size > 0:
            data = zlib.decompress(data)
        yield PureWindowsPath(name).as_posix(), data


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
        DX10 entries come back from the library's own `iter_files()` already
        synthesized into complete, standalone DDS byte strings; GNRL entries
        go through `_iter_gnrl_entries` (see module docstring)."""
        if archive.container.header.type == "GNRL":
            yield from _iter_gnrl_entries(archive)
        else:
            for entry in archive.iter_files():
                yield entry.filepath.as_posix(), entry.data

    def decode(self, path: Path) -> list[TextureItem]:
        items: list[TextureItem] = []
        archive = self._open(path)
        it = iter(self._iter_entries(archive))
        while True:
            try:
                inner, data = next(it)
            except StopIteration:
                break
            except Exception:  # noqa: BLE001 — bad/unsupported entry: skip, don't drop the archive
                continue
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
        it = iter(self._iter_entries(archive))
        while True:
            try:
                entry_inner, data = next(it)
            except StopIteration:
                break
            except Exception:  # noqa: BLE001 — skip a bad entry, keep looking
                continue
            if entry_inner == inner:
                return data
        raise KeyError(f"no such entry {inner!r} in {path}")

    def loose_target(self, inner: str) -> str:
        return inner

    def encode_inner(self, inner: str, orig_bytes: bytes, rgba: np.ndarray) -> bytes:
        return self._dds.encode_bytes(orig_bytes, rgba)

    def encode_file(self, path: Path, replacements: dict[str, np.ndarray]) -> bytes:
        raise UnsupportedTexture("BA2 is read-only; textures are written as loose files")
