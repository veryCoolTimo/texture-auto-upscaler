"""Bethesda BSA container (Skyrim / Skyrim SE) — read-only.

BSA is a game-directory format much like Source's VPK: the engine loads
assets straight out of the archive by path, and Bethesda's runtime prefers a
*loose* file over the packed copy of the same relative path when both exist.
Repacking a BSA in place is out of scope for this codec — instead, replaced
textures are written by the pipeline as loose .dds files next to the
container (mirroring `texup.codecs.vpkarc.VpkCodec`). See
`loose_output`/`loose_target`/`encode_inner`/`read_inner`, consumed by
`texup.pipeline._finalize_source`.

Parsing is delegated to `bethesda_structs.archive.bsa.BSAArchive`, which
supports the BSA versions used by Oblivion (103), Fallout 3/NV/Skyrim (104)
and Skyrim Special Edition (105).

Per-entry resilience — real, not just the fragile ``next()`` pattern
---------------------------------------------------------------------
`BSAArchive.iter_files()` is a single generator that walks the whole archive.
Python generators are permanently exhausted the moment they raise out of
their body: catching the exception around a subsequent `next()` call does
**not** resume the loop — everything after the failure point is silently
lost. An earlier version of this codec drove `iter_files()` with exactly
that broken `next()`-in-a-try pattern, which only ever protected the *first*
entry.

`BSAArchive` parses the entire directory/file record index up front (into
`archive.container`, during `__attrs_post_init__` — before `iter_files()` is
ever called) and only extracts each file's bytes lazily. That means we don't
need the library's generator at all: `_iter_bsa_entries` below walks
`archive.container.directory_blocks`/`file_records` itself (replicating
`BSAArchive.iter_files()`'s own logic, including its file-index bookkeping
and its "sticky" compressed/uncompressed struct selection) and wraps each
entry's own byte-slice-and-decompress step in its own `try/except`. A
failure decoding one entry can't affect any other: the guard lives inside
this generator's own loop body, not in the caller's `next()` call, so the
generator itself never raises — it just continues to the next record. This
gives true per-entry resilience for every record in the archive, since
`archive.container.file_names` is already fully parsed by the time we get
here.
"""
from __future__ import annotations

import hashlib
from pathlib import Path, PureWindowsPath
from typing import Iterator

import numpy as np
from bethesda_structs.archive import BSAArchive

from texup.codecs.base import TextureItem, UnsupportedTexture, is_safe_inner_path
from texup.codecs.dds import DdsCodec

_MAGIC = b"BSA\x00"


def _iter_bsa_entries(archive: BSAArchive) -> Iterator[tuple[str, bytes]]:
    """Yield (inner_path, data) for every file record in `archive`,
    extracting each entry's bytes independently so a malformed record can't
    take down iteration of the records after it (see module docstring).
    Mirrors `BSAArchive.iter_files()`'s own logic byte-for-byte — including
    its file-index bookkeeping across directories and its "sticky"
    compressed/uncompressed struct selection (once a record's compression
    flag mismatches the archive-wide default, the library keeps using the
    compressed struct for every subsequent record too; we replicate that
    exactly so well-formed archives decode identically to before)."""
    file_index = 0
    file_struct = archive.uncompressed_file_struct
    if archive.container.header.archive_flags.files_compressed:
        file_struct = archive.compressed_file_struct

    for directory_block in archive.container.directory_blocks:
        directory_path = PureWindowsPath(directory_block.name[:-1])
        for file_record in directory_block.file_records:
            current_index = file_index
            file_index += 1
            if file_record.size > 0 and (
                archive.container.header.archive_flags.files_compressed
                != bool(file_record.size & BSAArchive.COMPRESSED_MASK)
            ):
                file_struct = archive.compressed_file_struct
            try:
                file_container = file_struct.parse(
                    archive.content[
                        file_record.offset : (
                            file_record.offset + (file_record.size & BSAArchive.SIZE_MASK)
                        )
                    ]
                )
                filepath = directory_path.joinpath(
                    archive.container.file_names[current_index]
                )
                data = file_container.data
            except Exception:  # noqa: BLE001 — bad/unsupported record: skip just this one
                continue
            yield filepath.as_posix(), data


class BsaCodec:
    name = "bsa"
    loose_output = True

    def __init__(self) -> None:
        self._dds = DdsCodec()

    def detect(self, path: Path) -> bool:
        if path.suffix.lower() != ".bsa":
            return False
        try:
            with open(path, "rb") as f:
                return f.read(4) == _MAGIC
        except OSError:
            return False

    def _open(self, path: Path) -> BSAArchive:
        return BSAArchive.parse(path.read_bytes(), filepath=str(path))

    def decode(self, path: Path) -> list[TextureItem]:
        items: list[TextureItem] = []
        archive = self._open(path)
        for inner, data in _iter_bsa_entries(archive):
            if not inner.lower().endswith(".dds"):
                continue
            if not is_safe_inner_path(inner):  # untrusted archive: reject traversal/absolute names
                continue
            try:
                rgba, meta = self._dds.decode_bytes(data)
            except Exception:  # noqa: BLE001 — битые/неподдерживаемые записи пропускаем
                continue
            meta["content_sha"] = hashlib.sha256(data).hexdigest()
            items.append(TextureItem(path, inner, self.name, rgba, meta))
        return items

    def read_inner(self, path: Path, inner: str) -> bytes:
        """Re-read one entry's original bytes; used by the finalize stage to
        rebuild the DDS around the upscaled pixels (encode_inner needs the
        original header, and decode() doesn't keep raw bytes around)."""
        archive = self._open(path)
        for entry_inner, data in _iter_bsa_entries(archive):
            if entry_inner == inner:
                return data
        raise KeyError(f"no such entry {inner!r} in {path}")

    def loose_target(self, inner: str) -> str:
        return inner

    def encode_inner(self, inner: str, orig_bytes: bytes, rgba: np.ndarray) -> bytes:
        return self._dds.encode_bytes(orig_bytes, rgba)

    def encode_file(self, path: Path, replacements: dict[str, np.ndarray]) -> bytes:
        raise UnsupportedTexture("BSA is read-only; textures are written as loose files")
