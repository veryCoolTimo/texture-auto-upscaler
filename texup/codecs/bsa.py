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
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from bethesda_structs.archive import BSAArchive

from texup.codecs.base import TextureItem, UnsupportedTexture, is_safe_inner_path
from texup.codecs.dds import DdsCodec

_MAGIC = b"BSA\x00"


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
        for entry in archive.iter_files():
            inner = entry.filepath.as_posix()
            if not inner.lower().endswith(".dds"):
                continue
            if not is_safe_inner_path(inner):  # untrusted archive: reject traversal/absolute names
                continue
            try:
                data = entry.data
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
        for entry in archive.iter_files():
            if entry.filepath.as_posix() == inner:
                return entry.data
        raise KeyError(f"no such entry {inner!r} in {path}")

    def loose_target(self, inner: str) -> str:
        return inner

    def encode_inner(self, inner: str, orig_bytes: bytes, rgba: np.ndarray) -> bytes:
        return self._dds.encode_bytes(orig_bytes, rgba)

    def encode_file(self, path: Path, replacements: dict[str, np.ndarray]) -> bytes:
        raise UnsupportedTexture("BSA is read-only; textures are written as loose files")
