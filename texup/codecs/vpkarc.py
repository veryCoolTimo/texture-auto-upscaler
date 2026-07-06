"""Valve VPK container (Source engine) — read-only.

VPK is a game-directory format: the engine reads assets straight out of the
archive by path, and multi-chunk archives spread payload bytes across sibling
numbered files (`pak01_000.vpk`, ...) that we never touch. Repacking a VPK in
place is out of scope (see the plan's "vpk-repack" non-goal), so this codec
only decodes; replaced textures are written by the pipeline as *loose* .vtf
files next to the container, which Source engines load in preference to the
packed copy. See `loose_output`/`loose_target`/`encode_inner`/`read_inner`,
consumed by `texup.pipeline._finalize_source`.
"""
from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np
import vpk as _vpk

from texup.codecs.base import TextureItem, UnsupportedTexture
from texup.codecs.vtf import VtfCodec

_MAGIC = 0x55AA1234


class VpkCodec:
    name = "vpk"
    loose_output = True

    def __init__(self) -> None:
        self._vtf = VtfCodec()

    def detect(self, path: Path) -> bool:
        if not path.name.lower().endswith("_dir.vpk"):
            return False
        try:
            with open(path, "rb") as f:
                (magic,) = struct.unpack("<I", f.read(4))
            return magic == _MAGIC
        except (OSError, struct.error):
            return False

    def _open(self, path: Path) -> _vpk.VPK:
        return _vpk.VPK(str(path), read_header_only=False)

    def decode(self, path: Path) -> list[TextureItem]:
        items: list[TextureItem] = []
        pkg = self._open(path)
        for inner in pkg:
            if not inner.lower().endswith(".vtf"):
                continue
            try:
                data = pkg.get_file(inner).read()
                rgba, meta = self._vtf.decode_bytes(data)
            except Exception:  # noqa: BLE001 — битые/неподдерживаемые записи пропускаем
                continue
            meta["content_sha"] = hashlib.sha256(data).hexdigest()
            items.append(TextureItem(path, inner, self.name, rgba, meta))
        return items

    def read_inner(self, path: Path, inner: str) -> bytes:
        """Re-read one entry's original bytes; used by the finalize stage to
        rebuild the VTF around the upscaled pixels (encode_inner needs the
        original header/resources, and decode() doesn't keep them around)."""
        pkg = self._open(path)
        return pkg.get_file(inner).read()

    def loose_target(self, inner: str) -> str:
        return inner

    def encode_inner(self, inner: str, orig_bytes: bytes, rgba: np.ndarray) -> bytes:
        return self._vtf.encode_bytes(orig_bytes, rgba)

    def encode_file(self, path: Path, replacements: dict[str, np.ndarray]) -> bytes:
        raise UnsupportedTexture("VPK is read-only; textures are written as loose files")
