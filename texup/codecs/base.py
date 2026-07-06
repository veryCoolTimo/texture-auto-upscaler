from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Protocol, runtime_checkable

import numpy as np


class UnsupportedTexture(Exception):
    """Codec recognizes the file but this variant is not supported."""


def is_safe_inner_path(inner: str) -> bool:
    """True if `inner` is a relative archive-entry path with no traversal,
    no absolute/drive component, and no separator tricks. Container codecs
    (VPK, ZIP, ...) must reject any entry that fails this check before it
    ever reaches the manifest or gets used to build an on-disk path — a
    crafted entry like `../../../foo.vtf` would otherwise let an untrusted
    archive write outside the intended output directory."""
    if not inner:
        return False
    # normalize windows separators archives sometimes use
    norm = inner.replace("\\", "/")
    if norm.startswith("/"):
        return False
    # reject drive-letter / UNC style (e.g. "C:...")
    if ":" in norm.split("/", 1)[0]:
        return False
    parts = PurePosixPath(norm).parts
    if any(p == ".." for p in parts):
        return False
    return True


@dataclass(eq=False)
class TextureItem:
    source_path: Path
    inner_path: str | None
    codec: str
    pixels: np.ndarray  # (H, W, 4) uint8 RGBA
    meta: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        if self.inner_path is None:
            return str(self.source_path)
        return f"{self.source_path}::{self.inner_path}"

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])


@runtime_checkable
class Codec(Protocol):
    name: str

    def detect(self, path: Path) -> bool: ...

    def decode(self, path: Path) -> list[TextureItem]: ...

    def encode_file(self, path: Path, replacements: dict[str, np.ndarray]) -> bytes: ...

    # --- Optional "loose output" protocol -----------------------------------
    # A container that can't be rewritten in place (e.g. VPK: repacking is out
    # of scope, and multi-chunk archives span sibling files we never touch)
    # sets `loose_output = True` and implements the three methods below instead
    # of a meaningful `encode_file` (which should raise UnsupportedTexture).
    # Call sites use `getattr(codec, "loose_output", False)` so ordinary
    # single-file codecs need no changes and don't have to implement these.
    #
    #   loose_output: bool = False
    #   def loose_target(self, inner: str) -> str: ...          # rel path under the container's dir
    #   def encode_inner(self, inner: str, orig_bytes: bytes, rgba: np.ndarray) -> bytes: ...
    #   def read_inner(self, path: Path, inner: str) -> bytes: ...  # re-read one entry's original bytes


_REGISTRY: dict[str, Codec] = {}


def register(codec: Codec) -> None:
    _REGISTRY[codec.name] = codec


def find_codec(path: Path) -> Codec | None:
    for codec in _REGISTRY.values():
        if codec.detect(path):
            return codec
    return None


def get_codec(name: str) -> Codec:
    return _REGISTRY[name]
