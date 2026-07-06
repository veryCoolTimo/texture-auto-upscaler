from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np

from texup.codecs.base import TextureItem
from texup.codecs.dds import DdsCodec
from texup.codecs.standard import StandardCodec

_STD_EXTS = {".png", ".jpg", ".jpeg", ".tga", ".bmp"}
_ZIP_EXTS = {".zip", ".pk3", ".pk4"}


class ZipCodec:
    name = "zip"

    def __init__(self) -> None:
        self._std = StandardCodec()
        self._dds = DdsCodec()

    def detect(self, path: Path) -> bool:
        if path.suffix.lower() not in _ZIP_EXTS:
            return False
        try:
            with open(path, "rb") as f:
                return f.read(4) == b"PK\x03\x04"
        except OSError:
            return False

    def _entry_ext(self, name: str) -> str:
        return "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""

    def decode(self, path: Path) -> list[TextureItem]:
        items: list[TextureItem] = []
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                ext = self._entry_ext(info.filename)
                if ext not in _STD_EXTS and ext != ".dds":
                    continue
                data = zf.read(info)
                try:
                    if ext == ".dds":
                        rgba, meta = self._dds.decode_bytes(data)
                    else:
                        rgba, meta = self._std.decode_bytes(data, ext)
                except Exception:  # noqa: BLE001 — битый энтри пропускаем
                    continue
                meta["zip_entry"] = info.filename
                meta["content_sha"] = hashlib.sha256(data).hexdigest()
                items.append(TextureItem(path, info.filename, self.name, rgba, meta))
        return items

    def encode_file(self, path: Path, replacements: dict[str, np.ndarray]) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(path) as src, zipfile.ZipFile(buf, "w") as dst:
            for info in src.infolist():
                data = src.read(info)
                if info.filename in replacements:
                    ext = self._entry_ext(info.filename)
                    rgba = replacements[info.filename]
                    if ext == ".dds":
                        data = self._dds.encode_bytes(data, rgba)
                    else:
                        data = self._std.encode_bytes(rgba, ext)
                clone = zipfile.ZipInfo(info.filename, date_time=info.date_time)
                clone.compress_type = info.compress_type
                clone.external_attr = info.external_attr
                dst.writestr(clone, data)
        return buf.getvalue()
