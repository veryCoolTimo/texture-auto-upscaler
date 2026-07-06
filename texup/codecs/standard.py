from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image

from texup.codecs.base import TextureItem, UnsupportedTexture

_EXTS = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".tga": "TGA", ".bmp": "BMP"}


class StandardCodec:
    name = "standard"

    def detect(self, path: Path) -> bool:
        return path.suffix.lower() in _EXTS

    def decode_bytes(self, data: bytes, ext: str) -> tuple[np.ndarray, dict]:
        with Image.open(io.BytesIO(data)) as img:
            fmt = img.format or _EXTS[ext.lower()]
            rgba = np.asarray(img.convert("RGBA"))
        return rgba, {"format": fmt}

    def encode_bytes(self, rgba: np.ndarray, ext: str) -> bytes:
        fmt = _EXTS[ext.lower()]
        img = Image.fromarray(rgba, "RGBA")
        if fmt == "BMP":
            if (rgba[..., 3] != 255).any():
                raise UnsupportedTexture("BMP encode would lose alpha channel")
            img = img.convert("RGB")
        elif fmt == "JPEG":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format=fmt, quality=95)
        return buf.getvalue()

    def decode(self, path: Path) -> list[TextureItem]:
        data = path.read_bytes()
        rgba, meta = self.decode_bytes(data, path.suffix)
        content_sha = hashlib.sha256(data).hexdigest()
        meta["content_sha"] = content_sha
        return [TextureItem(path, None, self.name, rgba, meta)]

    def encode_file(self, path: Path, replacements: dict[str, np.ndarray]) -> bytes:
        return self.encode_bytes(replacements[""], path.suffix)
