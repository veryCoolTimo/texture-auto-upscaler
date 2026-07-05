from __future__ import annotations

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

    def decode(self, path: Path) -> list[TextureItem]:
        with Image.open(path) as img:
            fmt = img.format or _EXTS[path.suffix.lower()]
            rgba = np.asarray(img.convert("RGBA"))
        return [TextureItem(path, None, self.name, rgba, {"format": fmt})]

    def encode_file(self, path: Path, replacements: dict[str, np.ndarray]) -> bytes:
        rgba = replacements[""]
        fmt = _EXTS[path.suffix.lower()]
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
