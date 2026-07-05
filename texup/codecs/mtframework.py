from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from texup.codecs.base import TextureItem, UnsupportedTexture
from texup.codecs.bcn import bcn_size, build_mip_chain, decode_bcn, encode_bcn, mip_levels_for

TEX_MAGIC = b"TEX\x00"
_FMT_FROM_U32 = {0x31545844: "DXT1", 0x33545844: "DXT3", 0x35545844: "DXT5", 21: "RGBA8"}
_FMT_TO_U32 = {"DXT1": 0x31545844, "DXT3": 0x35545844, "DXT5": 0x35545844, "RGBA8": 21}
# DXT3 при пересборке кодируем как DXT5 (нет BC2-энкодера; размер блока тот же)


@dataclass
class TexInfo:
    version: int
    unk1: int
    mip_count: int
    image_count: int
    width: int
    height: int
    unk2: int
    fmt: str
    unk_floats: bytes  # 16 байт с 0x18, сохраняются как есть


def parse_tex(data: bytes) -> TexInfo:
    if data[:4] != TEX_MAGIC:
        raise UnsupportedTexture("not a TEX")
    version, unk1 = struct.unpack_from("<HH", data, 4)
    packed, = struct.unpack_from("<I", data, 8)
    mips, imgs = packed & 0xFF, (packed >> 8) & 0xFF
    w, h = struct.unpack_from("<HH", data, 0x0C)
    unk2, fmt_u32 = struct.unpack_from("<II", data, 0x10)
    if imgs != 1:
        raise UnsupportedTexture(f"image_count={imgs} (cubemap?)")
    if fmt_u32 not in _FMT_FROM_U32:
        raise UnsupportedTexture(f"tex format 0x{fmt_u32:x}")
    return TexInfo(version, unk1, mips, imgs, w, h, unk2, _FMT_FROM_U32[fmt_u32], data[0x18:0x28])


def tex_pixels(data: bytes, info: TexInfo) -> np.ndarray:
    offs = struct.unpack_from(f"<{info.mip_count}I", data, 0x28)
    size = bcn_size(info.width, info.height, info.fmt)
    return decode_bcn(data[offs[0] : offs[0] + size], info.width, info.height, info.fmt)


def build_tex(info: TexInfo, rgba: np.ndarray) -> bytes:
    h, w = rgba.shape[:2]
    if w > 0xFFFF or h > 0xFFFF:
        raise ValueError("dimensions exceed u16")
    mips = info.mip_count
    out_fmt = "DXT5" if info.fmt == "DXT3" else info.fmt
    blobs = [encode_bcn(m, out_fmt) for m in build_mip_chain(rgba, mips)]
    header_size = 0x28 + 4 * mips
    offsets, pos = [], header_size
    for b in blobs:
        offsets.append(pos)
        pos += len(b)
    head = TEX_MAGIC + struct.pack("<HH", info.version, info.unk1)
    head += struct.pack("<I", mips | (info.image_count << 8))
    head += struct.pack("<HH", w, h)
    head += struct.pack("<II", info.unk2, _FMT_TO_U32[info.fmt])
    head += info.unk_floats
    head += struct.pack(f"<{mips}I", *offsets)
    return head + b"".join(blobs)


class MtfTexCodec:
    name = "mtf-tex"

    def detect(self, path: Path) -> bool:
        if path.suffix.lower() != ".tex":
            return False
        try:
            with open(path, "rb") as f:
                return f.read(4) == TEX_MAGIC
        except OSError:
            return False

    def decode(self, path: Path) -> list[TextureItem]:
        data = path.read_bytes()
        info = parse_tex(data)
        rgba = tex_pixels(data, info)
        meta = {"format": info.fmt, "mip_count": info.mip_count, "tex": True}
        return [TextureItem(path, None, self.name, rgba, meta)]

    def encode_file(self, path: Path, replacements: dict[str, np.ndarray]) -> bytes:
        info = parse_tex(path.read_bytes())
        return build_tex(info, replacements[""])
