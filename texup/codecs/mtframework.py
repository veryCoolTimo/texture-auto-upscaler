from __future__ import annotations

import hashlib
import struct
import zlib
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
    if len(data) < 0x2C:
        raise UnsupportedTexture("truncated TEX header")
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
    # Мип-семантика как у DdsCodec.encode_file: multi-mip исходник получает
    # полную цепочку под НОВЫЕ размеры; одномиповый остаётся одномиповым.
    mips = mip_levels_for(w, h) if info.mip_count > 1 else 1
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


# --- TEX v2 (MT Framework v2, e.g. Resident Evil 0/1 HD) ---
# ARC container is byte-identical to v1 (magic/entries/hash/zlib/data-start); only the TEX
# payload header differs. Header layout, discovered empirically by size math across the
# full RE0 texture set (10k+ files, exact payload-size match):
#   0x00 magic "TEX\0"
#   0x04 u32 version; (v & 0xFFF) is the version number (157 for RE0), remaining bits are
#        flags — preserved verbatim on rebuild.
#   0x08 u32 packed: mip_count = v&0x3F, width = (v>>6)&0x1FFF, height = (v>>19)&0x1FFF
#   0x0C u32 packed: image_count = v&0xFF (6 = cubemap, unsupported),
#        format_id = (v>>8)&0xFF, upper 16 bits unknown — preserved verbatim
#   0x10 mip_count x u32 absolute file offsets (same semantics as v1, just no 16-byte
#        float preamble before the offset table)
#   then tightly-packed mip chain (block-compressed or raw, same layout as v1)
_TEX2_VERSIONS = {157, 154}  # 157 = RE0/RE1 HD (0x09D), 154 = RE6 (0x9A)
# format_id -> block/pixel codec. Multiple ids can share a codec (e.g. 20/25 both BC1 —
# likely sRGB vs linear DXGI variants of the same bit layout; we don't do gamma handling
# so both decode identically). format_id itself is preserved verbatim on rebuild.
_FMT2_FROM_ID = {
    20: "DXT1",  # BC1, _BM diffuse (opaque)
    24: "DXT5",  # BC3, _BM diffuse (alpha)
    25: "DXT1",  # BC1, _MM/_AM (metalness/AO, single-channel content in a BC1 container)
    31: "BC5",   # _NM normal maps
    43: "BC7",   # UI/photographic (_ID_HQ icons, logos — confirmed via CAPCOM logo decode)
    14: "RGBA8", # uncompressed 32bpp (e.g. *_z depth-ish room textures)
    39: "RGBA8", # uncompressed 32bpp (TVNoise/TVMask filter textures)
    40: "RGBA8", # uncompressed 32bpp (NullBlack/NullWhite, *_ID_HQ save icons)
}


@dataclass
class Tex2Info:
    version: int  # full u32 at 0x04, preserved verbatim
    mip_count: int
    width: int
    height: int
    image_count: int
    format_id: int
    packed2_upper: int  # bits 16-31 of the packed u32 at 0x0C, unknown, preserved verbatim
    fmt: str


def parse_tex2(data: bytes) -> Tex2Info:
    if data[:4] != TEX_MAGIC:
        raise UnsupportedTexture("not a TEX")
    if len(data) < 0x10:
        raise UnsupportedTexture("truncated TEX2 header")
    version, = struct.unpack_from("<I", data, 4)
    packed1, = struct.unpack_from("<I", data, 8)
    mip_count = packed1 & 0x3F
    width = (packed1 >> 6) & 0x1FFF
    height = (packed1 >> 19) & 0x1FFF
    packed2, = struct.unpack_from("<I", data, 0xC)
    image_count = packed2 & 0xFF
    format_id = (packed2 >> 8) & 0xFF
    upper16 = (packed2 >> 16) & 0xFFFF
    if image_count != 1:
        raise UnsupportedTexture(f"tex2 image_count={image_count} (cubemap?)")
    if mip_count == 0:
        raise UnsupportedTexture("tex2 mip_count=0")
    if format_id not in _FMT2_FROM_ID:
        raise UnsupportedTexture(f"tex2 format id {format_id}")
    header_size = 0x10 + 4 * mip_count
    if len(data) < header_size:
        raise UnsupportedTexture("truncated TEX2 offset table")
    return Tex2Info(version, mip_count, width, height, image_count, format_id, upper16,
                     _FMT2_FROM_ID[format_id])


def tex2_pixels(data: bytes, info: Tex2Info) -> np.ndarray:
    offs = struct.unpack_from(f"<{info.mip_count}I", data, 0x10)
    size = bcn_size(info.width, info.height, info.fmt)
    return decode_bcn(data[offs[0] : offs[0] + size], info.width, info.height, info.fmt)


def build_tex2(info: Tex2Info, rgba: np.ndarray) -> bytes:
    h, w = rgba.shape[:2]
    if w > 0x1FFF or h > 0x1FFF:
        raise ValueError("dimensions exceed 13-bit field")
    # Мип-семантика как у v1: multi-mip исходник получает полную цепочку под новые
    # размеры; одномиповый остаётся одномиповым.
    mips = mip_levels_for(w, h) if info.mip_count > 1 else 1
    blobs = [encode_bcn(m, info.fmt) for m in build_mip_chain(rgba, mips)]
    header_size = 0x10 + 4 * mips
    offsets, pos = [], header_size
    for b in blobs:
        offsets.append(pos)
        pos += len(b)
    packed1 = mips | (w << 6) | (h << 19)
    packed2 = info.image_count | (info.format_id << 8) | (info.packed2_upper << 16)
    head = TEX_MAGIC + struct.pack("<I", info.version)
    head += struct.pack("<I", packed1)
    head += struct.pack("<I", packed2)
    head += struct.pack(f"<{mips}I", *offsets)
    return head + b"".join(blobs)


def parse_tex_any(data: bytes) -> TexInfo | Tex2Info:
    """Dispatch on the low-12-bit version field: 112 -> v1 (RE5), 157/154 -> v2 (RE0/RE1 HD/RE6)."""
    if data[:4] != TEX_MAGIC:
        raise UnsupportedTexture("not a TEX")
    if len(data) < 8:
        raise UnsupportedTexture("truncated TEX header")
    version_u32, = struct.unpack_from("<I", data, 4)
    if (version_u32 & 0xFFF) in _TEX2_VERSIONS:
        return parse_tex2(data)
    return parse_tex(data)


def tex_pixels_any(data: bytes, info: TexInfo | Tex2Info) -> np.ndarray:
    if isinstance(info, Tex2Info):
        return tex2_pixels(data, info)
    return tex_pixels(data, info)


def build_tex_any(info: TexInfo | Tex2Info, rgba: np.ndarray) -> bytes:
    if isinstance(info, Tex2Info):
        return build_tex2(info, rgba)
    return build_tex(info, rgba)


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
        info = parse_tex_any(data)
        rgba = tex_pixels_any(data, info)
        meta = {
            "format": info.fmt, "mip_count": info.mip_count, "tex": True,
            "content_sha": hashlib.sha256(data).hexdigest(),
        }
        return [TextureItem(path, None, self.name, rgba, meta)]

    def encode_file(self, path: Path, replacements: dict[str, np.ndarray]) -> bytes:
        info = parse_tex_any(path.read_bytes())
        return build_tex_any(info, replacements[""])


# ARC container codec
ARC_MAGIC = 0x00435241  # 'ARC\0'
ARC_MAGIC_ENC = 0x43435241  # 'ARCC' (Blowfish, не поддерживаем)
ARC_TEXTURE_HASH = 0x241F5DEB  # rTexture


@dataclass
class ArcEntry:
    name: str
    type_hash: int
    comp_size: int
    uncomp_size: int
    quality: int
    offset: int


def parse_arc(data: bytes) -> tuple[int, list[ArcEntry]]:
    magic, = struct.unpack_from("<I", data, 0)
    if magic == ARC_MAGIC_ENC:
        raise UnsupportedTexture("encrypted ARC not supported")
    if magic != ARC_MAGIC:
        raise UnsupportedTexture("not an ARC")
    version, count = struct.unpack_from("<HH", data, 4)
    entries = []
    for i in range(count):
        off = 8 + i * 80
        name = data[off : off + 64].split(b"\x00", 1)[0].decode("ascii")
        type_hash, comp_size, flags, offset = struct.unpack_from("<IIII", data, off + 64)
        entries.append(ArcEntry(name, type_hash, comp_size, flags & 0x1FFFFFFF, flags >> 29, offset))
    return version, entries


def build_arc(version: int, entries: list[tuple[str, int, bytes]],
              quality: dict[int, int] | None = None,
              precompressed: dict[int, bytes] | None = None,
              uncomp_sizes: dict[int, int] | None = None,
              data_start: int | None = None) -> bytes:
    """entries: (name, type_hash, raw_bytes). precompressed[i] — готовый zlib-поток
    (для нетронутых энтри, чтобы репак был байт-идентичным); для таких энтри raw_bytes
    пустые, а несжатый размер берётся из uncomp_sizes[i].

    data_start: реальные архивы RE5 резервируют зануленную область перед первым
    блоком данных (первый offset обычно 32768 или 65536, а не сразу после
    таблицы заголовков). Если задан и больше header_size, между таблицей и
    данными добавляется зануленный паддинг соответствующего размера."""
    quality = quality or {}
    precompressed = precompressed or {}
    uncomp_sizes = uncomp_sizes or {}
    blobs = []
    for i, (name, type_hash, raw) in enumerate(entries):
        if len(name.encode("ascii")) >= 64:
            raise ValueError(f"entry name too long: {name!r}")
        comp = precompressed[i] if i in precompressed else zlib.compress(raw, 6)
        blobs.append((name, type_hash, comp, uncomp_sizes.get(i, len(raw))))
    header_size = 8 + len(blobs) * 80
    pos = max(header_size, data_start) if data_start else header_size
    out = struct.pack("<IHH", ARC_MAGIC, version, len(blobs))
    start_pos = pos
    for i, (name, type_hash, comp, usize) in enumerate(blobs):
        flags = (usize & 0x1FFFFFFF) | ((quality.get(i, 2) & 0x7) << 29)
        out += name.encode("ascii").ljust(64, b"\x00")
        out += struct.pack("<IIII", type_hash, len(comp), flags, pos)
        pos += len(comp)
    out += b"\x00" * (start_pos - header_size)
    return out + b"".join(comp for _, _, comp, _ in blobs)


class MtfArcCodec:
    name = "mtf-arc"

    def detect(self, path: Path) -> bool:
        if path.suffix.lower() != ".arc":
            return False
        try:
            with open(path, "rb") as f:
                return f.read(4) == b"ARC\x00"
        except OSError:
            return False

    def decode(self, path: Path) -> list[TextureItem]:
        data = path.read_bytes()
        _, entries = parse_arc(data)
        items = []
        for e in entries:
            if e.type_hash != ARC_TEXTURE_HASH:
                continue
            comp_blob = data[e.offset : e.offset + e.comp_size]
            raw = zlib.decompress(comp_blob)
            try:
                info = parse_tex_any(raw)
            except UnsupportedTexture:
                continue  # кубмапы и прочее — пропуск
            rgba = tex_pixels_any(raw, info)
            meta = {
                "format": info.fmt, "mip_count": info.mip_count, "tex": True, "arc_entry": e.name,
                # sha256 of the COMPRESSED entry blob (cheap, unique per content, avoids
                # paying for zlib.decompress just to hash the raw bytes).
                "content_sha": hashlib.sha256(comp_blob).hexdigest(),
            }
            items.append(TextureItem(path, e.name, self.name, rgba, meta))
        return items

    def encode_file(self, path: Path, replacements: dict[str, np.ndarray]) -> bytes:
        data = path.read_bytes()
        version, entries = parse_arc(data)

        out_entries, quality, precompressed, uncomp_sizes = [], {}, {}, {}
        for i, e in enumerate(entries):
            comp = data[e.offset : e.offset + e.comp_size]
            quality[i] = e.quality
            if e.type_hash == ARC_TEXTURE_HASH and e.name in replacements:
                raw = zlib.decompress(comp)
                new_raw = build_tex_any(parse_tex_any(raw), replacements[e.name])
                out_entries.append((e.name, e.type_hash, new_raw))
            else:
                out_entries.append((e.name, e.type_hash, b""))
                precompressed[i] = comp
                uncomp_sizes[i] = e.uncomp_size
        data_start = min((e.offset for e in entries), default=None)
        return build_arc(version, out_entries, quality, precompressed, uncomp_sizes, data_start=data_start)
