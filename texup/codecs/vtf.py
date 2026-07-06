"""Valve VTF texture format (Source engine) — decode + encode.

Layout facts (verified against srctools.vtf, our authoritative reference):
- Header: magic b"VTF\\0" + version (II) + a fixed-layout struct (see `_HEADER_CORE`)
  covering size/flags/format/mip-count/low-res info, then an optional depth (u16,
  version >= 7.2) and an optional resource directory (version >= 7.3).
- Pre-7.3 (no resource directory): the low-res thumbnail sits immediately after the
  header, followed directly by the high-res mip chain.
- 7.3+: an arbitrary set of "resources" (tag + flags + offset/value) is stored; the
  low-res and high-res images are just two well-known resource tags
  (b"\\x01\\0\\0" / b"\\x30\\0\\0") whose "offset" field points straight at the pixel
  data (no size prefix, unlike other resource kinds).
- Mip levels are stored SMALLEST-first, largest (full resolution) LAST — the inverse
  of DDS. `mipmap_count` mip levels are stored, sized as `max(dim >> level, 1)` for
  level in `0..mipmap_count-1` (level 0 = full res); note srctools sometimes doesn't
  go all the way down to a 1x1 mip (that's just how the reference happens to author
  files) — we don't assume anything about the smallest level, we just trust the field.
"""
from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np
from PIL import Image

from texup.codecs.base import TextureItem, UnsupportedTexture
from texup.codecs.bcn import bcn_size, build_mip_chain, decode_bcn, encode_bcn, mip_levels_for

_MAGIC = b"VTF\0"

# Fixed-layout struct right after magic+version (12 bytes in), matching
# srctools.vtf._HEADER exactly: header_size, width, height, flags, frame_count,
# first_frame_index, (pad) reflectivity(fff), (pad) bumpmap_scale, high_format,
# mipmap_count, low_format, low_width, low_height.
_HEADER_CORE = struct.Struct("<IHHIHH4xfff4xfiBiBB")

_ENVMAP_FLAG = 0x00004000

# VTF ImageFormats on-disk indices we support (see srctools.vtf.ImageFormats / FORMAT_ORDER).
_FMT_BY_IND = {
    0: "RGBA8888",
    2: "RGB888",
    3: "BGR888",
    12: "BGRA8888",
    13: "DXT1",
    14: "DXT3",
    15: "DXT5",
}
_IND_BY_FMT = {name: ind for ind, name in _FMT_BY_IND.items()}
_NONE_FMT_IND = -1

_LOW_RES_TAG = b"\x01\x00\x00"
_HIGH_RES_TAG = b"\x30\x00\x00"


def _frame_size(fmt: str, w: int, h: int) -> int:
    if fmt in ("DXT1", "DXT3", "DXT5"):
        return bcn_size(w, h, fmt)
    bpp = {"RGBA8888": 4, "BGRA8888": 4, "RGB888": 3, "BGR888": 3}[fmt]
    return w * h * bpp


def _decode_pixels(data: bytes, w: int, h: int, fmt: str) -> np.ndarray:
    if fmt in ("DXT1", "DXT3", "DXT5"):
        return decode_bcn(data, w, h, fmt)
    if fmt == "RGBA8888":
        return np.frombuffer(data, dtype=np.uint8, count=w * h * 4).reshape(h, w, 4).copy()
    if fmt == "BGRA8888":
        arr = np.frombuffer(data, dtype=np.uint8, count=w * h * 4).reshape(h, w, 4)
        return arr[..., [2, 1, 0, 3]].copy()
    if fmt == "RGB888":
        arr = np.frombuffer(data, dtype=np.uint8, count=w * h * 3).reshape(h, w, 3)
        out = np.empty((h, w, 4), dtype=np.uint8)
        out[..., :3] = arr
        out[..., 3] = 255
        return out
    if fmt == "BGR888":
        arr = np.frombuffer(data, dtype=np.uint8, count=w * h * 3).reshape(h, w, 3)
        out = np.empty((h, w, 4), dtype=np.uint8)
        out[..., 0] = arr[..., 2]
        out[..., 1] = arr[..., 1]
        out[..., 2] = arr[..., 0]
        out[..., 3] = 255
        return out
    raise UnsupportedTexture(f"vtf format {fmt!r}")


def _encode_pixels(rgba: np.ndarray, fmt: str) -> bytes:
    if fmt in ("DXT1", "DXT3", "DXT5"):
        # DXT3 (BC2) has no ispc_texcomp encoder; bcn.encode_bcn already falls back to
        # DXT5, same policy as the DDS codec. The written high_format is adjusted to
        # match (see encode_bytes).
        blob = encode_bcn(rgba, fmt)
        # ispc_texcomp's BC1 path (used for DXT1) returns a buffer padded to double the
        # real per-block size (an internal SIMD-batching quirk — the tail is
        # uninitialized, not meaningful padding). DDS never notices, since it only ever
        # reads mip 0 back from the front of the blob; VTF concatenates mips
        # back-to-back with no slack, so keep only the real bytes.
        h, w = rgba.shape[:2]
        return blob[: bcn_size(w, h, fmt)]
    if fmt == "RGBA8888":
        return np.ascontiguousarray(rgba).tobytes()
    if fmt == "BGRA8888":
        return np.ascontiguousarray(rgba[..., [2, 1, 0, 3]]).tobytes()
    if fmt == "RGB888":
        return np.ascontiguousarray(rgba[..., :3]).tobytes()
    if fmt == "BGR888":
        return np.ascontiguousarray(rgba[..., [2, 1, 0]]).tobytes()
    raise UnsupportedTexture(f"vtf format {fmt!r}")


class _Resource:
    __slots__ = ("tag", "flags", "data_pos", "value", "external_content")

    def __init__(self, tag: bytes, flags: int, data_pos: int, value: int):
        self.tag = tag
        self.flags = flags
        self.data_pos = data_pos  # byte offset (within file) of the 4-byte value field
        self.value = value
        self.external_content: bytes | None = None  # populated for passthrough resources


class _ParsedVtf:
    """Everything needed from a VTF header/body to decode the main image and, later,
    to rebuild a new file around a replacement image."""

    def __init__(self, data: bytes):
        self.data = data
        try:
            self._parse()
        except (struct.error, IndexError) as exc:
            raise UnsupportedTexture(f"truncated/malformed VTF: {exc}") from exc

    def _parse(self) -> None:
        data = self.data
        if len(data) < 12 or data[:4] != _MAGIC:
            raise UnsupportedTexture("not a VTF (bad magic)")
        version_major, version_minor = struct.unpack_from("<II", data, 4)
        if version_major != 7 or not (0 <= version_minor <= 5):
            raise UnsupportedTexture(f"unsupported VTF version {version_major}.{version_minor}")
        self.version = (version_major, version_minor)

        (
            header_size,
            width,
            height,
            flags,
            frame_count,
            _first_frame_index,
            _ref_r,
            _ref_g,
            _ref_b,
            _bump_scale,
            high_format,
            mipmap_count,
            low_format,
            low_width,
            low_height,
        ) = _HEADER_CORE.unpack_from(data, 12)

        self.header_size = header_size
        self.width = width
        self.height = height
        self.flags = flags
        self.frame_count = frame_count
        self.high_format_ind = high_format
        self.mipmap_count = mipmap_count
        self.low_format_ind = low_format
        self.low_width = low_width
        self.low_height = low_height

        # Absolute byte offsets (field positions within _HEADER_CORE + the 12-byte
        # magic/version prefix), used later by encode_bytes() to patch in place.
        self.width_pos = 12 + 4
        self.height_pos = 12 + 6
        self.high_format_pos = 12 + 40
        self.mipcount_pos = 12 + 44

        pos = 12 + _HEADER_CORE.size

        self.depth = 1
        if version_minor >= 2:
            (self.depth,) = struct.unpack_from("<H", data, pos)
            pos += 2

        self.resources: list[_Resource] = []
        low_res_offset = -1
        high_res_offset = -1

        if version_minor >= 3:
            (num_resources,) = struct.unpack_from("<3xI8x", data, pos)
            pos += 15
            for _ in range(num_resources):
                tag, res_flags, value = struct.unpack_from("<3sBI", data, pos)
                res = _Resource(tag, res_flags, pos + 4, value)
                self.resources.append(res)
                pos += 8
                if tag == _LOW_RES_TAG:
                    low_res_offset = value
                elif tag == _HIGH_RES_TAG:
                    high_res_offset = value

            # Passthrough content for any resource we don't understand, so encode()
            # can carry it through unchanged.
            for res in self.resources:
                if res.tag in (_LOW_RES_TAG, _HIGH_RES_TAG):
                    continue
                if res.flags & 0x02:
                    continue  # inline value, nothing to relocate
                offset = res.value
                (size,) = struct.unpack_from("<I", data, offset)
                res.external_content = data[offset + 4 : offset + 4 + size]
        else:
            low_res_offset = header_size
            low_fmt = _FMT_BY_IND.get(low_format)
            if low_format == _NONE_FMT_IND:
                low_size = 0
            elif low_fmt is not None:
                low_size = _frame_size(low_fmt, low_width, low_height)
            else:
                raise UnsupportedTexture(f"vtf low-res format ind {low_format}")
            high_res_offset = low_res_offset + low_size

        if high_res_offset < 0:
            raise UnsupportedTexture("VTF missing high-res image resource")

        self.low_res_offset = low_res_offset
        self.high_res_offset = high_res_offset
        # Trust the file's own header_size field for the header/body boundary rather
        # than our own running `pos`: pre-7.3 files carry a further fixed pad after the
        # depth field (srctools pads to a fixed 80 bytes there) that isn't otherwise
        # reflected in the fields we parse.
        self.header_end = header_size

    def validate_scope(self) -> str:
        """Check frames/faces/depth scope and return the resolved format name."""
        if self.flags & _ENVMAP_FLAG:
            raise UnsupportedTexture("cubemap VTF not supported")
        if self.frame_count != 1:
            raise UnsupportedTexture(f"animated VTF ({self.frame_count} frames) not supported")
        if self.depth != 1:
            raise UnsupportedTexture(f"volumetric VTF (depth={self.depth}) not supported")
        fmt = _FMT_BY_IND.get(self.high_format_ind)
        if fmt is None:
            raise UnsupportedTexture(f"vtf high-res format ind {self.high_format_ind}")
        if self.mipmap_count < 1:
            raise UnsupportedTexture("VTF has no mip levels")
        return fmt

    def top_mip_bytes(self, fmt: str) -> bytes:
        """Extract the full-resolution mip (stored last in the mip chain)."""
        offset = self.high_res_offset
        for level in range(self.mipmap_count - 1, 0, -1):
            w = max(self.width >> level, 1)
            h = max(self.height >> level, 1)
            offset += _frame_size(fmt, w, h)
        size = _frame_size(fmt, self.width, self.height)
        return self.data[offset : offset + size]


class VtfCodec:
    name = "vtf"

    def detect(self, path: Path) -> bool:
        if path.suffix.lower() != ".vtf":
            return False
        try:
            with open(path, "rb") as f:
                return f.read(4) == _MAGIC
        except OSError:
            return False

    def decode_bytes(self, data: bytes) -> tuple[np.ndarray, dict]:
        parsed = _ParsedVtf(data)
        fmt = parsed.validate_scope()
        top = parsed.top_mip_bytes(fmt)
        rgba = _decode_pixels(top, parsed.width, parsed.height, fmt)
        meta = {
            "format": fmt,
            "mip_count": parsed.mipmap_count,
            "vtf_version": parsed.version,
        }
        return rgba, meta

    def decode(self, path: Path) -> list[TextureItem]:
        data = path.read_bytes()
        rgba, meta = self.decode_bytes(data)
        meta["content_sha"] = hashlib.sha256(data).hexdigest()
        return [TextureItem(path, None, self.name, rgba, meta)]

    def encode_bytes(self, data_orig: bytes, rgba: np.ndarray) -> bytes:
        parsed = _ParsedVtf(data_orig)
        fmt = parsed.validate_scope()
        out_fmt = "DXT5" if fmt == "DXT3" else fmt

        new_h, new_w = rgba.shape[:2]
        new_mip_count = mip_levels_for(new_w, new_h) if parsed.mipmap_count > 1 else 1

        header = bytearray(data_orig[: parsed.header_end])
        struct.pack_into("<H", header, parsed.width_pos, new_w)
        struct.pack_into("<H", header, parsed.height_pos, new_h)
        struct.pack_into("<B", header, parsed.mipcount_pos, new_mip_count)
        if out_fmt != fmt:
            struct.pack_into("<i", header, parsed.high_format_pos, _IND_BY_FMT[out_fmt])

        # Full mip chain, largest first; file order is smallest -> largest.
        chain = build_mip_chain(rgba, new_mip_count)
        highres_blob = b"".join(_encode_pixels(m, out_fmt) for m in reversed(chain))

        thumb_fmt = _FMT_BY_IND.get(parsed.low_format_ind)
        if parsed.low_format_ind == _NONE_FMT_IND or parsed.low_width == 0 or parsed.low_height == 0:
            thumb_blob = b""
        elif thumb_fmt == "DXT1":
            img = Image.fromarray(rgba, "RGBA")
            small = np.asarray(img.resize((parsed.low_width, parsed.low_height), Image.BOX))
            thumb_blob = _encode_pixels(small, "DXT1")
        else:
            raise UnsupportedTexture(f"vtf low-res format {parsed.low_format_ind!r}")

        if parsed.version[1] < 3:
            return bytes(header) + thumb_blob + highres_blob

        # 7.3+: header already includes the resource directory; append the pixel
        # blobs plus any passthrough resources, then patch offsets in place.
        cursor = len(header)
        low_offset = cursor
        cursor += len(thumb_blob)
        high_offset = cursor
        cursor += len(highres_blob)

        extra_blobs: list[bytes] = []
        patches: list[tuple[int, int]] = []  # (pos, value)
        for res in parsed.resources:
            if res.tag == _LOW_RES_TAG:
                patches.append((res.data_pos, low_offset))
            elif res.tag == _HIGH_RES_TAG:
                patches.append((res.data_pos, high_offset))
            elif res.external_content is not None:
                patches.append((res.data_pos, cursor))
                block = struct.pack("<I", len(res.external_content)) + res.external_content
                extra_blobs.append(block)
                cursor += len(block)
            # inline (flags & 0x02) resources keep their literal value untouched.

        for pos, value in patches:
            struct.pack_into("<I", header, pos, value)

        return bytes(header) + thumb_blob + highres_blob + b"".join(extra_blobs)

    def encode_file(self, path: Path, replacements: dict[str, np.ndarray]) -> bytes:
        data = path.read_bytes()
        return self.encode_bytes(data, replacements[""])
