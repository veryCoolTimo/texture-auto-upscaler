from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np

from texup.codecs.base import TextureItem, UnsupportedTexture
from texup.codecs.bcn import bcn_size, build_mip_chain, decode_bcn, encode_bcn, mip_levels_for

_DDSD = 0x1 | 0x2 | 0x4 | 0x1000  # CAPS|HEIGHT|WIDTH|PIXELFORMAT
_DDSD_MIPMAPCOUNT = 0x20000
_DDSD_LINEARSIZE = 0x80000
_DDPF_FOURCC = 0x4
_DDPF_RGB = 0x40
_DDPF_ALPHAPIXELS = 0x1
_FOURCC = {b"DXT1": "DXT1", b"DXT3": "DXT3", b"DXT5": "DXT5", b"ATI2": "BC5", b"BC5U": "BC5"}
_FOURCC_OUT = {"DXT1": b"DXT1", "DXT3": b"DXT5", "DXT5": b"DXT5", "BC5": b"ATI2"}

# DXGI_FORMAT enum values (Microsoft dxgiformat.h) used by the DX10 extended header.
_DXGI_BC1_UNORM = 71
_DXGI_BC1_UNORM_SRGB = 72
_DXGI_BC2_UNORM = 74
_DXGI_BC2_UNORM_SRGB = 75
_DXGI_BC3_UNORM = 77
_DXGI_BC3_UNORM_SRGB = 78
_DXGI_BC5_UNORM = 83
_DXGI_BC5_SNORM = 84
_DXGI_BC7_UNORM = 98
_DXGI_BC7_UNORM_SRGB = 99
_DXGI_R8G8B8A8_UNORM = 28
_DXGI_R8G8B8A8_UNORM_SRGB = 29
_DXGI_TO_FMT = {
    _DXGI_R8G8B8A8_UNORM: "RGBA8",
    _DXGI_R8G8B8A8_UNORM_SRGB: "RGBA8",
    _DXGI_BC1_UNORM: "DXT1",
    _DXGI_BC1_UNORM_SRGB: "DXT1",
    _DXGI_BC2_UNORM: "DXT3",
    _DXGI_BC2_UNORM_SRGB: "DXT3",
    _DXGI_BC3_UNORM: "DXT5",
    _DXGI_BC3_UNORM_SRGB: "DXT5",
    _DXGI_BC5_UNORM: "BC5",
    _DXGI_BC5_SNORM: "BC5",
    _DXGI_BC7_UNORM: "BC7",
    _DXGI_BC7_UNORM_SRGB: "BC7",
}
_FMT_TO_DXGI = {
    "RGBA8": _DXGI_R8G8B8A8_UNORM,
    "DXT1": _DXGI_BC1_UNORM,
    "DXT3": _DXGI_BC2_UNORM,
    "DXT5": _DXGI_BC3_UNORM,
    "BC5": _DXGI_BC5_UNORM,
    "BC7": _DXGI_BC7_UNORM,
}
_DDS_DIMENSION_TEXTURE2D = 3


class DdsCodec:
    name = "dds"

    def detect(self, path: Path) -> bool:
        if path.suffix.lower() != ".dds":
            return False
        try:
            with open(path, "rb") as f:
                return f.read(4) == b"DDS "
        except OSError:
            return False

    def _parse(self, data: bytes) -> tuple[int, int, int, str, int, bool]:
        if data[:4] != b"DDS " or len(data) < 128:
            raise UnsupportedTexture("not a DDS")
        h, w = struct.unpack_from("<II", data, 12)
        mips = max(1, struct.unpack_from("<I", data, 28)[0])
        pf_flags, fourcc = struct.unpack_from("<I4s", data, 80)
        caps2 = struct.unpack_from("<I", data, 112)[0]
        if caps2 & 0x200:
            raise UnsupportedTexture("cubemap DDS not supported")
        if pf_flags & _DDPF_FOURCC:
            if fourcc == b"DX10":
                if len(data) < 148:
                    raise UnsupportedTexture("truncated DX10 header")
                dxgi_fmt = struct.unpack_from("<I", data, 128)[0]
                if dxgi_fmt not in _DXGI_TO_FMT:
                    raise UnsupportedTexture(f"DXGI format {dxgi_fmt}")
                return w, h, mips, _DXGI_TO_FMT[dxgi_fmt], 148, True
            if fourcc not in _FOURCC:
                raise UnsupportedTexture(f"fourcc {fourcc!r}")
            fmt = _FOURCC[fourcc]
        elif pf_flags & _DDPF_RGB:
            bitcount = struct.unpack_from("<I", data, 88)[0]
            if bitcount != 32:
                raise UnsupportedTexture(f"{bitcount}-bit uncompressed DDS")
            fmt = "RGBA8"
        else:
            raise UnsupportedTexture("unknown DDS pixel format")
        return w, h, mips, fmt, 128, False

    def decode_bytes(self, data: bytes) -> tuple[np.ndarray, dict]:
        w, h, mips, fmt, off, is_dx10 = self._parse(data)
        rgba = decode_bcn(data[off : off + bcn_size(w, h, fmt)], w, h, fmt)
        return rgba, {"format": fmt, "mip_count": mips, "is_dx10": is_dx10}

    def encode_bytes(self, data_orig: bytes, rgba: np.ndarray) -> bytes:
        _, _, mips, fmt, _, is_dx10 = self._parse(data_orig)
        h, w = rgba.shape[:2]
        new_mips = mip_levels_for(w, h) if mips > 1 else 1
        return self.build_dds(rgba, fmt, new_mips, force_dx10=is_dx10)

    def decode(self, path: Path) -> list[TextureItem]:
        data = path.read_bytes()
        rgba, meta = self.decode_bytes(data)
        meta["content_sha"] = hashlib.sha256(data).hexdigest()
        return [TextureItem(path, None, self.name, rgba, meta)]

    def build_dds(
        self, rgba: np.ndarray, fmt: str, mip_count: int, force_dx10: bool = False
    ) -> bytes:
        h, w = rgba.shape[:2]
        out_fmt = "DXT5" if fmt == "DXT3" else fmt
        chain = build_mip_chain(rgba, mip_count)
        blobs = [encode_bcn(m, out_fmt) for m in chain]
        flags = _DDSD | _DDSD_LINEARSIZE | (_DDSD_MIPMAPCOUNT if mip_count > 1 else 0)
        caps = 0x1000 | (0x400008 if mip_count > 1 else 0)  # TEXTURE | COMPLEX+MIPMAP
        # BC7 has no legacy FourCC — always requires the DX10 extended header.
        use_dx10 = force_dx10 or fmt == "BC7"
        if use_dx10:
            pf = struct.pack("<II4sIIIII", 32, _DDPF_FOURCC, b"DX10", 0, 0, 0, 0, 0)
        elif out_fmt == "RGBA8":
            pf = struct.pack(
                "<II4sIIIII", 32, _DDPF_RGB | _DDPF_ALPHAPIXELS, b"\0" * 4, 32,
                0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000,
            )
        else:
            pf = struct.pack("<II4sIIIII", 32, _DDPF_FOURCC, _FOURCC_OUT[fmt], 0, 0, 0, 0, 0)
        header = (
            b"DDS " + struct.pack("<IIIIII", 124, flags, h, w, len(blobs[0]), 0)
            + struct.pack("<I", mip_count) + b"\0" * 44 + pf
            + struct.pack("<IIIII", caps, 0, 0, 0, 0)
        )
        assert len(header) == 128
        if use_dx10:
            dx10_header = struct.pack(
                "<IIIII", _FMT_TO_DXGI[out_fmt], _DDS_DIMENSION_TEXTURE2D, 0, 1, 0
            )
            header += dx10_header
        return header + b"".join(blobs)

    def encode_file(self, path: Path, replacements: dict[str, np.ndarray]) -> bytes:
        data = path.read_bytes()
        return self.encode_bytes(data, replacements[""])
