from __future__ import annotations

import numpy as np
from PIL import Image


def _bgra_to_rgba(buf: bytes, w: int, h: int) -> np.ndarray:
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    return arr[..., [2, 1, 0, 3]].copy()


def decode_bcn(data: bytes, w: int, h: int, fmt: str) -> np.ndarray:
    import texture2ddecoder as t2d

    if fmt == "RGBA8":  # D3D A8R8G8B8: в памяти BGRA
        return _bgra_to_rgba(data[: w * h * 4], w, h)
    # DXT3 (BC2) uses BC3 decoder since BC2 not available in texture2ddecoder
    fmt = "DXT5" if fmt == "DXT3" else fmt
    decoders = {
        "DXT1": t2d.decode_bc1,
        "DXT5": t2d.decode_bc3,
        "BC5": t2d.decode_bc5,
    }
    return _bgra_to_rgba(decoders[fmt](data, w, h), w, h)


def encode_bcn(rgba: np.ndarray, fmt: str) -> bytes:
    import ispc_texcomp as itc

    h, w = rgba.shape[:2]
    if fmt == "RGBA8":
        return rgba[..., [2, 1, 0, 3]].tobytes()  # обратно в BGRA
    # DXT3 (BC2) не поддерживается в ispc_texcomp; кодируем как DXT5
    fmt = "DXT5" if fmt == "DXT3" else fmt
    # ispc_texcomp требует размеры кратные 4: паддинг краевыми пикселями
    pw, ph = (w + 3) // 4 * 4, (h + 3) // 4 * 4
    if (pw, ph) != (w, h):
        padded = np.empty((ph, pw, 4), dtype=np.uint8)
        padded[:h, :w] = rgba
        padded[h:, :w] = rgba[h - 1 :, :]
        padded[:, w:] = padded[:, w - 1 : w]
        rgba_padded = padded
    else:
        rgba_padded = rgba

    # BC5 требует 2-byte-per-pixel RG поверхность (stride = width*2)
    if fmt == "BC5":
        rg = np.ascontiguousarray(rgba_padded[..., :2])  # (h, w, 2) interleaved RG
        surf = itc.RGBASurface(rg.tobytes(), pw, ph, stride=pw * 2)
    else:
        # DXT1/DXT5 используют RGBA поверхность (stride = width*4)
        surf = itc.RGBASurface(rgba_padded.tobytes(), pw, ph, stride=pw * 4)

    encoders = {
        "DXT1": itc.compress_blocks_bc1,
        "DXT5": itc.compress_blocks_bc3,
        "BC5": itc.compress_blocks_bc5,
    }
    return bytes(encoders[fmt](surf))


def mip_levels_for(w: int, h: int) -> int:
    levels = 1
    while w > 1 or h > 1:
        w, h = max(1, w // 2), max(1, h // 2)
        levels += 1
    return levels


def build_mip_chain(rgba: np.ndarray, levels: int) -> list[np.ndarray]:
    chain = [rgba]
    img = Image.fromarray(rgba, "RGBA")
    w, h = img.size
    for _ in range(levels - 1):
        w, h = max(1, w // 2), max(1, h // 2)
        chain.append(np.asarray(img.resize((w, h), Image.BOX)))
    return chain


def bcn_size(w: int, h: int, fmt: str) -> int:
    if fmt == "RGBA8":
        return w * h * 4
    block = 8 if fmt == "DXT1" else 16
    return max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * block
