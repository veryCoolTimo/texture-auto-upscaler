from __future__ import annotations

import numpy as np
from PIL import Image


def _bgra_to_rgba(buf: bytes, w: int, h: int) -> np.ndarray:
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    return arr[..., [2, 1, 0, 3]].copy()


def _decode_bc2(data: bytes, w: int, h: int) -> np.ndarray:
    """Decode BC2/DXT3: explicit 4-bit alpha + BC1-style (always 4-color) RGB."""
    bw, bh = max(1, (w + 3) // 4), max(1, (h + 3) // 4)
    blocks = np.frombuffer(data, dtype=np.uint8, count=bw * bh * 16).reshape(bh, bw, 16)

    # --- alpha: bytes 0-7, 4 bits/texel, low nibble first within each byte ---
    alpha_bytes = blocks[..., 0:8]
    low = alpha_bytes & 0x0F
    high = (alpha_bytes >> 4) & 0x0F
    alpha_nibbles = np.stack([low, high], axis=-1).reshape(bh, bw, 16)  # texel order 0..15
    alpha8 = (alpha_nibbles.astype(np.int32) * 17).astype(np.uint8)

    # --- colors: bytes 8-9 = c0 (LE), bytes 10-11 = c1 (LE) ---
    c0 = blocks[..., 8].astype(np.uint32) | (blocks[..., 9].astype(np.uint32) << 8)
    c1 = blocks[..., 10].astype(np.uint32) | (blocks[..., 11].astype(np.uint32) << 8)

    def expand565(v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = ((v >> 11) & 0x1F).astype(np.int32) * 255 // 31
        g = ((v >> 5) & 0x3F).astype(np.int32) * 255 // 63
        b = (v & 0x1F).astype(np.int32) * 255 // 31
        return r, g, b

    r0, g0, b0 = expand565(c0)
    r1, g1, b1 = expand565(c1)
    r2, g2, b2 = (2 * r0 + r1) // 3, (2 * g0 + g1) // 3, (2 * b0 + b1) // 3
    r3, g3, b3 = (r0 + 2 * r1) // 3, (g0 + 2 * g1) // 3, (b0 + 2 * b1) // 3

    palette_r = np.stack([r0, r1, r2, r3], axis=-1)  # (bh, bw, 4)
    palette_g = np.stack([g0, g1, g2, g3], axis=-1)
    palette_b = np.stack([b0, b1, b2, b3], axis=-1)

    # --- indices: bytes 12-15, 32-bit LE, 2 bits/texel, texel 0 in lowest bits ---
    ib = blocks[..., 12:16].astype(np.uint32)
    idx32 = ib[..., 0] | (ib[..., 1] << 8) | (ib[..., 2] << 16) | (ib[..., 3] << 24)
    shifts = np.arange(16, dtype=np.uint32) * 2
    indices = (idx32[..., None] >> shifts) & 0x3  # (bh, bw, 16)

    by_idx = np.arange(bh)[:, None, None]
    bx_idx = np.arange(bw)[None, :, None]
    r_texels = palette_r[by_idx, bx_idx, indices]  # (bh, bw, 16)
    g_texels = palette_g[by_idx, bx_idx, indices]
    b_texels = palette_b[by_idx, bx_idx, indices]

    def blocks_to_image(t: np.ndarray) -> np.ndarray:
        return t.reshape(bh, bw, 4, 4).transpose(0, 2, 1, 3).reshape(bh * 4, bw * 4)

    out = np.empty((bh * 4, bw * 4, 4), dtype=np.uint8)
    out[..., 0] = blocks_to_image(r_texels)
    out[..., 1] = blocks_to_image(g_texels)
    out[..., 2] = blocks_to_image(b_texels)
    out[..., 3] = blocks_to_image(alpha8)
    return out[:h, :w]


def decode_bcn(data: bytes, w: int, h: int, fmt: str) -> np.ndarray:
    import texture2ddecoder as t2d

    if fmt == "RGBA8":  # D3D A8R8G8B8: в памяти BGRA
        return _bgra_to_rgba(data[: w * h * 4], w, h)
    if fmt == "DXT3":  # BC2: explicit alpha, not available in texture2ddecoder
        return _decode_bc2(data, w, h)
    decoders = {
        "DXT1": t2d.decode_bc1,
        "DXT5": t2d.decode_bc3,
        "BC5": t2d.decode_bc5,
        "BC7": t2d.decode_bc7,
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
        # DXT1/DXT5/BC7 используют RGBA поверхность (stride = width*4)
        surf = itc.RGBASurface(rgba_padded.tobytes(), pw, ph, stride=pw * 4)

    if fmt == "BC7":
        settings = itc.BC7EncSettings.from_profile("alpha_slow")
        return bytes(itc.compress_blocks_bc7(surf, settings))

    encoders = {
        "DXT1": itc.compress_blocks_bc1,
        "DXT5": itc.compress_blocks_bc3,
        "BC5": itc.compress_blocks_bc5,
    }
    out = bytes(encoders[fmt](surf))
    return out[: bcn_size(pw, ph, fmt)]


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
