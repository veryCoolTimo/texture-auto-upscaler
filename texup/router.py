from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image

from texup.codecs.base import TextureItem
from texup.presets import DEFAULT_PRESET, PRESETS


@dataclass
class Route:
    model: str | None
    pre: Callable[[np.ndarray], np.ndarray] | None = None
    post: Callable[[np.ndarray], np.ndarray] | None = None


def resize_classic(rgba: np.ndarray, scale: int) -> np.ndarray:
    h, w = rgba.shape[:2]
    img = Image.fromarray(rgba, "RGBA").resize((w * scale, h * scale), Image.LANCZOS)
    return np.asarray(img)


def renormalize(rgba: np.ndarray) -> np.ndarray:
    x = rgba[..., 0].astype(np.float32) / 127.5 - 1.0
    y = rgba[..., 1].astype(np.float32) / 127.5 - 1.0
    z = np.sqrt(np.clip(1.0 - x * x - y * y, 0.0, 1.0))
    out = rgba.copy()
    out[..., 2] = ((z + 1.0) * 127.5 + 0.5).astype(np.uint8)
    return out


def mtf_ag_unpack(rgba: np.ndarray) -> np.ndarray:
    """MT Framework DXT5-нормаль: X в A, Y в G -> обычная RG-нормаль."""
    out = np.zeros_like(rgba)
    out[..., 0] = rgba[..., 3]
    out[..., 1] = rgba[..., 1]
    out[..., 2] = 0
    out[..., 3] = 255
    return out


def mtf_ag_pack(rgba: np.ndarray) -> np.ndarray:
    """Обратно в DXT5nm: X -> A, Y -> G; R и B константные 255 (как в оригиналах RE5)."""
    out = np.empty_like(rgba)
    out[..., 0] = 255
    out[..., 1] = rgba[..., 1]
    out[..., 2] = 255
    out[..., 3] = rgba[..., 0]
    return out


def route_for(klass: str, item: TextureItem, preset: str = DEFAULT_PRESET) -> Route:
    if klass == "normal":
        if item.meta.get("tex") and item.meta.get("format") == "DXT5":
            return Route("normal-rg0-bc1", pre=mtf_ag_unpack, post=mtf_ag_pack)
        return Route("normal-rg0-bc1", post=renormalize)
    if klass == "font":
        return Route(None)
    if klass == "skip":
        return Route(None)
    # diffuse, material, ui
    mapping = PRESETS[preset]
    return Route(mapping[klass])
