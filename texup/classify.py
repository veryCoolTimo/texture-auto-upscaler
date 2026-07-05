from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from texup.codecs.base import TextureItem

CLASSES = ("diffuse", "normal", "material", "ui", "font", "skip")

_NORMAL_RE = re.compile(r"(_n|_nm|_normal|_bump|_nrm)$", re.I)
_SPEC_RE = re.compile(r"(_s|_spec|_specular|_rough|_metal|_ao|_mask|_mm|_dm)$", re.I)
_DIFFUSE_RE = re.compile(r"(_d|_diff|_diffuse|_albedo|_bm|_col|_color)$", re.I)
_FONT_RE = re.compile(r"(font|glyph|ascii)", re.I)
_UI_PATH_RE = re.compile(r"(^|[/\\])(ui|hud|menu|icon|cursor|title|interface)([/\\]|_|$)", re.I)


@dataclass
class Classification:
    klass: str
    confidence: float


def _stem(item: TextureItem) -> str:
    name = item.inner_path if item.inner_path is not None else item.source_path.stem
    name = name.rsplit(".", 1)[0]
    return re.sub(r"_(NOMIP|HQ|LQ)$", "", name, flags=re.I)


def classify(item: TextureItem) -> Classification:
    votes: dict[str, float] = {k: 0.0 for k in CLASSES}
    stem = _stem(item)
    base = stem.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    full = f"{item.source_path}/{item.inner_path or ''}"

    # 1. Имя и путь — сильнейший сигнал
    if _FONT_RE.search(base):
        votes["font"] += 2.0
    if _NORMAL_RE.search(base):
        votes["normal"] += 2.0
    if _SPEC_RE.search(base):
        votes["material"] += 2.0
    if _DIFFUSE_RE.search(base):
        votes["diffuse"] += 2.0
    if _UI_PATH_RE.search(full):
        votes["ui"] += 1.5

    # 2. Формат
    if item.meta.get("format") == "BC5":
        votes["normal"] += 2.0

    # 3. Цветовая статистика
    px = item.pixels
    sample = px[:: max(1, px.shape[0] // 64), :: max(1, px.shape[1] // 64)]
    rgb = sample[..., :3].astype(np.int32)
    mean = rgb.reshape(-1, 3).mean(axis=0)
    if mean[2] > 190 and 100 < mean[0] < 160 and 100 < mean[1] < 160:
        votes["normal"] += 1.5
    channel_spread = np.abs(rgb[..., 0] - rgb[..., 1]).mean() + np.abs(rgb[..., 1] - rgb[..., 2]).mean()
    if channel_spread < 4:
        votes["material"] += 1.0

    # 4. Размер
    if item.width <= 64 and item.height <= 64 and _UI_PATH_RE.search(full):
        votes["ui"] += 0.5

    best = max(votes, key=votes.get)
    score = votes[best]
    if score <= 0.5:
        return Classification("diffuse", 0.3)
    confidence = min(1.0, score / 3.0)
    return Classification(best, confidence)
