from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

CACHE_DIR = Path.home() / ".cache" / "texup" / "models"


@dataclass(frozen=True)
class ModelSpec:
    url: str
    filename: str
    scale: int
    sha256: str | None = None


MODELS: dict[str, ModelSpec] = {
    "realesrgan-x4plus": ModelSpec(
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        filename="RealESRGAN_x4plus.pth",
        scale=4,
    ),
    "remacri": ModelSpec(
        url="https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth",
        filename="4x_foolhardy_Remacri.pth",
        scale=4,
    ),
    "normal-rg0-bc1": ModelSpec(
        url="https://github.com/RunDevelopment/ESRGAN-models/raw/main/normals/4x-Normal-RG0-BC1.pth",
        filename="4x-Normal-RG0-BC1.pth",
        scale=4,
    ),
}


def _download(url: str, dst: Path) -> None:
    urllib.request.urlretrieve(url, dst)  # noqa: S310


def get_model_path(name: str, cache_dir: Path | None = None) -> Path:
    spec = MODELS[name]
    cache = Path(cache_dir) if cache_dir else CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    dst = cache / spec.filename
    if not dst.exists():
        tmp = dst.with_suffix(".part")
        _download(spec.url, tmp)
        tmp.rename(dst)
    return dst
