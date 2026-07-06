from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from texup.engine import Upscaler, load_upscaler, pick_device
from texup.presets import PRESETS

BENCH_DIR = Path.home() / ".cache" / "texup"
_SIZES = [(256, 256), (512, 512), (512, 512)]  # первый — прогрев, не считается


def default_models() -> list[str]:
    models = {m for mapping in PRESETS.values() for m in mapping.values()}
    models.add("normal-rg0-bc1")
    return sorted(models)


def _bench_path(cache_dir: Path | None) -> Path:
    return (Path(cache_dir) if cache_dir else BENCH_DIR) / "bench.json"


def run_bench(engine_factory: Callable[[str], Upscaler] = load_upscaler,
              models: list[str] | None = None,
              cache_dir: Path | None = None) -> dict:
    rng = np.random.default_rng(42)
    rates: dict[str, float] = {}
    for name in models or default_models():
        engine = engine_factory(name)
        mpx = 0.0
        elapsed = 0.0
        for i, (w, h) in enumerate(_SIZES):
            rgba = rng.integers(0, 255, (h, w, 4), dtype=np.uint8)
            rgba[..., 3] = 255
            t0 = time.perf_counter()
            engine.run(rgba, max_size=4096)
            dt = time.perf_counter() - t0
            if i > 0:  # без прогрева
                mpx += w * h / 1e6
                elapsed += dt
        rates[name] = round(mpx / elapsed, 4) if elapsed else 0.0
    data = {"device": pick_device(), "rates": rates,
            "measured_at": datetime.now(timezone.utc).isoformat()}
    path = _bench_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)
    return data


def load_bench(cache_dir: Path | None = None) -> dict | None:
    path = _bench_path(cache_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
