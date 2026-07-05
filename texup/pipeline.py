from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from texup.codecs import get_codec
from texup.engine import Upscaler, load_upscaler
from texup.project import Project
from texup.router import resize_classic, route_for


def _safe_name(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key)[-120:]


def _write_compare(out_dir: Path, klass: str, key: str, before: np.ndarray, after: np.ndarray) -> None:
    h, w = after.shape[:2]
    b = np.asarray(Image.fromarray(before, "RGBA").resize((w, h), Image.NEAREST))
    canvas = np.zeros((h, w * 2 + 8, 4), dtype=np.uint8)
    canvas[:, :w] = b
    canvas[:, w + 8 :] = after
    dst = out_dir / "_compare" / klass / f"{_safe_name(key)}.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, "RGBA").save(dst)


def _cache_path(cache_dir: Path, content_sha: str, model: str | None, max_size: int) -> Path:
    return cache_dir / f"{content_sha}-{model or 'classic'}-{max_size}.png"


def process(prj: Project, out_dir: Path, *, only: list[str] | None = None,
            sample: int | None = None, max_size: int = 4096,
            engine_factory: Callable[[str], Upscaler] = load_upscaler,
            compare: bool = False) -> dict:
    out_dir = Path(out_dir)
    cache_dir = out_dir / "_upcache"
    pending = prj.records(status="pending")
    if only:
        pending = [r for r in pending if r["klass"] in only]
    if sample is not None:
        by_class: dict[str, list[dict]] = defaultdict(list)
        for r in sorted(pending, key=lambda r: r["key"]):
            if len(by_class[r["klass"]]) < sample:
                by_class[r["klass"]].append(r)
        pending = [r for rs in by_class.values() for r in rs]

    engines: dict[str, Upscaler] = {}
    stats = {"done": 0, "failed": 0, "skipped": 0}

    by_source: dict[Path, list[dict]] = defaultdict(list)
    for r in pending:
        src, _ = Project.source_of(r["key"])
        by_source[src].append(r)

    for src, recs in sorted(by_source.items()):
        replacements: dict[str, np.ndarray] = {}
        codec_name = recs[0]["codec"]
        try:
            codec = get_codec(codec_name)
            items = {it.inner_path or "": it for it in codec.decode(src)}
        except Exception as e:  # noqa: BLE001
            for r in recs:
                prj.set_status(r["key"], "failed", reason=f"decode: {e}")
                stats["failed"] += 1
            prj.save()
            continue

        for r in recs:
            _, inner = Project.source_of(r["key"])
            try:
                item = items[inner]
                route = route_for(r["klass"], item)
                content_sha = item.meta.get("content_sha")
                cache_file = (
                    _cache_path(cache_dir, content_sha, route.model, max_size)
                    if content_sha else None
                )
                if cache_file is not None and cache_file.exists():
                    up = np.asarray(Image.open(cache_file).convert("RGBA"))
                else:
                    px = item.pixels
                    if route.pre:
                        px = route.pre(px)
                    if route.model is None:
                        up = resize_classic(px, 4)
                    else:
                        if route.model not in engines:
                            engines[route.model] = engine_factory(route.model)
                        up = engines[route.model].run(px, max_size=max_size)
                    if route.post:
                        up = route.post(up)
                    if cache_file is not None:
                        cache_file.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(up, "RGBA").save(cache_file)
                if compare:
                    _write_compare(out_dir, r["klass"], r["key"], item.pixels, up)
                replacements[inner] = up
                prj.set_status(r["key"], "done", model=route.model)
                stats["done"] += 1
            except Exception as e:  # noqa: BLE001
                prj.set_status(r["key"], "failed", reason=str(e))
                stats["failed"] += 1

        if replacements:
            try:
                blob = codec.encode_file(src, replacements)
                rel = src.relative_to(prj.game_dir)
                dst = out_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(blob)
            except Exception as e:  # noqa: BLE001
                for r in recs:
                    if r["status"] == "done":
                        prj.set_status(r["key"], "failed", reason=f"encode: {e}")
                        stats["done"] -= 1
                        stats["failed"] += 1
        prj.save()
    return stats
