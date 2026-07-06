from __future__ import annotations

import re
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from texup.codecs import get_codec
from texup.engine import Upscaler, load_upscaler
from texup.presets import DEFAULT_PRESET
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


# Small same-size textures sharing a model get stacked into one GPU forward pass.
BATCH = 8


def _finalize_source(codec, src: Path, game_dir: Path, out_dir: Path,
                      replacements: dict[str, np.ndarray], provisional: list[dict]) -> None:
    """Runs on the background worker thread: encode + write output + cache PNG
    writes + compare sheets for one source. Must not touch the manifest."""
    if getattr(codec, "loose_output", False):
        # Read-only containers (e.g. VPK): write each replaced entry as its own
        # loose file next to the container instead of repacking the whole file.
        rel_dir = src.parent.relative_to(game_dir)
        for inner, rgba in replacements.items():
            orig_bytes = codec.read_inner(src, inner)
            blob = codec.encode_inner(inner, orig_bytes, rgba)
            target = out_dir / rel_dir / codec.loose_target(inner)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
    else:
        blob = codec.encode_file(src, replacements)
        rel = src.relative_to(game_dir)
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(blob)
    for p in provisional:
        cache_file = p["cache_file"]
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(p["up"], "RGBA").save(cache_file, compress_level=1)
        if p["compare"]:
            _write_compare(out_dir, p["klass"], p["key"], p["before"], p["up"])


def process(prj: Project, out_dir: Path, *, only: list[str] | None = None,
            sample: int | None = None, max_size: int = 4096,
            engine_factory: Callable[[str], Upscaler] = load_upscaler,
            compare: bool = False, preset: str = DEFAULT_PRESET,
            on_texture: Callable[[dict], None] | None = None,
            compare_limit: int | None = None) -> dict:
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

    total_planned = len(pending)
    engines: dict[str, Upscaler] = {}
    stats = {"done": 0, "failed": 0, "skipped": 0}
    mem_cache: dict[Path, np.ndarray] = {}
    compare_counts: dict[str, int] = defaultdict(int)

    def _emit(cache_hit: bool) -> None:
        if on_texture is not None:
            on_texture({"done": stats["done"], "failed": stats["failed"],
                        "total": total_planned, "cache_hit": cache_hit})

    by_source: dict[Path, list[dict]] = defaultdict(list)
    for r in pending:
        src, _ = Project.source_of(r["key"])
        by_source[src].append(r)

    def resolve_pending(pending_finalize: tuple[Future, list[dict]] | None) -> None:
        if pending_finalize is None:
            return
        future, provisional = pending_finalize
        try:
            future.result()
        except Exception as e:  # noqa: BLE001
            for p in provisional:
                prj.set_status(p["key"], "failed", reason=f"encode: {e}")
                stats["failed"] += 1
                _emit(p["cache_hit"])
        else:
            for p in provisional:
                prj.set_status(p["key"], "done", model=p["model"])
                stats["done"] += 1
                _emit(p["cache_hit"])
        prj.save()

    pending_finalize: tuple[Future, list[dict]] | None = None
    with ThreadPoolExecutor(max_workers=1) as executor:
        for src, recs in sorted(by_source.items()):
            codec_name = recs[0]["codec"]
            try:
                codec = get_codec(codec_name)
                items = {it.inner_path or "": it for it in codec.decode(src)}
            except Exception as e:  # noqa: BLE001
                for r in recs:
                    prj.set_status(r["key"], "failed", reason=f"decode: {e}")
                    stats["failed"] += 1
                    _emit(False)
                prj.save()
                continue

            replacements: dict[str, np.ndarray] = {}
            provisional: list[dict] = []

            def _record_result(r: dict, inner: str, item, route, up: np.ndarray,
                                cache_file: Path | None, fresh: bool, cache_hit: bool) -> None:
                if fresh and cache_file is not None:
                    mem_cache[cache_file] = up
                replacements[inner] = up
                klass = r["klass"]
                write_compare = False
                if compare and not cache_hit and (compare_limit is None or compare_counts[klass] < compare_limit):
                    write_compare = True
                    compare_counts[klass] += 1
                provisional.append({
                    "key": r["key"],
                    "model": route.model,
                    "klass": klass,
                    "up": up,
                    "before": item.pixels if write_compare else None,
                    "cache_file": cache_file if fresh else None,
                    "compare": write_compare,
                    "cache_hit": cache_hit,
                })

            # Items eligible for batching (same model, fit in one tile, no cache hit)
            # are deferred here and processed together after this pass.
            batch_groups: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
            for r in recs:
                _, inner = Project.source_of(r["key"])
                try:
                    item = items[inner]
                    route = route_for(r["klass"], item, preset)
                    content_sha = item.meta.get("content_sha")
                    cache_file = (
                        _cache_path(cache_dir, content_sha, route.model, max_size)
                        if content_sha else None
                    )
                    if cache_file is not None and cache_file in mem_cache:
                        _record_result(r, inner, item, route, mem_cache[cache_file], cache_file, False, True)
                    elif cache_file is not None and cache_file.exists():
                        up = np.asarray(Image.open(cache_file).convert("RGBA"))
                        mem_cache[cache_file] = up
                        _record_result(r, inner, item, route, up, cache_file, False, True)
                    elif route.model is None:
                        up = resize_classic(item.pixels, 4)
                        _record_result(r, inner, item, route, up, cache_file, cache_file is not None, False)
                    else:
                        if route.model not in engines:
                            engines[route.model] = engine_factory(route.model)
                        engine = engines[route.model]
                        px = route.pre(item.pixels) if route.pre else item.pixels
                        h, w = px.shape[:2]
                        if max(h, w) <= engine.tile_size:
                            batch_groups[(route.model, w, h)].append({
                                "r": r, "inner": inner, "item": item, "route": route,
                                "px": px, "cache_file": cache_file,
                            })
                        else:
                            up = engine.run(px, max_size=max_size)
                            if route.post:
                                up = route.post(up)
                            _record_result(r, inner, item, route, up, cache_file, cache_file is not None, False)
                except Exception as e:  # noqa: BLE001
                    prj.set_status(r["key"], "failed", reason=str(e))
                    stats["failed"] += 1
                    _emit(False)

            for key in sorted(batch_groups):
                entries = batch_groups[key]
                engine = engines[key[0]]
                for i in range(0, len(entries), BATCH):
                    chunk = entries[i : i + BATCH]
                    try:
                        ups = engine.run_batch([c["px"] for c in chunk], max_size=max_size)
                    except Exception as e:  # noqa: BLE001
                        for c in chunk:
                            prj.set_status(c["r"]["key"], "failed", reason=str(e))
                            stats["failed"] += 1
                            _emit(False)
                        continue
                    for c, up in zip(chunk, ups):
                        try:
                            route = c["route"]
                            if route.post:
                                up = route.post(up)
                            _record_result(
                                c["r"], c["inner"], c["item"], route, up,
                                c["cache_file"], c["cache_file"] is not None, False,
                            )
                        except Exception as e:  # noqa: BLE001
                            prj.set_status(c["r"]["key"], "failed", reason=str(e))
                            stats["failed"] += 1
                            _emit(False)

            if replacements:
                # Surface the previous source's finalize errors before handing off
                # new background work (single-worker executor keeps ordering).
                resolve_pending(pending_finalize)
                future = executor.submit(
                    _finalize_source, codec, src, prj.game_dir, out_dir,
                    replacements, provisional,
                )
                pending_finalize = (future, provisional)
            else:
                prj.save()

        resolve_pending(pending_finalize)
    return stats
