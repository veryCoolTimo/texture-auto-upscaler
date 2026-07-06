from __future__ import annotations

from texup.presets import PRESETS
from texup.project import Project

_OVERHEAD = 1.10


def unique_pending_mpx(prj: Project) -> dict[str, float]:
    seen: set[str] = set()
    out: dict[str, float] = {}
    for r in prj.records(status="pending"):
        sha = r.get("content_sha") or r["key"]
        if sha in seen:
            continue
        seen.add(sha)
        out[r["klass"]] = out.get(r["klass"], 0.0) + r["width"] * r["height"] / 1e6
    return out


def estimate_seconds(prj: Project, preset: str, bench_data: dict) -> float | None:
    rates = bench_data.get("rates", {})
    mapping = dict(PRESETS[preset])
    mapping["normal"] = "normal-rg0-bc1"
    total = 0.0
    for klass, mpx in unique_pending_mpx(prj).items():
        model = mapping.get(klass)
        if model is None:
            continue  # font/skip — без нейросети, пренебрежимо
        if model not in rates or rates[model] <= 0:
            return None
        total += mpx / rates[model]
    return total * _OVERHEAD
