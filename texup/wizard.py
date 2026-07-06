from __future__ import annotations

from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.panel import Panel
from rich.progress import (BarColumn, Progress, TextColumn,
                            TimeElapsedColumn, TimeRemainingColumn)
from rich.prompt import Prompt

from texup.apply import apply_to_game
from texup.bench import load_bench, run_bench
from texup.engine import load_upscaler
from texup.eta import estimate_seconds
from texup.pipeline import process
from texup.presets import PRESETS
from texup.project import MANIFEST_NAME, Project
from texup.scan import scan_game

console = Console()

_ENGINE_NAMES = {"mtf-arc": "MT Framework", "mtf-tex": "MT Framework",
                  "zip": "ZIP/id-Tech", "dds": "loose files", "standard": "loose files"}


def rich_ask(question: str, choices: list[str], default: str) -> str:
    return Prompt.ask(question, choices=choices, default=default)


def _fmt_eta(sec: float | None) -> str:
    if sec is None:
        return "no calibration"
    if sec < 90:
        return f"~{int(sec)} s"
    if sec < 5400:
        return f"~{sec / 60:.0f} min"
    return f"~{sec / 3600:.1f} h"


def run_remaster(game_dir: Path, out_dir: Path | None, *,
                  ask: Callable[[str, list[str], str], str] = rich_ask,
                  engine_factory=load_upscaler,
                  bench_runner=run_bench) -> int:
    game_dir = Path(game_dir)
    out_dir = Path(out_dir) if out_dir else Path.cwd() / f"texup-out-{game_dir.name}"

    # 1. scan (or load existing manifest)
    if (out_dir / MANIFEST_NAME).exists():
        prj = Project.load(out_dir)
    else:
        console.print("[bold]Scanning game...[/bold]")
        prj = scan_game(game_dir, out_dir)

    # 2. summary
    recs = prj.records()
    engines = sorted({_ENGINE_NAMES.get(r["codec"], r["codec"]) for r in recs})
    by_class: dict[str, int] = {}
    shas = set()
    dups = 0
    for r in recs:
        by_class[r["klass"]] = by_class.get(r["klass"], 0) + 1
        sha = r.get("content_sha") or r["key"]
        if sha in shas:
            dups += 1
        shas.add(sha)
    lines = [f"Engine: {', '.join(engines) or '—'}",
             f"Textures: {len(recs)}  (duplicates: {dups})",
             "  " + ", ".join(f"{k}={v}" for k, v in sorted(by_class.items()))]
    console.print(Panel("\n".join(lines), title=f"texup · {game_dir.name}"))

    # 3. questions (only on first run — skipped once answers are persisted)
    answers = dict(prj.wizard)
    if "preset" not in answers:
        bench_data = load_bench()
        if bench_data is None:
            console.print("[dim]No hardware calibration found — calibrating now (~1 min)...[/dim]")
            bench_data = bench_runner()
        for name in PRESETS:
            eta = estimate_seconds(prj, name, bench_data) if bench_data else None
            console.print(f"  [bold]{name}[/bold]: {_fmt_eta(eta)}")
        answers["preset"] = ask("Quality mode", list(PRESETS), "detailed")
        answers["apply_mode"] = ask(
            "Write into the game (game, with backup) or output folder only (folder)?",
            ["game", "folder"], "folder")
        prj.set_wizard(answers)

    # 4. run
    pending = prj.records(status="pending")
    if not pending:
        console.print("[green]Everything already processed.[/green]")
    else:
        with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(),
                       TextColumn("{task.completed}/{task.total}"),
                       TimeElapsedColumn(), TimeRemainingColumn(), console=console) as prog:
            task = prog.add_task("Processing", total=len(pending))

            def on_texture(ev: dict) -> None:
                prog.update(task, completed=ev["done"] + ev["failed"])

            stats = process(prj, out_dir, engine_factory=engine_factory,
                             preset=answers["preset"], compare=True, compare_limit=5,
                             on_texture=on_texture)
        console.print(f"done={stats['done']} failed={stats['failed']}")

    # 5. apply
    if answers.get("apply_mode") == "game":
        st = apply_to_game(out_dir)
        console.print(f"Applied to game: {st['applied']} (backup: {game_dir / '.texup-backup'})")
        console.print(f'Rollback anytime: texup rollback "{game_dir}"')
    console.print(f"Comparison sheets: {out_dir / '_compare'}")
    return 0
