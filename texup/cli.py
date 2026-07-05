from pathlib import Path

import typer

app = typer.Typer(help="Auto-remaster game textures.", no_args_is_help=True)


@app.command()
def version() -> None:
    """Print texup version."""
    from texup import __version__
    typer.echo(__version__)


@app.command()
def scan(game_dir: str, out: str = typer.Option(..., help="Output folder for manifest and results")):
    """Scan a game folder: find and classify textures, write manifest."""
    from texup.scan import scan_game

    prj = scan_game(Path(game_dir), Path(out))
    _print_summary(prj)


@app.command()
def status(out: str):
    """Show manifest progress summary."""
    from texup.project import Project

    _print_summary(Project.load(Path(out)))


@app.command()
def upscale(
    out: Path,
    only: str = typer.Option(None, help="Comma-separated classes, e.g. ui,diffuse"),
    sample: int = typer.Option(None, help="Process at most N textures per class"),
    max_size: int = typer.Option(4096, help="Max long side of the result"),
    compare: bool = typer.Option(False, help="Write side-by-side before/after PNGs"),
):
    """Upscale pending textures from the manifest in OUT dir."""
    from texup.pipeline import process
    from texup.project import Project

    prj = Project.load(out)
    only_list = [s.strip() for s in only.split(",")] if only else None
    stats = process(prj, out, only=only_list, sample=sample, max_size=max_size, compare=compare)
    typer.echo(f"done={stats['done']} failed={stats['failed']}")
    _print_summary(prj)


@app.command()
def preview(texture: Path, max_size: int = 4096):
    """Upscale a single texture file, write before/after PNGs next to it."""
    import numpy as np
    from PIL import Image

    from texup.classify import classify
    from texup.codecs import find_codec
    from texup.engine import load_upscaler
    from texup.router import resize_classic, route_for

    codec = find_codec(texture)
    if codec is None:
        typer.echo("no codec for this file", err=True)
        raise typer.Exit(1)
    for item in codec.decode(texture):
        c = classify(item)
        route = route_for(c.klass, item)
        px = route.pre(item.pixels) if route.pre else item.pixels
        if route.model is None:
            up = resize_classic(px, 4)
        else:
            up = load_upscaler(route.model).run(px, max_size=max_size)
        if route.post:
            up = route.post(up)
        inner = (item.inner_path or "").replace("/", "_").replace("\\", "_")
        suffix = f".{inner}" if inner else ""
        base = texture.with_suffix("")
        Image.fromarray(item.pixels, "RGBA").save(f"{base}{suffix}.before.png")
        Image.fromarray(up, "RGBA").save(f"{base}{suffix}.after.png")
        typer.echo(f"{item.key}: {c.klass} ({c.confidence:.2f}) {item.width}x{item.height} -> {up.shape[1]}x{up.shape[0]}")


def _print_summary(prj) -> None:
    from collections import Counter

    recs = prj.records()
    by_class = Counter(r["klass"] for r in recs)
    by_status = Counter(r["status"] for r in recs)
    typer.echo(f"textures: {len(recs)}")
    typer.echo("by class:  " + ", ".join(f"{k}={v}" for k, v in sorted(by_class.items())))
    typer.echo("by status: " + ", ".join(f"{k}={v}" for k, v in sorted(by_status.items())))


if __name__ == "__main__":
    app()
