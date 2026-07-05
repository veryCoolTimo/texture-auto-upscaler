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
