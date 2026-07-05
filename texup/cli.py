import typer

app = typer.Typer(help="Auto-remaster game textures.", no_args_is_help=True)


@app.command()
def version() -> None:
    """Print texup version."""
    from texup import __version__
    typer.echo(__version__)


# Add a dummy command to force the app to be treated as a group
@app.command(hidden=True)
def _dummy() -> None:
    """Dummy command for internal use."""
    pass


if __name__ == "__main__":
    app()
