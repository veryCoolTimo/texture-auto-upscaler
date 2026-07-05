import numpy as np
from PIL import Image
from typer.testing import CliRunner

from texup.cli import app
from texup.scan import scan_game

runner = CliRunner()


def _make_game(tmp_path):
    game = tmp_path / "game"
    (game / "ui").mkdir(parents=True)
    rgba = np.zeros((16, 16, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    Image.fromarray(rgba, "RGBA").save(game / "wall_n.png")
    Image.fromarray(rgba, "RGBA").save(game / "ui" / "button.png")
    (game / "readme.txt").write_text("not a texture")
    (game / "broken.png").write_bytes(b"garbage")
    return game


def test_scan_finds_and_classifies(tmp_path):
    game = _make_game(tmp_path)
    prj = scan_game(game, tmp_path / "out")
    recs = {r["key"]: r for r in prj.records()}
    assert len(recs) == 3  # 2 ok + 1 skipped(broken)
    normal = [r for r in recs.values() if r["klass"] == "normal"]
    assert len(normal) == 1
    skipped = prj.records(status="skipped")
    assert len(skipped) == 1 and "broken.png" in skipped[0]["key"]


def test_cli_scan_and_status(tmp_path):
    game = _make_game(tmp_path)
    out = tmp_path / "out"
    r1 = runner.invoke(app, ["scan", str(game), "--out", str(out)])
    assert r1.exit_code == 0, r1.output
    assert (out / "texup-project.json").exists()
    r2 = runner.invoke(app, ["status", str(out)])
    assert r2.exit_code == 0
    assert "pending" in r2.output


def test_rescan_backs_up_previous_manifest(tmp_path):
    game = _make_game(tmp_path)
    out = tmp_path / "out"
    scan_game(game, out)
    first = (out / "texup-project.json").read_text()
    scan_game(game, out)
    assert (out / "texup-project.json.bak").read_text() == first
