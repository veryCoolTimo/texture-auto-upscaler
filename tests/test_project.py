from pathlib import Path

from texup.project import Project


def test_create_add_save_load(tmp_path):
    game, out = tmp_path / "game", tmp_path / "out"
    game.mkdir()
    prj = Project.create(game, out)
    prj.add_texture("g/a.png", codec="standard", klass="diffuse", confidence=0.9,
                    sha256="ab", width=64, height=64, fmt="PNG")
    prj.add_texture("g/x.arc::inner/t", codec="mtf-arc", klass="normal", confidence=0.7,
                    sha256="cd", width=32, height=32, fmt="DXT5")
    prj.save()

    prj2 = Project.load(out)
    assert prj2.game_dir == game
    assert len(prj2.records()) == 2
    assert len(prj2.records(klass="normal")) == 1
    assert prj2.records()[0]["status"] == "pending"


def test_set_status_and_filter(tmp_path):
    game, out = tmp_path / "game", tmp_path / "out"
    game.mkdir()
    prj = Project.create(game, out)
    prj.add_texture("k1", codec="standard", klass="ui", confidence=1.0,
                    sha256="x", width=8, height=8, fmt="PNG")
    prj.set_status("k1", "failed", reason="boom")
    assert prj.records(status="failed")[0]["reason"] == "boom"
    assert not prj.records(status="pending")


def test_source_of():
    assert Project.source_of("g/a.png") == (Path("g/a.png"), "")
    assert Project.source_of("g/x.arc::in/t") == (Path("g/x.arc"), "in/t")
