from texup.eta import estimate_seconds, unique_pending_mpx
from texup.project import Project


def _prj(tmp_path):
    game, out = tmp_path / "g", tmp_path / "o"
    game.mkdir()
    prj = Project.create(game, out)
    for i, (key, klass, sha, w) in enumerate([
        ("a.png", "diffuse", "sha1", 1024),
        ("b.png", "diffuse", "sha1", 1024),   # дубликат по содержимому
        ("c.png", "normal", "sha2", 512),
        ("d.png", "font", "sha3", 256),        # без модели — не в ETA
    ]):
        prj.add_texture(key, codec="standard", klass=klass, confidence=1.0,
                        sha256="f", width=w, height=w, fmt="PNG")
        prj._textures[key]["content_sha"] = sha
    return prj


def test_unique_pending_mpx_dedupes(tmp_path):
    mpx = unique_pending_mpx(_prj(tmp_path))
    assert abs(mpx["diffuse"] - 1024 * 1024 / 1e6) < 1e-6  # один из двух
    assert abs(mpx["normal"] - 512 * 512 / 1e6) < 1e-6


def test_estimate_seconds(tmp_path):
    bench_data = {"rates": {"remacri": 2.0, "normal-rg0-bc1": 1.0}}
    sec = estimate_seconds(_prj(tmp_path), "detailed", bench_data)
    expected = (1024 * 1024 / 1e6 / 2.0 + 512 * 512 / 1e6 / 1.0) * 1.10
    assert abs(sec - expected) < 1e-6


def test_estimate_none_when_model_missing(tmp_path):
    assert estimate_seconds(_prj(tmp_path), "detailed", {"rates": {}}) is None
