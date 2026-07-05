import pytest

from texup import models


def test_registry_has_mvp_models():
    assert models.MODELS["realesrgan-x4plus"].scale == 4
    assert models.MODELS["remacri"].scale == 4
    assert models.MODELS["normal-rg0-bc1"].scale == 4


def test_get_model_path_downloads_once(tmp_path, monkeypatch):
    calls = []

    def fake_urlretrieve(url, dst):
        calls.append(url)
        with open(dst, "wb") as f:
            f.write(b"fake model")

    monkeypatch.setattr(models, "_download", fake_urlretrieve)
    p1 = models.get_model_path("realesrgan-x4plus", cache_dir=tmp_path)
    p2 = models.get_model_path("realesrgan-x4plus", cache_dir=tmp_path)
    assert p1 == p2 and p1.exists()
    assert len(calls) == 1


def test_unknown_model():
    with pytest.raises(KeyError):
        models.get_model_path("nope")
