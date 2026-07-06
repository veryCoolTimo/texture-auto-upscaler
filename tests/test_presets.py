from pathlib import Path

import numpy as np

from texup.codecs.base import TextureItem
from texup.presets import DEFAULT_PRESET, PRESETS
from texup.router import route_for


def _item(meta=None):
    return TextureItem(Path("x.dds"), None, "dds", np.zeros((8, 8, 4), np.uint8), meta or {})


def test_presets_shape():
    assert set(PRESETS) == {"faithful", "detailed"}
    for mapping in PRESETS.values():
        assert set(mapping) == {"diffuse", "material", "ui"}
    assert DEFAULT_PRESET == "detailed"


def test_faithful_routes_realesrgan():
    r = route_for("diffuse", _item(), preset="faithful")
    assert r.model == "realesrgan-x4plus"


def test_detailed_routes_remacri_default():
    assert route_for("diffuse", _item()).model == "remacri"
    assert route_for("material", _item(), preset="detailed").model == "remacri"


def test_normal_and_font_ignore_preset():
    assert route_for("normal", _item(), preset="faithful").model == "normal-rg0-bc1"
    assert route_for("font", _item(), preset="faithful").model is None


def test_unknown_preset_raises():
    import pytest
    with pytest.raises(KeyError):
        route_for("diffuse", _item(), preset="nope")
