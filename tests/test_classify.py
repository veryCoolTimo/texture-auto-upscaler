from pathlib import Path

import numpy as np

from texup.classify import classify
from texup.codecs.base import TextureItem


def _item(name, rgba, meta=None, inner=None):
    return TextureItem(Path(name), inner, "dds", rgba, meta or {})


def _flat(color, w=64, h=64):
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:] = color
    return rgba


def test_normal_by_suffix():
    c = classify(_item("wall_n.dds", _flat((128, 128, 255, 255))))
    assert c.klass == "normal" and c.confidence > 0.8


def test_normal_by_color_stats():
    c = classify(_item("wall.dds", _flat((120, 135, 240, 255))))
    assert c.klass == "normal"


def test_re5_suffixes():
    assert classify(_item("a.arc", _flat((100, 90, 80, 255)), inner="em01_Body_BM")).klass == "diffuse"
    assert classify(_item("a.arc", _flat((128, 128, 255, 255)), inner="em01_Body_NM")).klass == "normal"
    assert classify(_item("a.arc", _flat((128, 128, 128, 255)), inner="em01_Body_MM")).klass == "material"


def test_ui_by_path():
    c = classify(_item("game/ui/button.dds", _flat((10, 200, 30, 255))))
    assert c.klass == "ui"


def test_font_by_name():
    assert classify(_item("font_ascii.dds", _flat((255, 255, 255, 128)))).klass == "font"


def test_grayscale_is_material():
    assert classify(_item("thing.dds", _flat((77, 77, 77, 255)))).klass == "material"


def test_bc5_format_is_normal():
    c = classify(_item("x.dds", _flat((128, 128, 0, 255)), meta={"format": "BC5"}))
    assert c.klass == "normal"


def test_default_diffuse_low_confidence():
    rgba = np.random.default_rng(0).integers(0, 255, (64, 64, 4), dtype=np.uint8)
    c = classify(_item("something.dds", rgba))
    assert c.klass == "diffuse"
