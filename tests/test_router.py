from pathlib import Path

import numpy as np

from texup.codecs.base import TextureItem
from texup.router import mtf_ag_pack, mtf_ag_unpack, renormalize, resize_classic, route_for


def _item(meta=None):
    px = np.zeros((8, 8, 4), dtype=np.uint8)
    return TextureItem(Path("x.dds"), None, "dds", px, meta or {})


def test_routes_defined_for_all_classes():
    from texup.classify import CLASSES
    for k in CLASSES:
        r = route_for(k, _item())
        assert r is not None


def test_font_has_no_model():
    assert route_for("font", _item()).model is None


def test_diffuse_uses_remacri():
    assert route_for("diffuse", _item()).model == "remacri"


def test_normal_uses_normal_model():
    assert route_for("normal", _item()).model == "normal-rg0-bc1"


def test_resize_classic():
    px = np.zeros((4, 4, 4), dtype=np.uint8)
    assert resize_classic(px, 4).shape == (16, 16, 4)


def test_renormalize_unit_length():
    px = np.zeros((2, 2, 4), dtype=np.uint8)
    px[..., 0] = 128  # x ~ 0
    px[..., 1] = 128  # y ~ 0
    out = renormalize(px)
    assert np.all(out[..., 2] >= 250)  # z ~ 1 -> B ~ 255


def test_mtf_ag_swizzle_roundtrip():
    rng = np.random.default_rng(0)
    px = rng.integers(0, 255, (4, 4, 4), dtype=np.uint8)
    unpacked = mtf_ag_unpack(px)
    assert np.array_equal(unpacked[..., 0], px[..., 3])  # X из альфы
    assert np.array_equal(unpacked[..., 1], px[..., 1])  # Y из G
    assert np.all(unpacked[..., 2] == 0)                 # B обнулён
    assert np.all(unpacked[..., 3] == 255)               # A константный
    packed = mtf_ag_pack(unpacked)
    assert np.all(packed[..., 0] == 255)
    assert np.array_equal(packed[..., 1], unpacked[..., 1])  # Y в G
    assert np.all(packed[..., 2] == 255)
    assert np.array_equal(packed[..., 3], unpacked[..., 0])


def test_mtf_dxt5_normal_gets_swizzle_route():
    item = _item({"tex": True, "format": "DXT5"})
    r = route_for("normal", item)
    assert r.pre is mtf_ag_unpack and r.post is mtf_ag_pack


def test_normal_default_route_has_renormalize_post():
    r = route_for("normal", _item())
    assert r.post is renormalize and r.pre is None
