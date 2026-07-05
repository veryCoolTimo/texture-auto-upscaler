from pathlib import Path

import numpy as np
import pytest

from texup.codecs.base import TextureItem, find_codec, get_codec, register


class DummyCodec:
    name = "dummy"

    def detect(self, path: Path) -> bool:
        return path.suffix == ".dum"

    def decode(self, path: Path):
        return []

    def encode_file(self, path: Path, replacements):
        return b""


def test_item_key_loose_and_archived():
    px = np.zeros((4, 4, 4), dtype=np.uint8)
    loose = TextureItem(Path("/g/a.png"), None, "standard", px, {})
    packed = TextureItem(Path("/g/x.arc"), "inner/t.tex", "mtf-arc", px, {})
    assert loose.key == "/g/a.png"
    assert packed.key == "/g/x.arc::inner/t.tex"
    assert loose.width == 4 and loose.height == 4


def test_registry_detect_and_get():
    codec = DummyCodec()
    register(codec)
    assert find_codec(Path("f.dum")) is codec
    assert find_codec(Path("f.xyz")) is None
    assert get_codec("dummy") is codec
    with pytest.raises(KeyError):
        get_codec("nope")
