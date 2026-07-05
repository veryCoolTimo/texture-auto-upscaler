from texup.codecs.base import (  # noqa: F401
    Codec,
    TextureItem,
    UnsupportedTexture,
    find_codec,
    get_codec,
    register,
)
from texup.codecs.standard import StandardCodec  # noqa: E402

register(StandardCodec())

from texup.codecs.dds import DdsCodec  # noqa: E402

register(DdsCodec())

from texup.codecs.mtframework import MtfTexCodec  # noqa: E402

register(MtfTexCodec())

from texup.codecs.mtframework import MtfArcCodec  # noqa: E402

register(MtfArcCodec())
