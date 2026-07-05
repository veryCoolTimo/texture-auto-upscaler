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
