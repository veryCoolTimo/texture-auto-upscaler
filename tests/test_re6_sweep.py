import glob
import os
from pathlib import Path

import pytest

RE6 = os.environ.get("TEXUP_RE6_DIR")

pytestmark = pytest.mark.skipif(not RE6, reason="TEXUP_RE6_DIR not set")


def test_re6_full_sweep():
    from texup.codecs.mtframework import MtfArcCodec

    codec = MtfArcCodec()
    parsed = failed = repack_ok = repack_diff = 0
    for ap in sorted(glob.glob(os.path.join(RE6, "**", "*.arc"), recursive=True)):
        p = Path(ap)
        data = p.read_bytes()
        if data[:4] != b"ARC\x00":
            continue
        try:
            items = codec.decode(p)
            parsed += len(items)
        except Exception as e:  # noqa: BLE001
            failed += 1
            continue
        # репак без замен байт-идентичен
        if codec.encode_file(p, {}) == data:
            repack_ok += 1
        else:
            repack_diff += 1
    print(f"RE6 sweep: parsed_textures={parsed} arcs_failed={failed} "
          f"repack identical={repack_ok} diff={repack_diff}")
    assert failed == 0
    assert repack_diff == 0
    assert parsed > 1000  # защита от пустого/неверного TEXUP_RE6_DIR
