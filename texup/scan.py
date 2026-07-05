from __future__ import annotations

import hashlib
from pathlib import Path

from texup.classify import classify
from texup.codecs import find_codec
from texup.codecs.base import UnsupportedTexture
from texup.project import Project


def scan_game(game_dir: Path, out_dir: Path) -> Project:
    game_dir = Path(game_dir)
    prj = Project.create(game_dir, out_dir)
    for path in sorted(game_dir.rglob("*")):
        if not path.is_file():
            continue
        codec = find_codec(path)
        if codec is None:
            continue
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            items = codec.decode(path)
        except Exception as e:  # noqa: BLE001  # UnsupportedTexture, битые файлы и т.п.
            prj.add_texture(str(path), codec=codec.name, klass="skip", confidence=1.0,
                            sha256=sha, width=0, height=0, fmt="?")
            prj.set_status(str(path), "skipped", reason=str(e))
            continue
        for item in items:
            c = classify(item)
            prj.add_texture(item.key, codec=codec.name, klass=c.klass,
                            confidence=c.confidence, sha256=sha,
                            width=item.width, height=item.height,
                            fmt=item.meta.get("format", "?"))
    prj.save()
    return prj
