from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from texup.project import Project

BACKUP_DIR = ".texup-backup"


def _manifest_hashes(prj: Project) -> dict[Path, str]:
    hashes: dict[Path, str] = {}
    for r in prj.records():
        src, _ = Project.source_of(r["key"])
        hashes[src] = r["sha256"]
    return hashes


def apply_to_game(out_dir: Path, *, force: bool = False) -> dict:
    out_dir = Path(out_dir)
    prj = Project.load(out_dir)
    game_dir = prj.game_dir
    hashes = _manifest_hashes(prj)
    stats = {"applied": 0, "skipped": 0}
    for new_file in sorted(out_dir.rglob("*")):
        if not new_file.is_file():
            continue
        rel = new_file.relative_to(out_dir)
        if rel.parts[0] in ("_compare",) or rel.name == "texup-project.json":
            continue
        target = game_dir / rel
        if not target.exists():
            stats["skipped"] += 1
            continue
        expected = hashes.get(target)
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        backup = game_dir / BACKUP_DIR / rel
        if expected != actual and not backup.exists() and not force:
            print(f"skip (game file changed since scan): {rel}")
            stats["skipped"] += 1
            continue
        if not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup)
        shutil.copy2(new_file, target)
        stats["applied"] += 1
    return stats


def rollback_game(game_dir: Path) -> int:
    game_dir = Path(game_dir)
    backup_root = game_dir / BACKUP_DIR
    count = 0
    for b in sorted(backup_root.rglob("*")):
        if not b.is_file():
            continue
        rel = b.relative_to(backup_root)
        shutil.copy2(b, game_dir / rel)
        count += 1
    return count
