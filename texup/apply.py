from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from texup.project import Project

BACKUP_DIR = ".texup-backup"
APPLIED_LEDGER_NAME = "applied.json"


def _is_contained(base: Path, target: Path) -> bool:
    """True if `target` resolves to a path under `base`. Guards against a
    poisoned applied-file ledger (rel keys are persisted JSON, not re-derived
    from a filesystem walk on every use) steering a copy/delete outside the
    game directory."""
    base_root = base.resolve()
    return os.path.commonpath([base_root, target.resolve()]) == str(base_root)


def _manifest_hashes(prj: Project) -> dict[Path, str]:
    hashes: dict[Path, str] = {}
    for r in prj.records():
        src, _ = Project.source_of(r["key"])
        hashes[src] = r["sha256"]
    return hashes


def _ledger_path(game_dir: Path) -> Path:
    return game_dir / BACKUP_DIR / APPLIED_LEDGER_NAME


def _load_applied_ledger(game_dir: Path) -> dict[str, str]:
    path = _ledger_path(game_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_applied_ledger(game_dir: Path, ledger: dict[str, str]) -> None:
    path = _ledger_path(game_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(ledger, indent=1))
    os.replace(tmp, path)


def apply_to_game(out_dir: Path, *, force: bool = False) -> dict:
    out_dir = Path(out_dir)
    prj = Project.load(out_dir)
    game_dir = prj.game_dir
    hashes = _manifest_hashes(prj)
    applied_ledger = _load_applied_ledger(game_dir)
    created = set(applied_ledger.get("created", []))
    stats = {"applied": 0, "skipped": 0}
    for new_file in sorted(out_dir.rglob("*")):
        if not new_file.is_file():
            continue
        rel = new_file.relative_to(out_dir)
        if rel.parts[0] in ("_compare", "_upcache") or rel.name == "texup-project.json" \
                or rel.name.endswith(".json.bak"):
            continue
        target = game_dir / rel
        rel_key = rel.as_posix()
        is_created_loose = rel_key in created
        if not target.exists():
            if target in hashes and not is_created_loose:
                # Used to exist in the game per the scan manifest but is gone now —
                # nothing to safely overwrite/backup; leave it to a rescan.
                stats["skipped"] += 1
                continue
            # Brand-new loose file (e.g. a VPK texture written out-of-container):
            # there is no original in the game dir, so there's nothing to back up.
            if not _is_contained(game_dir, target):
                print(f"skip (unsafe target escapes game dir): {rel}")
                stats["skipped"] += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(new_file, target)
            created.add(rel_key)
            applied_ledger[rel_key] = hashlib.sha256(new_file.read_bytes()).hexdigest()
            stats["applied"] += 1
            continue
        expected = hashes.get(target)
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        allowed = actual == expected or actual == applied_ledger.get(rel_key) or force or is_created_loose
        if not allowed:
            print(f"skip (game file changed since scan): {rel}")
            stats["skipped"] += 1
            continue
        if not is_created_loose:
            backup = game_dir / BACKUP_DIR / rel
            if not backup.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup)
        shutil.copy2(new_file, target)
        applied_ledger[rel_key] = hashlib.sha256(new_file.read_bytes()).hexdigest()
        stats["applied"] += 1
    applied_ledger["created"] = sorted(created)
    _save_applied_ledger(game_dir, applied_ledger)
    return stats


def rollback_game(game_dir: Path) -> int:
    game_dir = Path(game_dir)
    backup_root = game_dir / BACKUP_DIR
    count = 0
    for b in sorted(backup_root.rglob("*")):
        if not b.is_file():
            continue
        rel = b.relative_to(backup_root)
        if rel.as_posix() == APPLIED_LEDGER_NAME:
            continue
        target = game_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(b, target)
        count += 1

    ledger = _load_applied_ledger(game_dir)
    for rel_key in ledger.get("created", []):
        target = game_dir / Path(rel_key)
        if not _is_contained(game_dir, target):
            print(f"skip (unsafe ledger entry escapes game dir): {rel_key}")
            continue
        if not target.exists():
            continue
        target.unlink()
        parent = target.parent
        while parent != game_dir:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent

    ledger_path = _ledger_path(game_dir)
    if ledger_path.exists():
        ledger_path.unlink()
    return count
