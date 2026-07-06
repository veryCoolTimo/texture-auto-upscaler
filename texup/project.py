from __future__ import annotations

import json
import os
from pathlib import Path

MANIFEST_NAME = "texup-project.json"


class Project:
    _STATUSES = ("pending", "done", "failed", "skipped")

    def __init__(self, game_dir: Path, out_dir: Path, textures: dict[str, dict],
                 wizard: dict | None = None):
        self.game_dir = Path(game_dir)
        self.out_dir = Path(out_dir)
        self._textures = textures
        self._wizard: dict = dict(wizard) if wizard else {}

    @classmethod
    def create(cls, game_dir: Path, out_dir: Path) -> "Project":
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        prj = cls(game_dir, out_dir, {})
        prj.save()
        return prj

    @classmethod
    def load(cls, out_dir: Path) -> "Project":
        out_dir = Path(out_dir)
        data = json.loads((out_dir / MANIFEST_NAME).read_text())
        return cls(Path(data["game_dir"]), out_dir, data["textures"], data.get("wizard", {}))

    @property
    def wizard(self) -> dict:
        return dict(self._wizard)

    def set_wizard(self, answers: dict) -> None:
        self._wizard.update(answers)
        self.save()

    def add_texture(self, key: str, *, codec: str, klass: str, confidence: float,
                    sha256: str, width: int, height: int, fmt: str, content_sha: str | None = None) -> None:
        self._textures[key] = {
            "key": key, "codec": codec, "klass": klass, "confidence": round(confidence, 3),
            "sha256": sha256, "width": width, "height": height, "fmt": fmt,
            "status": "pending", "reason": None, "model": None,
        }
        if content_sha is not None:
            self._textures[key]["content_sha"] = content_sha

    def set_status(self, key: str, status: str, *, reason: str | None = None,
                   model: str | None = None) -> None:
        if status not in self._STATUSES:
            raise ValueError(f"unknown status: {status!r}")
        rec = self._textures[key]
        rec["status"] = status
        rec["reason"] = reason
        if model is not None:
            rec["model"] = model

    def records(self, status: str | None = None, klass: str | None = None) -> list[dict]:
        out = list(self._textures.values())
        if status is not None:
            out = [r for r in out if r["status"] == status]
        if klass is not None:
            out = [r for r in out if r["klass"] == klass]
        return out

    def save(self) -> None:
        payload = {"game_dir": str(self.game_dir), "textures": self._textures,
                   "wizard": self._wizard}
        target = self.out_dir / MANIFEST_NAME
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=1))
        os.replace(tmp, target)

    @staticmethod
    def source_of(key: str) -> tuple[Path, str]:
        if "::" in key:
            src, inner = key.split("::", 1)
            return Path(src), inner
        return Path(key), ""
