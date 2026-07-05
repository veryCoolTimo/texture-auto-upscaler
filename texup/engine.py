from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Upscaler:
    def __init__(self, model: torch.nn.Module, scale: int, device: str | None = None):
        self.device = device or pick_device()
        self.model = model.eval().to(self.device)
        self.scale = scale
        self.tile_size = 512
        self.tile_overlap = 16
        # fp16 speeds up MPS inference substantially; CPU/CUDA keep fp32 behavior.
        self.use_fp16 = False
        if self.device == "mps":
            try:
                self.model = self.model.half()
                self.use_fp16 = True
            except Exception:  # noqa: BLE001 — model doesn't support half precision
                self.use_fp16 = False

    @torch.inference_mode()
    def _run_rgb(self, rgb: np.ndarray) -> np.ndarray:
        """(H,W,3) uint8 -> (H*s,W*s,3) uint8, тайлингом."""
        h, w = rgb.shape[:2]
        s = self.scale
        tile = self.tile_size
        while True:
            try:
                if h <= tile and w <= tile:
                    return self._infer(rgb)
                out = np.zeros((h * s, w * s, 3), dtype=np.uint8)
                ov = self.tile_overlap
                for y0 in range(0, h, tile):
                    for x0 in range(0, w, tile):
                        y1, x1 = min(h, y0 + tile), min(w, x0 + tile)
                        ys, xs = max(0, y0 - ov), max(0, x0 - ov)
                        ye, xe = min(h, y1 + ov), min(w, x1 + ov)
                        patch = self._infer(rgb[ys:ye, xs:xe])
                        oy, ox = (y0 - ys) * s, (x0 - xs) * s
                        out[y0 * s : y1 * s, x0 * s : x1 * s] = patch[
                            oy : oy + (y1 - y0) * s, ox : ox + (x1 - x0) * s
                        ]
                return out
            except RuntimeError as e:  # OOM: на CUDA это OutOfMemoryError(RuntimeError), на MPS — RuntimeError
                if "memory" not in str(e).lower() or tile <= 64:
                    raise
                tile //= 2
                if self.device == "mps":
                    torch.mps.empty_cache()
                elif self.device == "cuda":
                    torch.cuda.empty_cache()

    def _forward(self, rgb: np.ndarray, fp16: bool) -> np.ndarray:
        dtype = torch.float16 if fp16 else torch.float32
        t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(
            self.device, dtype=dtype
        )
        out = self.model(t)
        return out.float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()

    def _infer(self, rgb: np.ndarray) -> np.ndarray:
        out = self._forward(rgb, self.use_fp16)
        if self.use_fp16:
            input_was_zero = not np.any(rgb)
            degenerate = np.isnan(out).any() or (not input_was_zero and not np.any(out))
            if degenerate:
                # fp16 produced garbage on this hardware/model combo — fall back to fp32
                # permanently for the rest of this Upscaler's lifetime, and redo this tile.
                self.model = self.model.float()
                self.use_fp16 = False
                out = self._forward(rgb, False)
        return (out * 255.0 + 0.5).astype(np.uint8)

    def run(self, rgba: np.ndarray, max_size: int = 4096) -> np.ndarray:
        rgb_up = self._run_rgb(rgba[..., :3])
        alpha = rgba[..., 3]
        h, w = rgb_up.shape[:2]
        if np.all(alpha == alpha.flat[0]):
            a_up = np.full((h, w), alpha.flat[0], dtype=np.uint8)
        else:
            # Alpha is structural (e.g. cutout masks), not photographic detail — a Lanczos
            # resize is visually indistinguishable from a second neural pass here and much
            # cheaper (skips a full model inference per texture).
            a_up = np.asarray(Image.fromarray(alpha, "L").resize((w, h), Image.LANCZOS))
        out = np.dstack([rgb_up, a_up])
        h, w = out.shape[:2]
        long_side = max(h, w)
        if long_side > max_size:
            k = max_size / long_side
            nw, nh = max(1, round(w * k)), max(1, round(h * k))
            out = np.asarray(Image.fromarray(out, "RGBA").resize((nw, nh), Image.LANCZOS))
        return out


def load_upscaler(model_name: str, device: str | None = None) -> Upscaler:
    from spandrel import ModelLoader

    from texup.models import MODELS, get_model_path

    path = get_model_path(model_name)
    loaded = ModelLoader().load_from_file(str(path))
    return Upscaler(loaded.model, MODELS[model_name].scale, device)
