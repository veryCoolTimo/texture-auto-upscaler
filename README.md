<p align="center">
  <img src="assets/banner.webp" alt="texup — AI texture remaster for old games" width="100%">
</p>

# texup

**One-command AI texture remaster for old games.** Point it at a game folder — it finds every texture, figures out what each one is, upscales it with the right neural model, and packs everything back exactly the way the game expects. Locally, on your machine, with full backup and rollback.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-MPS%20%7C%20CUDA%20%7C%20CPU-ee4c2c.svg)](https://pytorch.org/)
[![spandrel](https://img.shields.io/badge/models-OpenModelDB%20via%20spandrel-8A2BE2.svg)](https://openmodeldb.info/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)]()

The goal is a *respectful* remaster: sharper, more detailed textures that keep the original art style — not a generative repaint of the game.

## How it works

```
game folder ──► scan ──► classify ──► route ──► upscale ──► re-encode ──► output folder
                 │          │           │          │            │
              codecs     diffuse?    per-class   tiled fp16   same format,
             (DDS, TEX,  normal?     AI model    inference    same mips,
              ARC, PNG)  UI? font?   selection   on GPU       same layout
```

1. **Scan** — walks the game folder, decodes every texture it recognizes, including textures packed inside game archives.
2. **Classify** — filename patterns, color statistics and compression format decide whether each texture is diffuse, a normal map, a material mask, UI, or a font atlas.
3. **Route** — each class gets its own treatment: a detail-restoring GAN for diffuse, a normal-map-specific model (with the game's channel swizzle handled), classic filtering for font atlases — neural nets make text worse, not better.
4. **Upscale** — tiled inference with fp16 on Apple Silicon (MPS) or CUDA, automatic OOM fallback, alpha handled separately.
5. **Re-encode** — same compression family, regenerated mip chains, byte-faithful container layout. If the game had a DXT5 texture in a zlib archive, it gets a DXT5 texture in a zlib archive back.
6. **Apply / rollback** — results go to an output folder first. `apply` copies them into the game with originals backed up; `rollback` restores everything.

## Quick start

```bash
git clone git@github.com:veryCoolTimo/texture-auto-upscaler.git
cd texture-auto-upscaler
python3 -m venv .venv && .venv/bin/pip install -e .

.venv/bin/texup remaster "/path/to/GameFolder"
```

That's the whole workflow: texup scans the game, shows what it found (engine, texture classes, duplicates), asks two questions — quality mode with a time estimate **for your hardware**, and whether to write into the game (with automatic backup) or an output folder — then runs silently with a progress bar and ETA. Interrupt any time; it resumes where it left off. Models download automatically on first use (~130 MB).

| Command | What it does |
|---|---|
| `texup remaster <game>` | the one command: scan → 2 questions → run → report |
| `texup bench` | ~1-min hardware calibration that powers the time estimates |
| `texup scan / upscale / apply / rollback / status / preview` | the individual steps, for scripting |

Changed your mind after applying? `texup rollback "/path/to/GameFolder"` restores everything.

## Supported games

texup is format-driven, not game-driven: any game whose textures it can decode, it can remaster.

| Games | Formats | Status |
|---|---|---|
| **Any game with loose textures** — hundreds of older PC titles, most indie games, anything already unpacked by modding tools | PNG, JPG, TGA, BMP, DDS (BC1–BC7) | ✅ works today |
| **Resident Evil 5**, **Resident Evil 0 HD**, **Resident Evil 6** | MT Framework v1/v2 `.tex` + `.arc` | ✅ verified on full installs |
| **Dragon's Dogma, DMC4, Lost Planet** | same MT Framework formats | ✅ expected to work |
| **Quake III, Doom 3, RTCW** and other id-Tech era games | ZIP-based paks (PK3/PK4) | ✅ |
| **Half-Life 2, Portal, CS:S, L4D, TF2** and Source mods | VTF textures, VPK read → loose mod output | ✅ cross-validated vs srctools |
| **Skyrim, Fallout 4** and Bethesda mods | BSA / BA2 read → loose mod output | ✅ cross-validated vs bethesda-structs |

Verification standard: every codec is validated against a full real installation before it's called supported — RE5: 645/645 loose textures byte-exact, 1231/1232 archives repack byte-identical; RE0 HD: 10,454 textures parse with exact size match. Codecs are plugins — a new engine is one file implementing `detect / decode / encode_file`.

## Use with Claude Code

Don't want to touch a terminal? Install the skill and just say *"remaster the textures in my game"* — Claude runs texup for you, shows you the before/after sheets, and applies (with backup) once you approve:

```
/plugin marketplace add veryCoolTimo/texture-auto-upscaler
/plugin install texup-skill@texup
```

## Under the hood

| Layer | Support |
|---|---|
| Texture classes | diffuse / albedo, normal maps (incl. DXT5nm AG-swizzle), material masks, UI, font atlases |
| Hardware | Apple Silicon (MPS, fp16), NVIDIA (CUDA), CPU fallback |
| Models | anything [spandrel](https://github.com/chaiNNer-org/spandrel) loads — the registry ships Remacri, Real-ESRGAN x4plus and a BC-aware normal-map model |

## Built for real scale

A real 2009 game is ~36,000 textures. Naively that's a day and a half of GPU time. texup gets it down to a few hours:

| Optimization | Effect |
|---|---|
| Content dedupe cache | 59% of the test game's textures were exact copies packed into multiple archives — each unique texture is upscaled once |
| fp16 inference on MPS | ~2x, with automatic fp32 fallback if the model misbehaves |
| CPU ‖ GPU pipeline | archive re-encoding overlaps with inference of the next texture |
| Small-texture batching | icons and masks ride through the GPU in packs, not one by one |
| Resume | the manifest journals every texture; interrupt and continue any time |

## Choosing the look

Two quality modes, each shown with a time estimate calibrated to your machine:

- **Faithful** — Real-ESRGAN: clean and conservative, closest to the original.
- **Detailed** (default) — [Remacri](https://openmodeldb.info/models/4x-foolhardy-Remacri), the community favourite for texture packs: it *restores* surface detail (fabric weave, rust grain) instead of just sharpening.

Every run writes side-by-side comparison sheets to `_compare/` so you judge with your eyes, not metrics. Anything from [OpenModelDB](https://openmodeldb.info/) can be wired in via `texup/models.py` + `texup/presets.py`.

## Roadmap

- Claude Code skill — drive texup conversationally, no terminal knowledge needed
- Diffusion "hero mode" — one-step diffusion SR for handpicked environment textures
- Cubemap support for MT Framework

## Fair use

texup is a modding tool. Use it on game copies you own; don't redistribute processed game assets — share the tool, not the textures.

## License

[MIT](LICENSE)
