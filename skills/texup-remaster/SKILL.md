---
name: texup-remaster
description: >-
  Use when the user wants to remaster, upscale, or improve the textures of an
  old PC game — Resident Evil, Half-Life 2 / Source games, Skyrim, Fallout 4,
  Quake / Doom 3, or any game folder with loose or packed textures. Drives the
  local `texup` CLI end to end: install-check, scan, show before/after
  comparison sheets, run the upscale with a quality preset, apply into the game
  with automatic backup, and roll back on request. Triggers on "улучши текстуры
  в игре", "сделай ремастер текстур", "upscale my game's textures", "make
  <game> look better", "texture pack for <game>".
---

# texup-remaster

Turn "make my old game's textures look better" into a finished, applied texture
remaster — without the user touching a terminal. You run the `texup` CLI on
their behalf, narrate what it found, show them comparisons, and only apply to
the real game once they approve.

`texup` is a local tool (this repo). It scans a game folder, classifies every
texture (diffuse / normal map / material / UI / font), upscales each with the
right neural model, and repacks to the original format — DDS, MT Framework
`.tex`/`.arc` (Resident Evil), Source VTF/VPK, Bethesda BSA/BA2, id-Tech PK3/PK4.
Everything runs on the user's machine (Apple Silicon MPS, NVIDIA CUDA, or CPU).

## Before you start

1. **Confirm texup is installed.** Run `texup --help`. If it fails, install it:
   ```bash
   git clone https://github.com/veryCoolTimo/texture-auto-upscaler.git
   cd texture-auto-upscaler
   python3 -m venv .venv && .venv/bin/pip install -e .
   ```
   Then use `.venv/bin/texup` as the command. The first upscale downloads
   models (~130 MB) automatically.

2. **Get the game folder.** Ask the user for the path to their game's install
   directory if they haven't given it. Do not guess.

## The workflow

Prefer the single wizard command — it already does the whole flow with the
right pauses:

```bash
texup remaster "/path/to/GameFolder"
```

It scans, prints a summary (engine, texture counts, duplicates), asks two
questions (quality mode with a time estimate for this machine; write into the
game with backup, or into an output folder), then runs with a progress bar and
writes before/after comparison sheets to `_compare/`.

**Your job around it:**

- **Run a hardware calibration first** (`texup bench`, ~1 min) if the user wants
  an accurate time estimate before committing — otherwise the wizard offers it.
- **When the wizard asks questions, ask the user in chat and pass their answers.**
  Explain the two quality modes in plain terms:
  - **Faithful** — clean and conservative, closest to the original art.
  - **Detailed** (default) — restores surface detail (fabric weave, rust, grain);
    the "texture pack" look. Recommend this unless the user wants minimal change.
- **Show, don't tell.** After the sample/first pass, point the user at the
  `_compare/` folder (or open a couple of the sheets) so they judge the result
  with their eyes before a full run or before applying to the game.
- **Applying is reversible but outward-facing — confirm first.** Only choose the
  "write into the game" mode when the user has explicitly agreed. Originals are
  backed up automatically; tell the user the exact rollback command:
  `texup rollback "/path/to/GameFolder"`.

## For a careful, staged run (recommended for large games)

If the user wants to preview before spending hours of GPU time, drive the
individual commands instead of the one-shot wizard:

```bash
texup scan "/path/to/GameFolder" --out ./out          # find + classify
texup upscale ./out --sample 5 --compare              # 5 per class + sheets
# → show ./out/_compare to the user, get approval
texup upscale ./out                                   # full run (resumable)
texup apply ./out                                     # into game, with backup
texup rollback "/path/to/GameFolder"                  # undo anytime
```

`texup status ./out` shows progress at any point. A run interrupted with Ctrl+C
resumes from where it stopped.

## What to tell the user honestly

- **Already-remastered games see little change.** texup shines on genuinely old,
  low-resolution, compression-muddy textures. If a game already ships crisp
  high-res art (e.g. a 2016 HD re-release), the improvement is subtle — say so
  rather than overselling.
- **Time scales with the game.** A ~36,000-texture game is a few hours on
  Apple Silicon after texup's dedupe/fp16/batching optimizations. Give the
  `texup bench`-based estimate, don't guess wildly.
- **It's a modding tool.** Use it on game copies the user owns; processed game
  assets shouldn't be redistributed — the tool is shareable, the textures aren't.

## Red flags — stop and ask

- The game folder path doesn't exist or contains no recognizable textures
  (`texup scan` finds zero) — tell the user, don't invent a path.
- The user asks to apply to the game before seeing any comparison — offer to
  show `_compare/` first.
- A full run would take many hours and the user hasn't confirmed the time — quote
  the estimate and get a yes before starting.
