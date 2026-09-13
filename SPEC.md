# Tetris Coach — Architecture & Build Spec

## Product summary

A macOS desktop tool. The user starts the program, drags a selection rectangle over
the board area of *any* Tetris game visible on screen (plus a second rectangle over
the next-piece preview), and the tool overlays the optimal placement of the current
piece on top of the game. Purely visual — capture in, overlay out, no game input.
Purpose: training human placement intuition at full game speed.

## Hard requirements

- **Game-agnostic**: works on any Tetris rendering a standard 10×20 board with
  reasonably solid cell colors distinguishable from the board's background — dark,
  light, and colored themes alike. No per-game config beyond the user-selected
  regions.
- **Fast**: end-to-end capture→overlay under ~100 ms; hint for a newly spawned piece
  visible within ~1 frame in the common case (via precomputation, see below).
- **Visual only**: never send input to the game.

## Module layout (`src/tetris_coach/`)

```
core/
  board.py        # Bitboard: 20 rows × 10 cols, one int per row (bits 0..9).
                  # Ops: occupancy from bool array, drop piece, clear lines,
                  # column heights, hole count, transitions, wells.
  pieces.py       # 7 tetrominoes; all distinct rotations as (rotation, cells) with
                  # per-rotation column masks/bottom profiles for fast drops.
solver/
  evaluate.py     # Dellacherie features: landing height, eroded piece cells,
                  # row transitions, column transitions, holes, cumulative wells.
                  # Classic Dellacherie weights as defaults; weights in a dataclass.
  search.py       # enumerate all (rotation, column) placements of current piece;
                  # for each, resulting board -> best next-piece placement score
                  # (2-ply lookahead, no pruning needed at this size: ~34×34 evals).
                  # Must run < 50 ms in pure Python via bitboards; benchmark it.
vision/
  grid.py         # Given BGR image of board region + (rows, cols): per-cell
                  # occupancy via color distance from a per-frame background
                  # estimate (top-row cell-color median), split by Otsu.
                  # Returns 20×10 bool array + confidence.
  pieces_vision.py# explain_grid: diff the observed board against the tracker's
                  # committed stack memory and classify the frame (QUIET, FALLING,
                  # LOCKED, UNEXPLAINED). The falling piece is the added-cell diff
                  # matched against tetromino shapes — never guessed from a single
                  # frame — so lock delay, floor contact, and post-clear debris
                  # cannot confuse it. Locks are verified structurally: revealed by
                  # the next spawn, or a line-clear placement (tiered: last observed
                  # position, gravity drop from it, any clearing gravity drop)
                  # reproducing the observation exactly.
                  # Next-piece region: threshold, crop to bounding box, normalize to
                  # cell grid, match shape signature. Color is a hint, not required.
  state.py        # GameState tracker: keeps the committed stack as the one
                  # authoritative memory (never None), feeds explain_grid, and
                  # debounces (2 consistent frames) before committing. UNEXPLAINED
                  # frames touch nothing (line-clear animations, torn frames);
                  # PIECE_LOCKED fires when a lock is structurally verified, not at
                  # touchdown; BOARD_RESET only after several consecutive identical
                  # unexplainable frames (new game, garbage, mid-game attach).
capture/
  screen.py       # mss-based capture of a screen rect at native (Retina) scale;
                  # handles logical-vs-pixel coordinate scaling. Protocol/interface
                  # so tests can inject frames from PNG files instead.
overlay/
  window.py       # PySide6 frameless, transparent, always-on-top, click-through
                  # window (WA_TransparentForMouseEvents, WindowStaysOnTopHint,
                  # WindowTransparentForInput, NoDropShadowWindowHint) positioned
                  # exactly over the board region.
  renderer.py     # Draw target placement: 4 cell outlines + subtle fill; distinct
                  # color (configurable); optional arrow/rotation count badge.
region_select.py  # Full-screen dim + drag-rectangle picker (Qt), returns rect in
                  # logical coords; run twice (board, next box).
app.py            # Main loop wiring: capture -> vision -> state -> solve -> overlay.
                  # Precompute: while piece A falls, assume it lands on target and
                  # pre-solve piece B; on lock, flip hint instantly; if observed
                  # board != predicted, re-solve from observed.
cli.py            # `tetris-coach` entry point: select regions, start loop; flags
                  # for poll rate, colors, and a terminal debug view (per
                  # committed frame: observed grid, falling piece, next piece,
                  # confidence, events). A *graphical* debug window is deferred
                  # to the live phase..
```

## Key design decisions (already made — do not relitigate)

1. **Bitboards** (one int per row) for the solver. Target: full 2-ply search
   < 50 ms in CPython. Benchmark in tests; if over budget, optimize before adding
   features. Numba is a fallback, not a default dependency.
2. **Dellacherie evaluation** (6 features, fixed classic weights) rather than
   height/holes/bumpiness-only. One-piece + next-piece search (2-ply), no deeper.
3. **Precompute-ahead loop** (app.py above) so perceived latency ≈ one repaint.
4. **User-selected regions** rather than auto-detection of the board. Auto-detect is
   a possible later enhancement, not v1.
5. **PySide6** for region picker + overlay. **mss** for capture. **OpenCV + NumPy**
   for vision. Python 3.11+.
6. Vision works on *occupancy*, not colors, for game-agnosticism. Colors only help
   piece identification as a secondary signal.

## What can be built & tested headless (Linux CI / cloud)

- All of `core/`, `solver/`: pure logic. Unit tests + a self-play simulation test
  (solver should survive ≥ 1000 pieces on random sequences without topping out —
  Dellacherie's algorithm averages far more) + a pytest-benchmark for search time.
- All of `vision/`: test against generated synthetic board screenshots (render
  boards with varied palettes/cell sizes/gridlines/backgrounds via Pillow) plus a
  few real screenshots committed under `tests/fixtures/` (small PNGs of e.g.
  Jstris/Tetr.io boards).
- `capture/screen.py` and `overlay/` are macOS-only at runtime: keep them thin,
  isolate behind interfaces, write them to spec, exclude from headless test runs
  (import-guard so the package imports cleanly without a display).

## Definition of done for the headless phase

- `pytest` green, including vision fixture tests and solver self-play.
- Benchmark report in `BENCH.md`: search time p50/p95 on M-class and CI hardware.
- `python -m tetris_coach.cli --demo` runs a terminal demo: simulated game where
  the solver plays itself, printing the board — proves the whole non-GUI stack.
- Code formatted (ruff), typed (mypy clean on core/solver at least).
