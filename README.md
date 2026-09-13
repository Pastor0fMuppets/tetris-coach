# Tetris Coach

A game-agnostic Tetris training overlay for macOS. You highlight the region of the
screen where any Tetris game is being played; Tetris Coach watches the board, computes
the optimal placement for the current piece (using the next piece for lookahead), and
draws a ghost outline of that placement on a transparent click-through overlay — so a
human player can learn optimal stacking by playing toward the hint.

**It never touches the game.** No keystrokes, no memory reading, no integration —
purely visual: screen capture in, overlay out. Built as a training tool to develop
placement intuition.

## How it works

1. **Setup** — drag a rectangle over the game's board, and a second one over the
   "next piece" preview box.
2. **Watch** — the board region is captured ~15×/sec and thresholded into a 10-wide
   occupancy grid (rows configurable via `--rows`, default 20); the falling piece
   and next piece are recognized by shape.
3. **Solve** — all placements of the current piece are searched, with lookahead over
   all placements of the next piece, scored by a Dellacherie-style evaluation
   (bitboard implementation for speed).
4. **Show** — a transparent, always-on-top, click-through Qt window aligned to the
   board draws the target placement. The next hint is precomputed while you play, so
   it appears within a frame of each piece locking.

## Status

Under construction. See [SPEC.md](SPEC.md) for the full architecture and build plan.

## Requirements

- macOS (Screen Recording permission required for capture)
- Python 3.11+
