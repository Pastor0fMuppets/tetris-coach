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
2. **Watch** — the board region is captured ~15×/sec and read into a 10-wide grid
   (rows configurable via `--rows`, default 20). By default each cell is named by
   the colour it is drawn in, which names the falling piece from its first visible
   cell; the next piece is read from its own box.
3. **Solve** — all placements of the current piece are searched, with lookahead over
   all placements of the next piece, scored by a Dellacherie-style evaluation
   (bitboard implementation for speed).
4. **Show** — a transparent, always-on-top, click-through Qt window aligned to the
   board draws the target placement. The next hint is precomputed while you play, so
   it appears within a frame of each piece locking.

## Two trackers

There are two ways to read a frame, and `--tracker` chooses between them.

- `--tracker colour` (default) names each cell by the direction of its colour away
  from the board's background, and re-derives the board from every frame. It keeps
  no committed picture of the stack, so a misread frame costs one frame.
- `--tracker shape` is the original reader: each cell becomes one bit, and the
  falling piece is a set difference against a committed stack memory.

Colour is the default on measurement, not taste. Replayed through the real engine
over six captured windows of real play, judged against a pixel-derived answer sheet
that no tracker takes part in building (`python -m tetris_coach.race`): over 422
frames the colour reader drew no hint for the wrong piece where the shape reader
drew 6, invented no piece over the stretches the answer sheet abstains on (line
clears and covered boards), showed a hint the frame each piece appeared where the
shape reader took up to 3 frames, never moved a target while its piece was in
flight, and left the overlay blank on 4 frames against 25. Window by window it is
better or equal on every one of those. `--tracker shape` is kept as an escape hatch: the two refuse
frames on different evidence, so a game or theme the colour rules cannot read is a
flag away rather than a rebuild.

Both readers are written to stay game-agnostic — where colour says nothing (a
monochrome theme, two pieces a game draws alike), the contradicted colour is retired
from naming and the tracker names by shape, which is where a colour-blind reader
always was; where a contradiction is instead two colours the matching tolerance ran
together, the class comes apart and both keep their names — but the evidence above is one game, and the only pieces in those windows
are I, O and T.

## Status

Under construction. See [SPEC.md](SPEC.md) for the full architecture and build plan.

## Requirements

- macOS (Screen Recording permission required for capture)
- Python 3.11+
