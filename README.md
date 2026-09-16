# Tetris Coach

A game-agnostic Tetris training overlay for macOS. You highlight the region of the
screen where any Tetris game is being played; Tetris Coach watches the board, computes
the optimal placement for the current piece (using the next piece for lookahead), and
draws that placement — and where the piece after it goes — on a transparent
click-through overlay, so a human player can learn optimal stacking by playing toward
the hint.

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
   board draws both targets: a bold solid outline where the piece in play goes, and
   a dashed outline in a second colour where the piece after it goes if you take the
   first. The second one is already computed while you play, so the hint also flips
   within a frame of each piece locking.

## Two hints

The overlay shows two placements at once, and they are told apart by the stroke
rather than by one of them being faint:

| | drawn as | means |
|---|---|---|
| current piece | bold **solid** outline (`--hint-color`) | put this piece here |
| next piece | **dashed** outline, second colour (`--next-hint-color`) | ...and then the next one goes here |

The second hint is conditional, and it disappears rather than mislead you. It is
where the next piece goes on the board the first placement would leave behind, so
it is only shown while that is still the board you are about to produce: put the
piece somewhere else and the next frame re-solves both. It is also not shown when
the first placement clears a line, because a clear shifts every row above it and
the two would no longer be talking about the same rows. In practice it is there
most of the time: over this project's captured sessions it is on screen for 529
of the 627 frames that have a hint at all, and the line clear is what accounts
for every one of the rest. `--no-next-hint` turns it off for good if you would
rather have one square to aim at.

### Why the hints are outlines

Both hints are drawn entirely in the outer quarter of each cell. That band is the
part of a cell the vision code never samples — it reads each cell from its middle —
so the coach physically cannot read its own drawing back as board content. This is
worth a line in a README because it is the bug this tool has had three times: a
hint that flickered, a hint that pointed at a piece you did not have, a phantom
piece made of the coach's own paint. It is now geometry rather than a rule, and the
rule is kept behind it. It also means the hints can be a lot bolder than a hint
that had to stay out of the way.

`--hint-fill 0.18` fills the current hint's cells again, the way older versions
drew it. That fill is inside the part of the cell vision reads, so turning it on
re-opens exactly that path: the reading is then only correct because the paint is
recognized and taken back out again. It is off by default and says so when you use
it. With the fill on, `--next-hint-color` also has to be far enough from
`--hint-color` for the coach to tell its own two marks apart — the recognition
goes by the colour of the outline round the fill, so a second hint in the first's
colour is read as a fill that was never painted. That pair is refused at startup
rather than drawn.

The overlay also keeps out of the next-piece rectangle. In games that float the
NEXT box on top of the playfield, that box is inside the region you select as the
board, so the window covers it — and a hint painted across it would blind the
coach's own reading of the upcoming piece. The part of a hint that would land
there is simply not drawn; it was behind the game's own panel anyway.

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

## Options

```
--hint-color COLOR        colour of the current hint (default #00e5ff)
--next-hint-color COLOR   colour of the next-piece hint (default #ff00e5)
--no-next-hint            show one target only
--hint-fill OPACITY       fill the current hint's cells (default 0 = outline only)
--tracker {colour,shape}  which reader reads the frames (default colour)
--rows N                  board height in rows (width is always 10)
--poll-rate N             captures per second (default 15)
--debug                   print what vision sees, frame by frame
```

## Requirements

- macOS (Screen Recording permission required for capture)
- Python 3.11+
