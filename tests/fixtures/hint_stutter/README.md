# Live session capture: the hint flashing between two targets

51 CONSECUTIVE ticks (frames 00350-00400) from the session the user
reported as "sometimes the recommendation stutters ... it would quickly
flash from one location to another and back and forth between the two".
PNGs are RGB; convert with `[:, :, ::-1]`.

Session geometry:

    board_rect = Rect(left=198, top=316, width=480, height=573)
    next_rect  = Rect(left=581, top=315, width=98,  height=93)
    rows       = 12
    compute_overlap_mask(...) -> {(0, 8), (0, 9), (1, 8), (1, 9)}

The window holds one whole J descent, from the frame it appears clipped by
the top edge (00355) to the frame it locks (00401, just past the window).
It is the first J in the fixture corpus: the adoption review flagged J and
Z as the two tetrominoes nothing committed exercised, and this is the bug
that was hiding in that gap.

## What the user sees

Measured over these 51 frames, through `ColourTracker` alone:

    hint target A->B->A within 12 frames             31
    falling piece -> None -> same piece              16   all of them J
    stack-size oscillations A B A                    34
    PIECE_LOCKED events                              17   the J locks once

Seventeen locks for one piece. At 15 fps that is the hint blinking on and
off about seven times a second for most of a second and a half at a time.

## The mechanism, frame by frame

Two candidates alternate, and they share ZERO cells:

    00355  falling J  cells (0,3) (0,4) (0,5)   stack 22   the real J,
                                                           clipped by the
                                                           board's TOP EDGE
    00357  falling ?  cells (8,8)               stack 25   one stray cell,
                                                           mid-board
    00358  falling J  cells (0,3) (0,4) (0,5)   stack 22
    00359  falling ?  cells (8,8)               stack 25
    ... alternating to 00399

On the frames the stray wins, the J is not the piece in flight, so it
falls into the settled stack (22 -> 25 cells), the board handed to the
solver is three cells wrong, `falling.piece` reads None and the overlay
blanks. `_events` sees a piece that stopped being in flight over a stack
that grew and calls it a LOCK, so a spawn is reported on the way back too.
The next frame it flips back. That is the stutter.

`_rank` orders on youth before position, and the stray's colour class
changes every frame, so its age is 0 on every frame it is read and it
outranks a J that has held still.

## What the stray cell is: THE COACH'S OWN ROTATION BADGE

It is not noise, and it is not the game's landing preview. Cell (8, 8)
holds no board content at all -- its sampled patch reads
(252.0, 251.9, 250.9), the background, to within a tenth of a uint8 unit.
What is in it is 122 pixels of `#00e5ff` across its **top margin**: the
rotation badge of this tool's own hint, which on 00357 is drawn as a J at
(8,9) (9,9) (10,8) (10,9).

`rotation_badge_rect` hung the badge at `min(row), min(col)` -- the corner
of the hint's BOUNDING BOX -- and for 6 of the 19 rotations that is a cell
the piece does not occupy. Both T verticals, S and Z in one orientation
each, one each of J and L. That is why the session's counts are
{J: 16, Z: 11, T: 2} and why no I or O ever appears in them.

On a cell of bare board the badge covers part of one edge and none of the
sampled patch, so `own_paint_states` -- which told our translucent FILL
from our opaque DRAWING by the patch alone -- called it a fill. Undoing a
fill that is not there maps the board's own ground to a vector
`alpha/(1-alpha)` of the way from the background AWAY from the hint
colour: 55 units off, which is content, which floats, and whose colour
class is re-interned every frame so its age is always 0 and it outranks
everything.

So the stutter is the coach reading its own drawing, and this window is
the evidence for three separate fixes:

- the badge now goes in the leftmost cell of the hint's TOP ROW, which
  every rotation occupies;
- `own_paint_states` recognizes our fill by its ring running along all
  four sides of a cell, and calls anything else of ours opaque drawing --
  skipped, never un-composited;
- `_rank` admits a sub-tetromino candidate only where something could be
  hiding the rest of it: clipped by the top edge, under the NEXT panel
  (`unobservable_cells`), or under this tool's OWN OPAQUE PAINT, which
  `Palette.classify` skips and so cannot read a piece through. The J at
  (0,3) (0,4) (0,5) is the first; so are the three-cell readings at 00390
  and 00392-00397, whose fourth cell is at (1,8) / (1,9) under the panel;
  `pale_preview` board_00477 is the third, a T whose fourth cell is under
  the badge itself. A lone cell at row 8 of open board is none of them.

  Two later bounds on the same rule, both from a review of this work and
  both reproduced before they were changed: the exemption that lets the
  piece already in flight skip this test is a LOAN OF ONE FRAME (a
  fragment was otherwise handed the belief a cell at a time and walked
  four rows into open board), and it covers a piece being DRAGGED and not
  only one parked (a cell dropping out of a moving piece was otherwise
  fatal: stack, lock, spawn, hint withdrawn -- this very stutter, made by
  the rule against it). Where a lone cell is admissible at all -- the
  whole of row 0 and the panel's edge, 11 of this session's 120 cells --
  it now sorts BELOW every other candidate, because one cell is the least
  evidence there is and a stray up there floats.

`tests/test_hint_stutter.py` replays this window and asserts all of it.
All ten of its assertions fail against the reading that shipped.
