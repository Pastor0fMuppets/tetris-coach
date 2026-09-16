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

## Why the stray is admissible at all

`_pick_falling` / `_rank` accept a sub-tetromino candidate ANYWHERE on the
board. A partial sighting is only explicable where something hides the
rest of the piece: clipped by the top edge of the board region, or under
the NEXT panel (`unobservable_cells`). The J at (0,3) (0,4) (0,5) is the
first; so are the three-cell readings at 00390 and 00392-00397, whose
fourth cell is at (1,8) / (1,9) under the panel. A lone cell at row 8 in
open board is neither.
