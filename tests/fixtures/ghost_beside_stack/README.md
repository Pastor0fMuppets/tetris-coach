# Live session capture: a ghost the ghost rule refuses (ROAS Stacker)

48 CONSECUTIVE ticks (frames 00138-00185). PNGs are RGB; convert with
`[:, :, ::-1]`.

Session geometry:

    board_rect = Rect(left=276, top=314, width=477, height=578)
    next_rect  = Rect(left=659, top=317, width=98,  height=95)
    rows       = 12
    compute_overlap_mask(...) -> {(0, 8), (0, 9), (1, 8), (1, 9)}

## The symptom the user reported

"The overlay moves around a lot for the same piece and turn." Measured over
this session: 11 of 24 piece-episodes showed the hint jumping between two
different targets. For this I piece the hint alternates between leftmost
column 4 and leftmost column 0 - a jump across the whole board.

## The cause

The OBSERVED grid flickers. The I piece sits at row 1, cols 4-7. Its GHOST
sits at row 11, cols 4-7 - the same columns, one row off the floor:

    frame 143-144  r11 obs "........##"   ghost NOT read (conf 0.36)
    frame 145-146  r11 obs "....######"   ghost READ as occupied (conf 0.26)
    frame 147-148  r11 obs "........##"   ghost gone again

When the ghost reads as occupied the tracker sees the I arrive at the
bottom, commits a LOCK, and re-solves - moving the hint. When it vanishes
the frame is unexplainable and the cycle restarts. It repeats every ~11
frames: FALLING(stack 2) -> LOCKED(stack 2) -> LOCKED(stack 6) ->
UNEXPLAINED -> QUIET(stack 6) -> FALLING(stack 2) -> ...

## Why the existing ghost rule refuses this ghost

`_ghost_layer`'s test 4 drops any candidate with a solid cell to its left,
right or above. That test exists to protect a real pale-periwinkle piece
that the stack runs into. But here the ghost lands beside existing stack at
r11 cols 8-9, so the test refuses it and the ghost is kept as board content.
A ghost landing next to the stack is an ordinary board configuration, not an
edge case.

Note the strong structural signal available and unused: this ghost occupies
the SAME COLUMNS as the falling piece, with the same shape, resting at the
piece's landing position. That is what a ghost IS, in every Tetris.

## CORRECTION: it is not a ghost

Everything above describes the symptom correctly and the cause wrongly.
There is no ghost in this window. ROAS Stacker draws none.

What sits at row 11 on frames 145-146 and 147-150 is THIS TOOL'S OWN HINT,
painted over the game by `overlay/renderer.py:draw_hint` and still on
screen when the next frame was captured. The color settles it: cell (11,0)
of frame 147 is 1560 px of `(206, 248, 253)` ringed by 442 px of exactly
`(0, 229, 255)` — `HintStyle.color` `#00e5ff` for the pen at full opacity,
and the same color at `HintStyle.fill_opacity` (0.18) over this board's own
`(251, 252, 252)` for the fill. `grep` the window for that RGB value and it
appears on 24 of the 48 frames, at cols 4-7 and cols 0-3, and nowhere else.

So the loop is tighter than "the grid flickers": the coach painted a hint,
read it back as four locked cells, committed a LOCK, re-solved, painted the
hint somewhere ELSE, found that the locked cells had moved, gave up on the
frame, reset, cleared the hint — and the now-blank board solved back to the
first hint. Five ticks per cycle, ~11 frames.

The "strong structural signal" this README pointed at — the layer occupying
the SAME COLUMNS as the falling piece — is a coincidence of this window and
does not hold. A hint marks where the SOLVER wants the piece. On
`ghost_session` 61-64 the same widget is at cols 2-3 while the O it belongs
to is at cols 4-5; here it alternates between cols 4-7 and cols 0-3 while
the I never leaves cols 0-7 of rows 1-3. A column-matching rule would refuse
the other windows and would have named the wrong half of this one.

The fix is `vision.grid._own_paint_layer`, which recognizes the paint by its
color instead of guessing at it from structure. See
`tests/test_ghost_beside_stack.py` for the replay and the before/after.
