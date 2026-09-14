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
