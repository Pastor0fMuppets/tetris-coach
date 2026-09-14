# Live session capture: the ghost piece (ROAS Stacker)

95 CONSECUTIVE ticks (frames 00056-00150) from a real session in which
tracking worked ~80% of the time. This window holds the failure that
accounts for the rest. PNGs are RGB; convert with `[:, :, ::-1]`.

Session geometry:

    board_rect = Rect(left=238, top=317, width=476, height=574)
    next_rect  = Rect(left=620, top=314, width=97,  height=96)
    rows       = 12
    compute_overlap_mask(...) -> {(0, 8), (0, 9), (1, 8), (1, 9)}

## What happens

This game draws a GHOST (landing preview) under the falling piece. Measured
cell scores against the remembered background, which are three distinct
levels, not two:

    background  0.00 - 0.02
    ghost       0.32
    solid piece 0.57

The Otsu split lands below 0.32 on some frames and above it on others, so
the ghost is read as real board content intermittently:

- 00059-00060: the ghost at rows 10-11 is read as OCCUPIED and committed as
  a LOCKED piece (a phantom lock - nothing actually locked).
- 00061: the "locked" cells MOVE (cols 0-1 -> cols 2-3), which a locked
  piece cannot do; the frame goes UNEXPLAINED.
- 00065: the ghost cells drop to 0.02 and the phantom vanishes entirely.
- 00064 onward: four identical unexplainable frames trip a BOARD_RESET,
  which adopts the observed board and absorbs the REAL falling piece into
  the stack. From there the frames read QUIET (observed == committed) with
  no falling piece, so no hint is shown.
- 00064-00148: 85 frames (~5.7 s) with no hint at all, while the board
  plainly shows a green O falling at rows 2-3 and the NEXT box shows an I.

The ghost is also what collapses the confidence measure on this game: the
intermediate cluster sits between the two classes, narrowing the gap the
confidence is computed from, which is why frames are rejected at the gate.

## After

`vision.grid._ghost_layer` names the third level and takes it out of the
split. Replayed through `CoachEngine` (tests/test_ghost_session.py):

- 0 UNEXPLAINED frames, 0 BOARD_RESETs, 3 verified locks.
- 93 of 95 frames carry a hint; the only gap is the two frames before
  anything has been committed at all.
- The O is tracked from frame 57 to 105 as the player drags it down and
  left, hinted at cols 0-1 on the floor throughout, and that is where it
  actually locks on frame 106.
- Frame 150 is still refused, on purpose: the preview there carries the
  game's one-cell round "1" badge, which scores 0.49 - a full piece
  color - so nothing can name it, and the rule refuses the whole widget
  rather than leave an unexplainable cell behind. That frame reads as it
  always did: below the gate, last hint held.

Note for anyone reading these frames by eye: the pale CYAN fill
(RGB 206, 248, 253) is the preview. The white outlines are a second,
fainter preview the classifier never sees at all - they score 0.02,
because each cell is sampled at its center and the outline is at its
edge.
