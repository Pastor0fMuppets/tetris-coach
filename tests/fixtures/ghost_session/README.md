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
