# Live session capture: a reset absorbs the falling piece (ROAS Stacker)

81 CONSECUTIVE ticks (frames 00280-00360) from the session that tracked
~80% of the time. This window holds the dominant remaining failure.
PNGs are RGB; convert with `[:, :, ::-1]`.

Session geometry:

    board_rect = Rect(left=238, top=317, width=476, height=574)
    next_rect  = Rect(left=620, top=314, width=97,  height=96)
    rows       = 12
    compute_overlap_mask(...) -> {(0, 8), (0, 9), (1, 8), (1, 9)}

## The self-sustaining loop

The falling piece here is a PALE PERIWINKLE T, scoring 0.346 against the
background - a real piece whose colour sits barely under MIN_SPREAD (0.35)
and inside the ghost band. It IS detected (frame 00330 classifies at
confidence 0.404 with the T present), so this is NOT a contrast failure.

The failure is that the T ends up committed as STACK, and then cannot get
out:

    frame 307  obs r0 "...#......"  comm r0 "...#......"   QUIET
               obs r1 "..###....."  comm r1 "..###....."   (T is stack)
    frame 309  obs r1 "...#......"  comm r0 "...#......"   UNEXPLAINED
               obs r2 "..###....."  comm r1 "..###....."   (T moved down)

Committed stack cells appear to VANISH from rows 0-1 and reappear at rows
1-2. Stack cells cannot vanish, so the frame is unexplainable; four
identical unexplainable frames trip a BOARD_RESET; the resync adopts the
observed board, which re-absorbs the T at its NEW position; the next frame
it falls again and the whole cycle repeats.

The piece therefore never reads as falling, and no hint is shown for its
entire descent: frames 00301-00358, 58 frames, ~3.9 s.

## Why the existing guard does not catch it

`strip_entering_piece` deliberately keeps only a CLIPPED (1-3 cell)
row-0-touching fragment out of a resync's stack; a FULLY VISIBLE piece is
absorbed by design. Here the T is complete (4 cells at r0c3 + r1c2,c3,c4)
and touches row 0, so it is absorbed.

Note the general shape of the bug: any reset that swallows the falling
piece creates this loop, because the piece's own motion then looks like
stack cells vanishing.
