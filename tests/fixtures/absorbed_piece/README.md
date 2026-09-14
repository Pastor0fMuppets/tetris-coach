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

## The coach's own overlay is in these frames too

Frames 00296-00297 carry this tool's placement hint as a T at (10,3),
(11,2), (11,3), (11,4); frames 00298-00301 as a T at (9,1), (9,2), (9,3),
(10,2) with `_draw_rotation_badge`'s badge over (8,1); frame 00360 as a
vertical I at col 0 rows 6-9 with a badge over (5,0). It is drawn by
`overlay/renderer.py:draw_hint` and is on screen when the next capture is
taken.

Read as board content it cost a phantom lock here as well: on 296-297 the
tracker saw a T arrive at the floor — while the real pale periwinkle T was
still up at rows 0-2 — committed a `PIECE_LOCKED`, and re-solved, which
moved the hint off the piece the player was actually holding. With
`vision.grid._own_paint_layer` taking the paint out, those two frames read
FALLING, the hint holds at (10,3), and the window's confidence on them
rises from 0.372 to 0.404.

Note that the pale periwinkle T itself is untouched by that rule and must
stay so: it is 23.77 uint8 units from the hint composite, three times the
tolerance, and it is the nearest thing in any committed fixture to the
coach's own paint.
