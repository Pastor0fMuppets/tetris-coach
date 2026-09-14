# Live session capture (ROAS Stacker)

96 CONSECUTIVE ticks (frames 00040-00135) from a real failing live session,
dumped by `tetris-coach --dump-frames`. Consecutive matters: the tracker
explains each frame as a diff against the previous committed one, so a
sampled set can only ever replay as UNEXPLAINED.

PNGs are RGB; the capture pipeline produces BGR. Convert with `[:, :, ::-1]`.

Session geometry:

    board_rect = Rect(left=303, top=313, width=477, height=579)
    next_rect  = Rect(left=686, top=316, width=94,  height=94)
    rows       = 12
    compute_overlap_mask(...) -> {(0, 8), (0, 9), (1, 8), (1, 9)}

What the window contains:

- 00043: the I piece is tracked correctly (FALLING, hint I). The user
  confirmed this first hint was right.
- 00054-00058: the I descends, row 1 -> row 2. Still tracked.
- 00059: the I hard-drops to row 11 AND a green O spawns, of which only the
  bottom two cells (row 0, cols 4-5) are inside the board region - its top
  half is above the capture. Cols 8-9 of row 0 are the NEXT panel.
- 00059-00067: byte-identical grids. This game has effectively no gravity
  (drag to move, drag to drop), so a spawning piece SITS at the top edge.
- 00120: a piece is briefly tracked again before the session degrades.

The preview crops in this window: next_00040-00058 show a yellow-green O
(2x2 of 19 px cells), next_00059-00109 a blue horizontal I (4x1), and
next_00110-00135 the O again. The flip at 00059 is the frame above: the I
leaves the preview as it is dealt onto the board. Each crop also carries a
faint grey "NEXT" caption in its top-left corner and a lot of white space —
the piece is neither centered nor anywhere near filling the box, which is
what Bug 2 was about.
