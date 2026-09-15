# Live session capture: waiting to name a piece already on screen

161 CONSECUTIVE ticks (frames 00080-00240) from a session the user called
"much closer", covering several spawn->lock cycles. PNGs are RGB; convert
with `[:, :, ::-1]`.

Session geometry:

    board_rect = Rect(left=204, top=311, width=480, height=577)
    next_rect  = Rect(left=587, top=309, width=99,  height=102)
    rows       = 12
    compute_overlap_mask(...) -> {(0, 8), (0, 9), (1, 8), (1, 9)}

## The cost, measured over the whole 705-frame session

    OCCLUDED frames                 164 / 705  (23%)
    spawn latency, first partial
      sighting -> first hint        median 13 frames = 0.87 s
      (samples: 15 13 15 8 9 9 8 14 14 12)
    preview readable                493 / 705  (69%)

A piece entering from above the board shows only 1-3 cells at first. Those
fit several tetrominoes (two adjacent cells fit O, S, Z, J and L), so
`explain_grid` deliberately holds rather than guess, and the frame reads
OCCLUDED. The coach therefore shows nothing for roughly the first second
of every piece, until enough of it has descended to be unambiguous.

The user, who is the one watching it: "The hint doesn't show until the
piece has fully dropped both rows onto the board. Ideally we would store
the piece from the next piece viewer to speed things up so it can show
when the new piece is only partially visible as it is falling."

## The signal that is already there and unused

The piece now entering is the piece that just LEFT the preview box. The
tracker already reads the preview (69% of frames here) and already keeps
`_preview_piece` / `_entering_hint`. What it does NOT do is let that
identity resolve an ambiguous clipped fragment at spawn time.

The known hazard, from the earlier work that deferred this: the signal is
only sound at the MOMENT OF ENTRY. On the frames after, the committed
`next_piece` has already advanced to the piece after it, so naming a
fragment from the CURRENT next_piece would name the wrong piece - which is
exactly the "confused two pieces" failure to avoid.
