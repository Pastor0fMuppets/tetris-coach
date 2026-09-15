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

## What it costs now

The signal is used, and the thing that was throwing it away turned out
to be the BOARD RESET. The game wipes the field, deals an O, and the box
flips O -> I on 00158 — the same frame the O's first two cells appear at
the top edge. Four frames later the resync fires on that same run of
frames and dropped the hint naming the very piece it was resyncing onto.
A resync now keeps a hint set INSIDE the run it is adopting, and drops
only one from before it.

Measured over this window (161 frames, 124 past the confidence gate):

    OCCLUDED frames                 43 -> 31  of 124 accepted
    frames with a hint on screen   108 -> 120
    sighting -> first hint          17 -> 17   frames   no flip seen
                                    14 ->  2   frames   the flip names the O
                                    16 -> 16   frames   the flip is a deal late

Two frames is the commit debounce, not a wait: the name is available on
00162 and the tracker commits any candidate on its second identical
frame. The O is named with two of its four cells still above the
capture — which is what the user asked for.

The two episodes that do not move are the two with no sound evidence,
and they must not move. 00086 has seen no preview CHANGE at all (the box
holds an O from the first frame, and None -> X says nothing about what
was dealt). 00217's flip is I -> O and it is a deal LATE, because of the
pale T above: naming the fragment from it is the "confused two pieces"
failure.

## What keeps the late flip from being believed

Three rules, all about DATING a flip rather than counting what it
outlives, since a flip says "the piece that was here has been dealt" and
never says when:

- a flip counts only when the PREVIOUS frame read the box too. Two
  consecutive readable frames cannot straddle two deals (a tenure here
  is ~14-22 frames), which is exactly what the 18-frame gap at
  00198-00215 did.
- a hint survives a lock only while it is younger than that lock's own
  debounce. The flip reporting a deal lands on the frame the lock commit
  is debouncing, so this deal's hint is the young one and an older hint
  names the piece that just locked. Counting locks could not separate
  those: both show exactly one lock since the hint was set.
- a hint is dropped by any frame that REFUTES it — the fragment is a
  piece entering from above and no placement of the hinted piece fits
  it. A preview reading is a hypothesis, and a hint briefly withheld
  beats one confidently wrong.
