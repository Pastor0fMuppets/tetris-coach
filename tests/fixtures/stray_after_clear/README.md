# Live session capture: the same stutter, a Z, and a line clear

41 CONSECUTIVE ticks (frames 00545-00585) from the same session as
`hint_stutter`, same geometry. PNGs are RGB; convert with `[:, :, ::-1]`.

    board_rect = Rect(left=198, top=316, width=480, height=573)
    next_rect  = Rect(left=581, top=315, width=98,  height=93)
    rows       = 12
    compute_overlap_mask(...) -> {(0, 8), (0, 9), (1, 8), (1, 9)}

The second half of the stutter the user reported, on the other piece the
fixture corpus never had: Z. The window opens on the eight frames of a
line-clear animation (refused, correctly, as "a completed row is still on
screen"), then deals a Z at 00554 and stutters the whole way down.

    frames                                           41
    refused (the clear animation)                     8
    hint target A->B->A within 12 frames             18
    falling piece -> None -> same piece              11   all of them Z
    stack-size oscillations A B A                    23
    PIECE_LOCKED events                              12   the Z locks once

Same mechanism as `hint_stutter` and a different stray cell: here it is a
single cell at **(9, 8)**, one row below the (8, 8) of that window, in the
same column, and again nowhere near the piece it displaces. The Z is at
the top edge on (0,4) (0,5) while the stray sits at the bottom of column
8, so the two candidates share no cell and no neighbourhood.

The stray is the same thing it is in `hint_stutter`: this tool's own
rotation badge, hung at the corner of the hint's bounding box. On 00555
the hint is (9,9) (10,8) (10,9) (11,8) -- an S/Z orientation whose
bounding-box corner (9, 8) the piece does not occupy -- and the badge puts
134 hint-coloured pixels into the top margin of that empty cell. See
`hint_stutter/README.md` for the mechanism and the three fixes.

Three things this window adds over that one:

- the stray is at a DIFFERENT cell, so a fix that hard-codes (8, 8) or
  reasons about one screen position does not pass here;
- the piece it displaces is two cells wide at spawn (00554-00563), which
  is the shortest legitimate clipped sighting in the corpus. Whatever rule
  admits a partial sighting has to keep admitting this one;
- it holds the corpus' only S (falling, 00545) and its only L named by the
  game's own landing preview (00582-00585, where the L never descends past
  the top edge inside the window and three cells name no tetromino). That
  L is the first episode in the corpus the oracle names by anything other
  than shape.
