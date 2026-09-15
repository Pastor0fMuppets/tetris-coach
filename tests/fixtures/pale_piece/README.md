# Live session capture: a pale piece read as background (ROAS Stacker)

61 CONSECUTIVE ticks (frames 00640-00700), the tail of a session whose
jitter was fixed but which still showed a 6-second dropout. PNGs are RGB;
convert with `[:, :, ::-1]`.

Session geometry:

    board_rect = Rect(left=276, top=314, width=477, height=578)
    next_rect  = Rect(left=659, top=317, width=98,  height=95)
    rows       = 12
    compute_overlap_mask(...) -> {(0, 8), (0, 9), (1, 8), (1, 9)}

## The failure

A pale periwinkle T sits at rows 1-2 for this entire window. It is NOT
detected: the occupancy holds only the blue stack at cols 0 and 9, and the
classifier reports confidence 0.58 - it is confidently wrong, not unsure.
No falling piece, so no hint, for 90 frames (~6 s).

    frames 648-680   r8-r11 "#........#"   <- only the blue stack
                     the pale T at rows 1-2 is absent entirely

## Why it is hard

Measured scores against the remembered background on this theme:

    background        0.00 - 0.02
    ghost             0.320
    PALE PIECE        0.346     <- a REAL piece
    solid piece       0.57 - 0.82

MIN_SPREAD is 0.35, sitting between the ghost and the pale piece, and the
two are four thousandths apart. No threshold on this scale can separate
them - a cut low enough to admit the pale piece admits the ghost too. Only
STRUCTURE can tell them apart, which is what `_ghost_layer` exists to do.

Here the pale piece never reaches that rule: the Otsu split puts the
threshold above 0.346 (the solid pieces at 0.8 dominate the split), so the
cells land in the EMPTY class and are never candidates for anything.

Note also tests/fixtures/roas_stacker, which contains the same periwinkle
colour as settled stack content - a fix must keep those cells too, and
must not start reading ghosts as pieces.
