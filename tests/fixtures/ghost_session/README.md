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
intermediate cluster sits between the two classes and narrows the gap the
confidence is computed from. That is a SECOND, separate cost, and it is
worth being exact about where it lands, because it is not here. On this
window the gate rejects 1 of 95 frames both before and after - the same
frame 150, at 0.096, refused on purpose (see below). The collapse shows
up on the other committed window: live_session, 32 of 96 frames rejected
before, 17 of 96 after. What the ghost cost THIS session was hints, not
frames: 8 of 95 before, 93 of 95 after.

## After

`vision.grid._ghost_layer` names the third level and takes it out of the
split. It names a layer only where the whole structure of a landing
preview is there, and the test that carries the weight is that the layer
must be a copy of a piece actually in flight: on every frame below the
layer is an O and an O is what hangs at the top of the board. (Without
it the rule deletes any four band-scored cells forming a tetromino at
rest with open air beside and above - an ordinary landing - and this
game has a real piece color in the band, the pale periwinkle at 0.346 in
tests/fixtures/roas_stacker.) Replayed through `CoachEngine`
(tests/test_ghost_session.py):

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

## CORRECTION: the "ghost" here is the coach's own overlay

The narrative above holds frame for frame; the attribution does not. The
translucent layer at rows 10-11 on frames 59-64 is not a landing preview
drawn by ROAS Stacker — it is this tool's own placement hint, drawn by
`overlay/renderer.py:draw_hint` and still on screen when the next frame was
captured. Every one of those cells carries `HintStyle.color` `#00e5ff` at
full opacity round its edge and the same color at `fill_opacity` 0.18 over
the board's own ground inside it.

Frame 150's "little round `1` badge" is `_draw_rotation_badge`'s, and the
digit in it is the rotation index of the hint below it.

This is why the layer "moved" from cols 0-1 to cols 2-3 on frame 61 while
the real O stayed at cols 4-5: the solver changed its mind, not the game.
`vision.grid._own_paint_layer` now names these cells by their color;
`_ghost_layer` remains for games that really do draw a preview. The frames
this window reads, and the numbers under "After", are unchanged by that.
