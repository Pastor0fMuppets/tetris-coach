# Live session capture: the preview goes blind on pale pieces

Sampled frames from the session geometry:

    board_rect = Rect(left=204, top=311, width=480, height=577)
    next_rect  = Rect(left=587, top=309, width=99,  height=102)
    rows       = 12

PNGs are RGB; convert with `[:, :, ::-1]`.

## The failure

`identify_next` returns None on 226 of 705 frames (32%), in runs up to 45
frames: (565, 45), (273, 41), (477, 32), (1, 29), (675, 26), (398, 23).

next_00580.png plainly shows a T. `identify_next` returns None for it.

The cause is the same pale periwinkle that was invisible on the BOARD until
the intermediate-band work (see tests/fixtures/pale_piece). That fix taught
`grid.py` to hand the band to structure instead of to a threshold. The
PREVIEW path never got it: `identify_next` still thresholds, so a pale
piece in the box reads as an empty box.

## Why it matters now

The entering-piece acceleration (tests/fixtures/spawn_latency) names a
half-visible spawn from the piece that just LEFT the preview. It needs the
box readable on consecutive frames around the deal. Measured on this
session, it usually is not, so the acceleration rarely fires:

    spawn_latency window   one episode 14 frames -> 2 frames
    this whole session     median 13 frames -> 12 (0.87 s -> 0.80 s)

The hint logic is not the bottleneck. The preview reading is.

## What fixed it

`identify_next` stopped anchoring its only mask at the uniformity floor.
When the thresholded reading names nothing, the intermediate band
`[MIN_SPREAD/2, MIN_SPREAD)` is read on its own and handed to the same
shape rules — the board's fix, one region over, shaped for a box:

- the band is SPLIT first (Otsu over the band's own pixels), because a
  crop carries furniture a board cell's patch mean never sees: the box's
  hairline border and gridlines score 0.179-0.250 here, under the piece
  at 0.254-0.349 and touching it. Taken whole, the band merges the
  piece's cells through that border — 0 of these 15 crops readable, and
  0 of `pale_piece`'s 61.
- the band's own ceiling is what keeps the caption out: the "NEXT" core
  scores 0.586, solid class, so only its skirt is in the band and a
  hollow outline is not block-like.
- the coach's own hint fill is refused outright: over a light box it
  composites to 0.321, four thousandths from the piece, and the panel
  floats over the board corner where a hint can be drawn.

Measured on this window, before -> after:

    named                    0 of 15 -> 13 of 15, every one a T
    next_00580                  None -> T
    tests/fixtures/pale_piece   0/61 -> 47/61
    all committed crops      452/557 -> 530/557
    wrong pieces                   0 -> 0

The two still unnamed (00273, 00280) are a deal animation mid swap: the
outgoing piece's panel slides over the incoming one and the crop has no
background left to estimate from. Nothing there is a piece to name.

And what it bought the thing it was blocking — the entering-piece
acceleration on `tests/fixtures/spawn_latency`, whose third episode had
no evidence because this box could not be read at the deal:

    sighting -> hint    16 frames -> 2      (the T, named from the flip)
    preview read        132 of 161 -> 150 captures
    frames with a hint  120 -> 134          OCCLUDED 31 -> 17
    PIECE_LOCKED 2 -> 2 BOARD_RESET 2 -> 2  wrong names 0 -> 0

The other two episodes of that window do not move, and cannot: one opens
on a piece already at the top edge with no flip ever seen, and one is a
deal late. 17 -> 17 and 2 -> 2.

## What it does to every other committed window: nothing

Every window replayed end to end through `CoachEngine`, before -> after
(frames hinted / PIECE_LOCKED / BOARD_RESET / rejected at the gate /
frames whose hint names a piece other than the committed falling one):

    absorbed_piece      66 / 1 / 1 /  5 / 0    (identical)
    ghost_beside_stack  31 / 0 / 1 /  5 / 0    (identical)
    ghost_session       93 / 3 / 0 /  1 / 0    (identical)
    live_session        75 / 4 / 0 / 17 / 0    (identical)
    pale_piece          45 / 0 / 1 / 14 / 0    (identical)
    spawn_latency      120 -> 134 / 2 / 2 / 37 / 0
    pale_preview         0 / 0 / 0 /  3 / 0    (identical; these frames
                        are sampled, not consecutive, so the engine has
                        no continuity to read here — the crops are the
                        evidence in this window, not the replay)

Only `spawn_latency` moves at all, and only in the direction the box
being readable buys: 14 more frames with a hint on them. Sighting ->
hint distances are unchanged everywhere else (absorbed_piece 15,
ghost_session 2, live_session 2 — before and after), no window gains or
loses a lock or a reset, and no frame in any window carries a hint for a
piece other than the one the tracker has committed as falling.
