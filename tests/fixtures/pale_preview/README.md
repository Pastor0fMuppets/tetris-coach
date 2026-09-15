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
