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

## What the band is not allowed to do

Reading the band is reading what the threshold refused, so the box's own
furniture arrives with the pale piece. Three rules bound it, and the
measurements behind them are in `tests/test_pieces_vision.py`:

- the FLUSH hypothesis is off (one band divided into cells by assertion,
  which makes every solid rectangle a piece). With it on, an empty box's
  inner well read 'O' at every panel/well shade pair tried (0.223-0.307),
  a pale caption bar above the piece read a confident 'I' on 150 of 396
  bar geometries — the band keeps its upper class, and nothing says the
  furniture sits under the piece — and a lone rectangle was named on 162
  of 540 geometries. It costs a flush skin its PALE O and I; of the 530
  crops named across every committed window, none is named that way.
- a cell read out of one band is measured edge to edge, because that is
  what flush claims. Across the style matrix 1128 of 1169 flush readings
  fill their cell rectangles >= 0.90 (688 exactly); 407 round blobs reach
  0.847 at most, and the disc is this tool's own rotation badge, which
  lands in the box and read as an 'O' at every radius, on every theme, in
  both passes. It costs the 41 flush readings under the floor: small
  cells under a caption, none of them a committed crop.
- our own hint fill is refused in BOTH readings of the box and over the
  box's own levels, not its average: the composite is a distance from the
  ground, so over a black box it scores 0.374 and arrives as a solid
  class, and a box drawn as a panel around a well puts the crop median on
  one shade while the paint lands on the other.

What is still named and is not a piece, unchanged from before any of this
(all of it in the THRESHOLD pass, where a lone rectangle is still read as
a flush O or I): an empty panel-and-well box whose well is at solid
contrast reads 'O', a lone solid rectangle is named on 42 of 135
geometries, and four round badges in a row read 'I'.

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

## What else is in this window: board_00477, the badge over a piece

Sampled and not consecutive, this window is left out of the oracle and of
the race, and it has been left out of measurements it should have been in.
The admission rule in `ColourTracker._rank` (a sub-tetromino sighting is
only admissible where something could be hiding the rest) was committed
with "measured over all nine committed windows, every one of the 170
sub-tetromino sightings the tracker picks is explicable" — measured over a
corpus this window was not in. Replayed, it holds 5 more such sightings,
and one of them was the counterexample:

    board_00477   falling T at (2,1) (3,1) (3,2)

The fourth cell is (4,1), and (4,1) is not empty. Its commonest colour is
the piece's own pale periwinkle — the same value as (3,1), 1073 of its
pixels — under 520 pixels of `#00e5ff`: this tool's own rotation badge,
from the days it was hung in the cell ABOVE the hint's top-left corner,
which is a cell the falling piece passes through. `Palette.classify` skips
a cell our opaque drawing covers, so the cell arrives EMPTY whatever the
game drew in it, and the sighting is three cells in open board.

What refusing it costs is not a blank frame, it is a wrong one. With the
top edge and the NEXT panel as the only hiding places, this frame reads

    falling I at (4,0) (5,0) (6,0) (7,0)

— a grounded column of the settled stack — and hands the solver a board
with a four-cell hole in column 0: a hint solved for a piece that is not
in flight. So `_hidden` has a third place in it now, our own opaque paint,
and this frame is the captured evidence for it (`tests/
test_colour_tracker_sessions.py::test_a_piece_cell_under_our_own_badge_is_read_as_hidden`).

With that place counted, all 175 sub-tetromino sightings across all nine
windows are explicable, and turning the rule off changes no frame of any
window — this one included.
