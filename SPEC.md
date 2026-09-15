# Tetris Coach — Architecture & Build Spec

## Product summary

A macOS desktop tool. The user starts the program, drags a selection rectangle over
the board area of *any* Tetris game visible on screen (plus a second rectangle over
the next-piece preview), and the tool overlays the optimal placement of the current
piece on top of the game. Purely visual — capture in, overlay out, no game input.
Purpose: training human placement intuition at full game speed.

## Hard requirements

- **Game-agnostic**: works on any Tetris rendering a board 10 wide × a configurable
  number of rows (default 20, `--rows`) with reasonably solid cell colors
  distinguishable from the board's background — dark, light, and colored themes
  alike. No per-game config beyond the user-selected regions and the row count.
- **Fast**: end-to-end capture→overlay under ~100 ms; hint for a newly spawned piece
  visible within ~1 frame in the common case (via precomputation, see below).
- **Visual only**: never send input to the game.

## Module layout (`src/tetris_coach/`)

```
core/
  board.py        # Bitboard: 10 cols × configurable rows (default 20), one int
                  # per row (bits 0..9); the height flows from len(rows).
                  # Ops: occupancy from bool array, drop piece, clear lines,
                  # column heights, hole count, transitions, wells.
  pieces.py       # 7 tetrominoes; all distinct rotations as (rotation, cells) with
                  # per-rotation column masks/bottom profiles for fast drops.
solver/
  evaluate.py     # Dellacherie features: landing height, eroded piece cells,
                  # row transitions, column transitions, holes, cumulative wells.
                  # Classic Dellacherie weights as defaults; weights in a dataclass.
  search.py       # enumerate all (rotation, column) placements of current piece;
                  # for each, resulting board -> best next-piece placement score
                  # (2-ply lookahead, no pruning needed at this size: ~34×34 evals).
                  # Must run < 50 ms in pure Python via bitboards; benchmark it.
vision/
  grid.py         # Given BGR image of board region + (rows, cols): per-cell
                  # occupancy via color distance from a background estimate,
                  # split by Otsu. Returns (rows × 10) bool array + confidence.
                  # The estimate is a cross-frame memory (GridClassifier:
                  # the empty class of every accepted frame re-measures it),
                  # bootstrapped from the top-row cell-color median. That
                  # bootstrap holds only while a MAJORITY of the sampled
                  # top-row cells really are background; past that the median
                  # locks onto piece colors and the whole board inverts, so a
                  # self-estimated reading is vouched for only when its own
                  # top row is consistent with the premise: a strict majority
                  # of the OBSERVABLE top-row cells read empty, those cells
                  # are ONE PIECE IN FLIGHT (all airborne — not 4-connected
                  # down to the floor — all in the same airborne component,
                  # and that component no bigger than a tetromino), and
                  # board-wide no more than a tetromino's worth of cells
                  # hangs OVER THE VOID: airborne with nothing at all below
                  # them in their column. A piece spawning or falling
                  # through row 0 fits that and keeps full confidence; a
                  # stack grounded at row 0 (legal: side columns stacked to
                  # the top, versus garbage pushed up) does not and is
                  # capped to 0.0, because an inverted reading's occupied
                  # cells are the true background, which runs from row 0
                  # down to the stack. A count alone separates nothing here:
                  # benign and inverted readings BOTH show a minority
                  # occupied, which is why the earlier "any occupied top-row
                  # cell caps" rule was total, and why a plain majority rule
                  # is no rule at all (measured: it lets monochrome-theme
                  # inversions through at 0.95). Nor is a budget on what
                  # hangs off row 0 alone: the air over a stack topping out
                  # at rows 0 and 1 is four cells, a tetromino exactly, so
                  # the whole-board rule is what refuses it (measured: 142
                  # of 300 seeded legal near-top-out boards read INVERTED
                  # above the gate, up to 0.97, under the per-column budget;
                  # 0 under this one). The budget is spent on cells over the
                  # void rather than on every airborne cell because a
                  # hole-riddled real stack falls into floating bands that
                  # are not pieces (measured: budgeting those refused 257 of
                  # 360 legal piece-in-flight frames).
                  # The classifier stays pure of layout, but not of what the
                  # capture cannot see: it takes the same unobservable-cell
                  # set the engine does. When a game floats its NEXT preview
                  # over the board's top corner (inside the selected board
                  # region — ROAS Stacker does this at cols 8-9, rows 0-1),
                  # those cells show the NEXT piece, not the board;
                  # app.compute_overlap_mask names them from the two rects,
                  # grid.py leaves them out of the top-row median, out of the
                  # top-row cap and out of the memory's re-measurement, and
                  # the engine discards their reading (see app.py). Read as
                  # board content they added a second tetromino of cells every
                  # frame -> UNEXPLAINED -> spurious BOARD_RESET; counted in
                  # the top-row prior they made the top row permanently
                  # occupied -> every frame capped -> the memory, which
                  # anchors only from ACCEPTED frames, could never form. That
                  # second one is a deadlock, not a degradation: measured on
                  # a real session, 36 of 36 frames at confidence 0.00.
                  # The memory is usable after one accepted two-class frame
                  # (CONFIRMED: it stops falling back, which is what gates a
                  # bright overlay) but not settled until a second frame has
                  # CORROBORATED it — until then every accepted frame is also
                  # read from scratch by the top-row prior, and a vouched
                  # reading naming a different background drops the memory and
                  # the frame both (one of the two is inverted and nothing
                  # says which). One frame cannot settle it because a legal
                  # near-top-out board can be read inverted AND vouched for:
                  # attached mid-game on one, the coach anchored on a piece
                  # color and, since confirmed memory never falls back, read
                  # every later frame 120 of 120 cells wrong at 0.96 for the
                  # rest of the session. Frames the prior refuses are no
                  # evidence and leave the memory alone, so a game whose top
                  # row is never vouchable still runs on a confirmed,
                  # never-corroborated anchor. Cost: ~0.08 ms/frame.
                  # Corroboration is necessary and NOT sufficient: two
                  # frames are independent as CAPTURES but not as BOARDS,
                  # and the shape that inverts is a stack, which persists
                  # across ticks 67 ms apart — frame two's own inverted
                  # reading corroborates frame one's wrong anchor
                  # (measured: holding that board for two frames, or
                  # playing five different boards of the same family,
                  # wedges the session permanently at 116 of 116
                  # observable cells wrong). So at EVERY stage,
                  # corroborated included, a frame the anchor reads as a
                  # board containing a COMPLETED ROW is refused: a
                  # completed row clears the instant it completes, and
                  # inverted, the empty air above an ordinary stack reads
                  # as row after row of them. That contradiction needs no
                  # prior and no second frame, and it is loudest on
                  # exactly the ordinary frames a near-top-out bootstrap
                  # lacks. The anchor is then asked for a second opinion,
                  # which an ordinary frame answers at once (its own top
                  # row is clean, so the prior reads it from scratch and
                  # names the true background, dropping the anchor on the
                  # spot); a frame with no opinion only counts against it,
                  # and 15 in a row — ~1 s, past any clear animation —
                  # drop it. Rows a UI panel touches never count, since
                  # versus garbage is 9/10 filled and its one gap can sit
                  # behind the panel.
                  # Masking has a limit: cover the WHOLE top row and there is
                  # no sample to bootstrap from at all, so every frame is
                  # refused (before, the "no occupied top-row cells" exit
                  # vouched for a background read off the covering panel —
                  # measured: a board filled from row 8 down read as rows
                  # 1-11 fully occupied at 0.72). app.selection_warning says
                  # so on stderr at startup, because that mask is what an
                  # ordinary mis-selection produces: a NEXT queue drawn as a
                  # bar across the top of the playfield.
                  # THIRD LEVEL: a board can show a translucent layer whose
                  # cells score BETWEEN the background and a real piece,
                  # and Otsu has only two classes to give them to.
                  # Measured on tests/fixtures/ghost_session: background
                  # 0.00-0.02, layer 0.32, solid 0.57-0.82, with the split
                  # landing either side depending on what else is on the
                  # board. Read as content that layer is a tetromino that
                  # TELEPORTS, which the tracker cannot explain: two
                  # identical frames of it are exactly its debounce, so it
                  # committed a phantom LOCK, the next drag "moved" locked
                  # cells, and four unexplainable frames later BOARD_RESET
                  # adopted the observed board and swallowed the real
                  # falling piece (measured on that window: 52 of 95 frames
                  # UNEXPLAINED, 9 resets, 8 frames with a hint, an 85-frame
                  # / 5.7 s stretch with none). The same intermediate
                  # cluster also sits inside the class gap confidence is
                  # measured from, so readable frames were reported
                  # ambiguous and dropped at the gate — a SEPARATE cost,
                  # and one that shows up on the other committed window
                  # rather than this one (live_session 32 of 96 frames
                  # rejected, ghost_session only 1 of 95).
                  # WHERE THAT LAYER COMES FROM was diagnosed wrong, and
                  # the correction is the point of _own_paint_layer. It
                  # was read as the GHOST almost every modern Tetris
                  # draws — the landing preview under the falling piece.
                  # In every committed fixture it is THIS TOOL'S OWN
                  # HINT: overlay/renderer.draw_hint paints it over the
                  # game, the window is on screen when mss takes the next
                  # shot, and the coach reads its own output back as
                  # board content. The pixels settle it — on
                  # ghost_beside_stack frame 147 cell (11,0) is 1560 px
                  # of (206,248,253) ringed by 442 px of exactly
                  # (0,229,255), i.e. HintStyle.color #00e5ff for the pen
                  # and the same color at HintStyle.fill_opacity over
                  # that board's own (251,252,252) for the fill. Measured
                  # over every observable cell of all four session
                  # windows plus the roas_stacker frames (39040 cells):
                  # of the 582 that reach the band below, 272 are this
                  # overlay and the other 310 are the real pale
                  # periwinkle T of absorbed_piece. Not one is a
                  # game-drawn ghost — ROAS Stacker draws none.
                  # _own_paint_layer therefore names it by ARITHMETIC
                  # rather than by structure. The fill is a straight
                  # alpha composite, so its color over a known
                  # background is known too (background + opacity *
                  # (paint - background)), and a cell within 8.0 uint8
                  # units of that is our paint. That separates three
                  # clusters, not two: the fill over the board's own
                  # ground at 0.470-0.565, the fill over a REAL PIECE at
                  # 58.41-240.63, and the nearest cell with no hint on
                  # it at 23.77 (the periwinkle). The middle cluster is
                  # what makes it safe to delete anything at all — a
                  # hint drawn ON TOP of real content composites
                  # differently and is never taken out (live_session
                  # 59-62, where the I hard-drops onto the very square
                  # the hint was marking), so the claim is only ever
                  # "this cell is empty board with our paint on it".
                  # Two structural tests remain, both about what is
                  # around the paint: it must be exactly one tetromino
                  # (the guard for a partial match — a widget the panel
                  # mask cuts in half, or one lying half over content —
                  # which refuses the whole thing rather than deleting a
                  # fragment), and nothing solid may sit directly ABOVE
                  # it. The second is not arbitrary: a hint marks a hard
                  # drop's landing square and a piece reaches one by
                  # falling down its own columns, so the cells above it
                  # are empty by construction — except for the ROTATION
                  # BADGE, which is this tool's paint too but drawn
                  # opaque with a black digit through it, so it matches
                  # no composite and nothing can name it. Naming the
                  # hint under it would leave the badge as an
                  # unexplainable added cell (measured: four UNEXPLAINED
                  # frames, a BOARD_RESET, and the badge committed to
                  # the stack), so the whole widget is refused and those
                  # frames read as they always did — below the gate,
                  # last hint held.
                  # The rule needs no score band and must not have one:
                  # on the fixtures' near-white ground the fill scores
                  # 0.321, inside the band, but on a BLACK ground the
                  # same composite scores 0.374, above MIN_SPREAD
                  # entirely, and over grey grounds it runs 0.284-0.374.
                  # The theme decides where the paint lands, which is
                  # exactly what a rule keyed to the paint itself does
                  # not care about.
                  # The real fix is for the capture never to contain the
                  # overlay: that is platform-specific window-exclusion
                  # work in capture/, and this layer keeps the reading
                  # correct whether or not it lands. app.py passes its
                  # configured hint_color in, so a session run with
                  # --hint-color looks for what it actually drew; a color
                  # string this module cannot parse (Qt takes names too)
                  # turns the rule off rather than guessing.
                  # _ghost_layer below is now the FALLBACK, for the games
                  # that really do draw a ghost. The two are alternatives
                  # rather than a union on purpose: a frame carrying both
                  # our paint and a real ghost has the paint taken out
                  # and the ghost left in, which is a held frame — the
                  # direction every rule here errs in — and no committed
                  # fixture contains one to write a composition against.
                  # _ghost_layer names that level instead, and it is taken
                  # out of the split entirely, so both the threshold and the
                  # confidence are measured on the separation that decides
                  # occupancy: background versus a real piece color.
                  # The score band [MIN_SPREAD/2, MIN_SPREAD) only nominates
                  # a CANDIDATE; it can never be the answer, because this
                  # same game has a real piece color inside it — a pale
                  # periwinkle at 0.346 against the ghost's 0.320, four
                  # thousandths under the floor (on the stack in
                  # roas_stacker's live2_board_00500 and 00600 a hundred
                  # ticks apart, and caught in mid-air, unambiguously
                  # falling, in 00800). No threshold on this scale
                  # separates those; only where the cells SIT can. So a
                  # candidate is not NAMED unless the whole structure of
                  # a preview is there (and a candidate no rule names is
                  # content, not background — see below): clear of the
                  # background cluster by
                  # MIN_SPREAD/2 (the background's own spread is 0.02, a
                  # ghost's clearance 0.21-0.30), exactly one tetromino,
                  # RESTING on the floor or on a piece color, with no
                  # solid cell to its left, to its right, or above it, and
                  # a copy OF something — the piece it previews must be on
                  # the board, in flight: a whole tetromino of the SAME
                  # PIECE TYPE, airborne (not 4-connected to the floor).
                  # That last one is the structure a landing preview
                  # cannot be without, since a game draws a ghost because
                  # a piece is falling, and it is what separates the two
                  # cases on the real pixels: 21 of 21 named frames across
                  # both committed sessions have their own piece in the
                  # air (an O over ghost_session's O layer, an I over
                  # live_session's I layer), while the pale periwinkle
                  # that must NOT be deleted sits under a falling J.
                  # Those 21 frames turned out to hold this tool's own
                  # hint rather than a game ghost, and the test survives
                  # the correction for the same reason it worked: a hint
                  # is a PLACEMENT of the falling piece, so it is a copy
                  # of a piece in flight exactly as a ghost is. What the
                  # correction does cost is the evidence — no committed
                  # fixture holds a game-drawn ghost any more, so this
                  # rule is now argued from the shape space rather than
                  # from pixels, and _own_paint_layer is what the real
                  # sessions are carried by.
                  # Position alone does not carry it: the neighbour test
                  # saves the periwinkle on live2_board_00500 only because
                  # the stack abuts it there, and ONE legal board
                  # difference (that abutting cell gone) had the rule
                  # delete four real locked cells at confidence 0.26.
                  # The neighbour test is the other one the band cannot
                  # do: a preview
                  # marks space the piece can still drop into, so it is the
                  # topmost thing in its own cells with open air either
                  # side, while a piece the stack has grown around got there
                  # by being played. Support from BELOW stays legal, so a
                  # ghost resting on a flat stack surface is still named; a
                  # ghost in a WELL is not, since its sides touch and it is
                  # the same picture as a piece played into the notch.
                  # ANY solid neighbour refuses, not merely a grounded one,
                  # because a one-cell round badge can float directly over
                  # the layer: the badge is furniture too but scores 0.49,
                  # so nothing can name it, and naming the layer under it
                  # left the badge as an unexplainable added cell
                  # (measured: 4 UNEXPLAINED frames, a BOARD_RESET, and
                  # the badge committed as a phantom). The badge is not
                  # the game's — it is overlay/renderer's own rotation
                  # badge, carrying the rotation index as its digit, and
                  # _own_paint_layer refuses those widgets for the same
                  # reason. On a game that really does draw a ghost the
                  # test costs a ghost with anything beside it, which is
                  # the trade the rest of this rule is written around.
                  # Unobservable cells are left out of the band, out of the
                  # background cluster, out of the neighbour tests and out
                  # of the pieces counted as in flight (a piece a panel
                  # cuts in half is not a whole tetromino), for the same
                  # reason they are left out of everything else
                  # (measured: the covered cell (0,8) scores 0.15 against a
                  # background otherwise topping out at 0.02, which alone
                  # put a real ghost 0.005 inside the clearance margin and
                  # refused every frame of the live session).
                  # Every test errs toward LEAVING THE CELLS ALONE, because
                  # the costs are not symmetric: a ghost left in costs held
                  # frames, while real content taken out deletes stack and
                  # hands the solver room that does not exist. The two
                  # residual errors are the opposite corners — a ghost drawn
                  # opaque enough to leave the band, or resting in a well,
                  # stays content and the frame is simply held (the common
                  # one, and the preferred direction — and since the band
                  # flip below, a GUARANTEED one rather than whichever
                  # side of the split Otsu happened to put it); a real
                  # band-colored
                  # piece freshly landed on a FLAT surface with open air
                  # beside and above it is deleted IF the piece now
                  # falling is the same type. The neighbour test alone
                  # does not make that narrow — measured over every legal
                  # placement on 4000 random stacks (644563 landings), 49%
                  # come to rest with nothing occupied beside or above
                  # them, i.e. a coin flip — which is why the piece must
                  # also be in flight: roughly one landing in seven, none
                  # at all between a lock and the next spawn, and over the
                  # moment a neighbour arrives.
                  # A named layer also caps the frame's confidence at 0.5,
                  # the same number a uniform-empty frame reports, because
                  # both mean "structurally grounded rather than measured":
                  # the gap is taken with the layer out of BOTH classes,
                  # so without the cap a three-level frame reports like a
                  # clean two-level one (measured on a synthetic board
                  # where the rule deletes content: 0.94). It is a
                  # signature, not a second defense — the same board reads
                  # 0.58 with the rule off, and the conservative measure
                  # that would flag it (layer counted in the empty class)
                  # reads 0.078 on live_session's fifteen REAL ghost
                  # frames, which is the pre-rule number that had them
                  # rejected. The rule is the defense.
                  # WHAT HAPPENS TO A CANDIDATE NO RULE NAMES is the
                  # other half of the band, and it used to be nothing:
                  # the cells went back into the ordinary split and the
                  # threshold decided them. That is the one thing the
                  # threshold provably cannot do — it is why this band
                  # exists — and on tests/fixtures/pale_piece it
                  # decided wrong in the expensive direction. Otsu
                  # maximizes between-class variance, the blue stack at
                  # 0.815 dominates it, the cut lands above 0.346, and a
                  # real pale periwinkle T spends 61 frames in the EMPTY
                  # class: not misread, ABSENT. No falling piece, no
                  # hint, 90 frames (~6 s). Two settled cells of the same
                  # colour went with it, so the floor the solver was
                  # handed read "#........#" for a board that is
                  # "#....#..##".
                  # So the band is resolved by structure end to end, and
                  # every candidate gets one of three dispositions:
                  #   NAMED a landing preview (by _own_paint_layer's
                  #     arithmetic or _ghost_layer's structure) -> EMPTY,
                  #     out of the split, nothing is there.
                  #   OUR OWN PAINT that _own_paint_layer REFUSED to name
                  #     (a widget the panel cut in half, or one under the
                  #     rotation badge) -> the FRAME is refused, 0.0. It
                  #     cannot be deleted (the badge above it would be
                  #     stranded as an added cell nothing can explain)
                  #     and it cannot be called content (it is this
                  #     tool's own overlay). A band that cannot be
                  #     resolved is a frame that must not be
                  #     committed, and losing the confidence is how
                  #     it says so rather than guessing.
                  #   EVERYTHING ELSE -> OCCUPIED, up to
                  #     _content_budget. Board content until something
                  #     says otherwise; past the budget, the FRAME is
                  #     refused, exactly as for our own paint above.
                  # That last default is the flip, and the errors it
                  # trades between are NOT symmetric. A ghost read as a
                  # piece is a tetromino that teleports: a phantom lock,
                  # unexplainable frames, a BOARD_RESET, a jittering
                  # hint — bad, and BOUNDED, because the tracker refuses
                  # to explain it and holds. A piece read as background
                  # is not on the board at all, and nothing downstream
                  # can recover what it was never told. The structural
                  # rules still get first refusal, so a preview they can
                  # name is still deleted; what moved is where the
                  # silence falls.
                  # BOTH HALVES OF THAT ASYMMETRY ARE ABOUT ONE PIECE,
                  # which is what _content_budget holds the default to.
                  # At board scale there is no asymmetry left: a phantom
                  # the size of the playfield is not held, it is a new
                  # board, and four frames of it is a BOARD_RESET that
                  # commits the phantom as stack. And board scale is
                  # exactly what the band admits, since all it asks of a
                  # level is a ~14 uint8 step clear of the background
                  # cluster — which a playfield drawn in TWO BACKGROUND
                  # SHADES clears by construction. Measured on 10x12
                  # boards against a remembered background, every one of
                  # them at 0.31, twice the gate: alternating column
                  # shading over a deep stack, a top-out danger tint
                  # over the top four rows, a half-dimmed board behind a
                  # menu, shading drawn in full rows. Three bounds, each
                  # catching a shading the other two do not — a quarter
                  # of the playfield (the shadings that reach the
                  # floor), _PIECE_CELLS cells over the void (the ones
                  # that float), and no row the promotion completes (the
                  # ones drawn in full rows over a stack). Costed on the
                  # real pixels: over the five committed windows (767
                  # readings) the band never exceeds 8 cells of ~118
                  # observable and the only promotion in any of them is
                  # pale_piece's 6, four times clear of the budget.
                  # The budget is the WHOLE of the defense on the
                  # band-only branch (a pale piece on an otherwise clear
                  # board), where there is no threshold to have dropped
                  # anything and no gap to measure: the band IS the
                  # reading, at a flat 0.5, so a translucent pause or
                  # menu panel over a near-empty board reported board
                  # content at three times the gate.
                  # TWO MEASUREMENTS KEEP THE FLIP HONEST. A candidate is
                  # promoted only where the threshold actually DROPPED it,
                  # so every frame Otsu had already put with the pieces
                  # reads byte-identically — roas_stacker's settled
                  # periwinkle and the whole of absorbed_piece included.
                  # And a promotion's confidence is clamped by the air
                  # ABOVE the band as well as below it, because a level
                  # has air on both sides and one slice of a lighting ramp
                  # does not: measured, 0.469 over the pale piece against
                  # 0.074 over a paper-white board under a 140-unit
                  # vertical gradient, which puts that frame back under
                  # the gate where it belongs. No new threshold: the
                  # weakest boundary the three-level reading rests on IS
                  # the frame's confidence. That air is looked for on the
                  # BOARD and not on the frame — a named layer's cells
                  # were just declared not to be there, so they cannot be
                  # the next level up. On a black ground, where this
                  # tool's own hint composites to 0.374, counting them
                  # collapsed a correct pale-piece reading from 0.433 to
                  # 0.035 and lost the piece at the gate instead of at
                  # the split.
                  # The band is conditioned on standing clear of the
                  # background cluster by MIN_SPREAD/2 at BOTH ends of
                  # that reasoning, which is what refuses a CONTINUUM:
                  # the game's own start screen is text antialiased over
                  # white, climbing 0.174 to 0.196 with no gap anywhere,
                  # a clearance of 0.022. Nothing there is a layer and
                  # nothing there is content.
                  # The uniform-near branch yields to the same test. With
                  # no cell reaching MIN_SPREAD that branch calls the
                  # frame an empty board at 0.5, which is right for a
                  # board wipe and wrong for a pale piece spawning onto a
                  # clear board — the frame a coach is most needed on. A
                  # band standing clear of the background is what tells
                  # those apart; there is no two-class gap to measure
                  # there, so such a frame reports the uniform-empty
                  # number for the uniform-empty reason.
                  # TWO-LEVEL THEMES BEHAVE EXACTLY AS BEFORE, and that
                  # is a measurement rather than an assumption: no cell
                  # of any theme in the synthetic matrix reaches the band
                  # at all (0 band cells over 840 readings — 7 styles x 3
                  # cell sizes x 40 random boards — every one of which
                  # also reads exactly or falls under the gate), so the
                  # whole apparatus is unreachable on an ordinary skin.
                  # Measured after: ghost_session 0 UNEXPLAINED, 0 resets,
                  # 93 of 95 frames hinted (8 of 95 before), longest gap 2
                  # (the bootstrap); gate rejections across both committed
                  # sessions 33 of 191 -> 18 of 191, all of it
                  # live_session's own 32 -> 17. The gate is NOT what was
                  # wrong with the ghost session: 1 of its 95 frames is
                  # rejected before and after, the same frame 150 at 0.096
                  # (the badge, refused on purpose). On that window the
                  # layer cost hints, not frames. Those numbers are
                  # UNCHANGED by _own_paint_layer taking the naming over:
                  # it names the same cells on the same frames of both
                  # windows, so the two replays read frame for frame as
                  # they did. What it adds is the case the structural
                  # rule could not reach — a hint landing BESIDE the
                  # stack (tests/fixtures/ghost_beside_stack, where
                  # UNEXPLAINED goes 22 of 48 -> 4, resets 3 -> 1,
                  # phantom locks 2 -> 0, frames hinted 18 -> 31, and the
                  # hint targets for the one I piece 2 -> 1) and the
                  # phantom lock on absorbed_piece 296-297 (LOCKED 4 -> 2,
                  # FALLING 53 -> 55, the hint no longer jumping off the
                  # piece the player is holding).
                  # MEASURED FOR THE BAND FLIP, every committed window
                  # replayed through CoachEngine, before -> after
                  # (frames hinted / LOCKED / BOARD_RESET / rejected at
                  # the gate / piece-episodes whose hint changes target
                  # mid-flight):
                  #   pale_piece          0->45 / 0->0 / 1->1 / 14->14 / 0->0
                  #   ghost_beside_stack  31 / 0 / 1 / 5 / 0    (identical)
                  #   ghost_session       93 / 3 / 0 / 1 / 1    (frame 150
                  #     alone moves, 0.096 -> 0.0, rejected either way)
                  #   live_session        75 / 4 / 0 / 17 / 1   (frames
                  #     121-135 alone move, 0.078 -> 0.0, likewise)
                  #   absorbed_piece      59->66 / 1->1 / 3->1 / 0->5 / 0->0
                  # pale_piece's T is seen on all 47 accepted frames,
                  # tracked as one falling T from 00656 and hinted to the
                  # end of the window on one target; its floor row reads
                  # "#....#..##" rather than "#........#". The three
                  # windows in the middle are frame-for-frame what they
                  # were, apart from badge frames now refused on purpose
                  # instead of by four hundredths of an accident.
                  # absorbed_piece IMPROVES, and by that same rule: its
                  # four badge frames read 0.372 — above the gate — and
                  # went UNEXPLAINED, which tripped a BOARD_RESET whose
                  # resync adopted the widget as stack rows 8-9 and lost
                  # the T for seven frames. Refusing those frames holds
                  # the piece instead (UNEXPLAINED 14 -> 4).
  pieces_vision.py# explain_grid: diff the observed board against the tracker's
                  # committed stack memory and classify the frame (QUIET, FALLING,
                  # LOCKED, OCCLUDED, UNEXPLAINED). The falling piece is the
                  # added-cell diff matched against tetromino shapes — never
                  # guessed from a single frame — so lock delay, floor contact,
                  # and post-clear debris cannot confuse it. Locks are verified
                  # structurally: revealed by the next spawn, or a line-clear
                  # placement (tiered: last observed position, gravity drop from
                  # it, any clearing gravity drop) reproducing the observation
                  # exactly.
                  # Settled cells cannot vanish, so a frame saying they did
                  # is either a new world or a WRONG MEMORY. _carried_piece
                  # is the second hypothesis: when at most a tetromino's
                  # worth of committed stack goes missing, those cells are
                  # part of one tetromino the stack holds, that piece is
                  # airborne without it, and the frame then reads as an
                  # ordinary FALLING frame of the same piece, the stack was
                  # carrying a piece in flight and it is taken back out.
                  # Ordered AFTER the clear rules so a real clear reads as
                  # a clear. It is the self-healing half of the
                  # absorbed-piece fix and works however the piece got in,
                  # which is what lets the resync above stop at row 0.
                  # The piece that vouches for the removal must be where
                  # the cells could have MOVED to — at or below the rows
                  # they vacated, columns still touching — because the name
                  # alone vouches for nothing: any same-named tetromino
                  # anywhere on the board answered for four deletions, so a
                  # real post-clear island was vouched for by a piece eight
                  # rows up and eight columns across, one dropped cell of a
                  # settled I authorised deleting all four (only the
                  # VANISHED cells need be part of the candidate, which the
                  # real absorbed frames depend on), and a line clear's
                  # fade frames — which the clear rules cannot explain, so
                  # ordering does not reach them — deleted the clearing
                  # row's own cells and handed the solver a phantom gap.
                  # Touching columns, not overlapping: measured over 364
                  # consecutive same-piece observations on the session
                  # replays, one tick moves a piece at most two columns,
                  # and a vertical I stepping one or an O stepping two
                  # leaves spans that merely touch.
                  # unknown_rows marks cells the capture cannot observe (a game
                  # panel over the playfield). They are evidence for NOTHING —
                  # never an added cell, never a missing one — and free to stand
                  # in for a hypothesis' hidden cells: 1-3 added cells that are
                  # the visible part of a tetromino are a partly hidden piece,
                  # named when exactly one placement completes them and OCCLUDED
                  # (coherent, no candidate) when several do; a lock reveal
                  # matches on the locked piece's visible cells. Without
                  # unknown_rows every rule is the fully-observed one.
                  # The TOP EDGE of the board region is an unobservable
                  # region no mask can name: pieces enter the playfield
                  # from above row 0, so the frame a piece appears in shows
                  # only its bottom 1-3 cells, and in a game with no
                  # gravity (drag to move, drag to drop) that fragment SITS
                  # there for seconds. It is read the same way as a panel's
                  # edge: a 1-3 cell added fragment that TOUCHES row 0 and
                  # is AIRBORNE is a piece entering from above, named when
                  # exactly one tetromino completes it above the board and
                  # OCCLUDED when several do (measured over the shape
                  # space: 3 of 28 clipped rotations are nameable, the rest
                  # hold — two cells side by side fit O, S, Z, J and L, and
                  # a guessed name is a guessed hint). Its cells are NEVER
                  # merged into the stack; only a verified lock's are, and
                  # a lock revealed by a clipped spawn (4 + 1..3 added
                  # cells, the hard-drop-plus-spawn frame) commits the lock
                  # alone. Both guards carry their own failure: without the
                  # row-0 touch a stray blob mid-board would be read as a
                  # piece, and without the airborne test a column stacked
                  # to the top (legal, and load-bearing) would be deleted
                  # as one. A panel completion and a clipped one are
                  # hypotheses about the SAME cells, so they are ONE
                  # candidate set, counted together, on every path that
                  # reads a fragment (the frame itself, a lock reveal's
                  # spawn, a clearing lock's residual). Asking the panel
                  # first and answering from it alone is a wrong answer
                  # dressed as a unique completion: in this session's own
                  # geometry the fragment (0,5),(0,6),(0,7) has exactly one
                  # panel completion (a flat I hiding its fourth cell under
                  # the panel) and three clipped ones (T, J, L entering from
                  # above), and the panel's I came with a position too
                  # (measured: 6 such fragments at cols 5-7, each of which
                  # named a piece that is not there and then went
                  # UNEXPLAINED for 4 frames -> BOARD_RESET when the real
                  # piece dropped). A clipped piece's bounding box starts
                  # off-grid (row < 0), which the two position-anchored
                  # lock rules skip rather than index the board with.
                  # Measured on 96 consecutive frames of the failing
                  # session (tests/fixtures/live_session): before, 34
                  # frames UNEXPLAINED, 7 spurious BOARD_RESETs, 0 verified
                  # locks, and the reset absorbed the fragment as phantom
                  # row-0 stack cells — the board handed to the solver had
                  # blocks in it that do not exist, which is why the first
                  # hint was right and every later one was random. After: 0
                  # unexplained, 0 resets, 4 locks, and a committed stack
                  # equal to the board in the image.
                  # That tie is broken by EVIDENCE, where there is any,
                  # rather than by a guess: a game deals the previewed
                  # piece and shows the one after it, so a preview
                  # changing X -> Y says X is the piece now entering
                  # (entering_hint, supplied by the tracker, which
                  # watches the preview — only a change of a KNOWN preview
                  # counts, since None -> X says nothing about what was
                  # dealt). The hint only SELECTS among completions the
                  # structural rule already accepts, and only when exactly
                  # one of them carries that name: it can never create,
                  # suppress or relocate an explanation, a name no
                  # completion carries changes nothing, and a name that
                  # fits several placements of the same piece still holds.
                  # A mis-read preview can therefore misname an entering
                  # piece — the deliberate trade against no name at all for
                  # as long as it sits at the top edge (measured on the live
                  # session: 13 frames, ~0.9 s) — and the ordinary rules
                  # rename it the moment it descends into view. That trade
                  # holds only while the cost of a wrong name IS a wrong
                  # name, so a hinted name is marked as one and the tracker
                  # never keeps it as the last observation: explain_grid
                  # refuses a lock whose piece disagrees with the last
                  # OBSERVED name, so a misnamed entering piece would block
                  # its own lock (4 identical unexplainable frames -> a
                  # spurious BOARD_RESET, hint cleared, overlay blank),
                  # making a wrong name strictly worse than no name.
                  # A preview reading is a HYPOTHESIS, so the frame that
                  # FALSIFIES one ends it: when the fragment is a piece
                  # entering from above and no placement of the hinted
                  # piece fits it, explain_grid reports hint_refuted and
                  # the tracker drops the hypothesis on that frame
                  # (Explanation.hint_refuted -> GameStateTracker.update),
                  # and TAKES BACK the name that hypothesis had already
                  # committed (PIECE_UNNAMED; see state.py rule (4)).
                  # Two cells side by side fit an O as well as an L; the
                  # row below them does not, so a misnamed piece whose
                  # fragment GROWS before it is whole contradicts itself
                  # at the top edge, and one that goes straight from two
                  # cells to four is renamed by the ordinary rules on the
                  # frame it is whole. What the correction costs is the
                  # frames BEFORE it — while the fragment fits the hinted
                  # name as well as the real one, the wrong name is on
                  # screen, measured at 12 frames (0.80 s) on the
                  # committed window with a consistently lying box. What
                  # it never costs is anything structural: a hinted name
                  # authorises no lock (it is not an observation) and an
                  # unnameable frame is OCCLUDED, which never counts
                  # toward a reset.
                  # Only a fragment coming in from ABOVE is evidence about
                  # the hint — a piece sliding under a panel was named
                  # from its own earlier frames — and a piece seen WHOLE
                  # deliberately is not: on the frame of a flip the board
                  # still shows the piece that just locked, which would
                  # refute every hint on the frame it was set. It only
                  # became possible once the preview could be read at all
                  # (see identify_next); the panel case is deliberately left
                  # alone, because a piece that slides under a panel was
                  # named from its own earlier frames, while a piece
                  # entering the field has never been seen whole.
                  # Next-piece region: threshold, keep the BLOCK-LIKE
                  # connected components, derive the cell grid from those
                  # blocks, match the shape. Color is a hint, not required.
                  # None of it assumes the crop is tight, centered or
                  # square, because a real preview box is none of those: the
                  # captured one (tests/fixtures/live_session) is 94x94
                  # holding a 19px-celled piece over about a fifth of it,
                  # off center, under a faint grey "NEXT" caption. The
                  # caption is the killer — it scores as far from the white
                  # ground as the pieces do (measured: caption 0.57, I blue
                  # 0.75, O yellow-green 0.53), so NO threshold separates
                  # them, and inside the raw bounding box (45x79 for a
                  # 19x79 piece) every rotation either failed its aspect
                  # test (a horizontal I at 0.44) or resampled to a
                  # non-tetromino: None on 96 of 96 frames of the session,
                  # i.e. no lookahead at all and every hint 1-ply. So the
                  # furniture is dropped on SHAPE: a block nearly fills its
                  # own bounding box (>= 0.5; the merged T/S/Z/J/L box is
                  # 4/6, a glyph stroke or a hollow border far less) and is
                  # within 8x of the largest block (a cell against four
                  # merged is 4x; the live caption's fragments are 1-9 px
                  # against 357-361 px cells). The grid then comes from the
                  # blocks, per axis: every row and column of a tetromino's
                  # bounding box holds a cell, so N bands mean N cells
                  # (centers and cell size read off them, which is what
                  # makes an inset skin readable), one band means cells
                  # drawn flush (an even division is then exact), and
                  # anything else contradicts the hypothesis and is refused
                  # rather than guessed. A cell read out of ONE band is
                  # then measured EDGE TO EDGE rather than at its center,
                  # since flush is exactly what that hypothesis claimed,
                  # and the central sample is blind to a cell's edges on
                  # purpose: measured over the style matrix, 1128 of 1169
                  # flush readings fill their cell rectangles >= 0.90 and
                  # 688 fill them exactly, while 407 round blobs (every
                  # radius and ratio, aliased and antialiased, noisy, with
                  # and without a digit) reach 0.847 at most. The disc is
                  # not hypothetical: it is this tool's own rotation
                  # badge, which lands in the box the same way the hint
                  # does and read as a confident 'O' at every radius on
                  # every theme — and being opaque rather than a
                  # composite, it is the one part of our overlay the color
                  # arithmetic below cannot catch, so it is refused on
                  # shape or not at all. The floor costs the other 41
                  # readings: small cells under a caption, where the
                  # caption's class lifts the threshold and the mask loses
                  # the seam between two cells (a vertical S of 14 px
                  # cells on a flush skin, which reads at 18 px and with
                  # no caption). No committed crop is read this way at
                  # all. Those bands ARE the cells, so they
                  # must look like cells: all the same size (within the same
                  # 25%; measured, they agree exactly on every style and
                  # every live crop) and separated by gaps smaller than one
                  # of them (a skin's inset or gridline; measured at most
                  # 0.67 of a cell, 0.05 live). Both refuse the furniture
                  # the SOLIDITY filter cannot: a caption drawn as a solid
                  # BAR rather than as glyph strokes is block-like by every
                  # test above, and one wide enough to bridge the gap
                  # between two cells merges them into a band three times
                  # its neighbours' width, which the median band size read
                  # as an ordinary cell (measured: a 40x9 bar over a
                  # horizontal I is a confident J, on 120 of 495 bar
                  # geometries; 0 of 495 now, and 43 of 143665 over the
                  # style matrix, all of them a bar that IS the picture of
                  # a horizontal I). Each cell is sampled at its CENTER
                  # (insets and gridlines sit at the edges), the derived
                  # cell must be square within 25% (measured: exactly 1.000
                  # on every synthetic style and every live crop), an
                  # occupied sample must read >= 0.75 and an empty one
                  # <= 0.45 (measured: 1.00 and <= 0.40), and two pieces
                  # matching at once says nothing. Those last gates are
                  # what refuse TEXT, which is the adversarial case for a
                  # shape rule: "NEXT" is four shapes in a row, exactly a
                  # horizontal I's arrangement, and it fails on fill (0.60)
                  # and on squareness. Returning None is a first-class
                  # answer here; a confident WRONG piece is worse, since it
                  # feeds a 2-ply hint planning around a piece the game
                  # never deals — and, since the preview now NAMES the
                  # half-visible piece entering the board, a wrong reading
                  # of the box becomes a wrong hint about a piece that is
                  # on screen.
                  # The threshold is anchored at MIN_SPREAD, so a piece
                  # drawn too PALE to reach it is not misread but absent:
                  # the mask keeps nothing, the box reads empty, and the
                  # coach has no lookahead for as long as that piece sits
                  # there (measured on a real session: None on 226 of 705
                  # frames, 32%, in runs up to 45 — the same failure the
                  # BOARD had until grid.py stopped letting a threshold
                  # decide the intermediate band). So when the thresholded
                  # reading names nothing, the band [MIN_SPREAD/2,
                  # MIN_SPREAD) is read on its own and handed to the SAME
                  # shape rules above. Three things make that safe in a
                  # box, which is not a playfield:
                  #  - the band is SPLIT first, by the same Otsu the frame
                  #    is: a crop carries furniture a board cell's patch
                  #    mean never sees — the box's hairline border and
                  #    gridlines, measured at 0.179-0.250 against the pale
                  #    piece at 0.254-0.349, and touching it — and taken
                  #    whole the band merges the piece's cells through
                  #    that border (0 of 76 pale crops readable; 60 with
                  #    the split). A band with nothing to split (one flat
                  #    level) is taken whole.
                  #  - the band's CEILING is what keeps the caption out,
                  #    rather than a rule about captions: a caption's core
                  #    is solid class (0.586 on the live crops), so only
                  #    its antialiased skirt is in the band, and a hollow
                  #    outline is not block-like. Admitted instead as a
                  #    lowered threshold it composes into one solid blob
                  #    that survives the block filter and drags the
                  #    bounding box off the piece.
                  #  - our OWN hint fill is refused outright. The box
                  #    floats over the top corner of the playfield, so a
                  #    hint drawn there is drawn over the box — a
                  #    tetromino of square cells, in the place a tetromino
                  #    is expected, which no shape rule can tell from the
                  #    piece the game dealt. Over a light box it
                  #    composites INTO this band (0.321, four thousandths
                  #    from the session's pale piece), so it is named the
                  #    way grid.py names it: by the color arithmetic of
                  #    OwnPaint, whose tolerance the nearest real band
                  #    pixel stands 2.6x clear of. A match refuses the
                  #    box (None), the answer grid.py gives its own
                  #    unnameable paint. The composite is a DISTANCE from
                  #    the box's ground, not a constant, so the rule is
                  #    asked in both readings of the box and over the
                  #    box's own LEVELS rather than its average: over a
                  #    BLACK box the same fill scores 0.374 and arrives
                  #    as a solid class the threshold used to name
                  #    ('O'/'T'/'I', the placement on screen), and a box
                  #    drawn in TWO shades — a panel around an inner
                  #    well, the ordinary skin — puts the median on one
                  #    while the paint lands on the other, 21 units apart
                  #    against a tolerance of 8. So the grounds are the
                  #    median plus the median of each Otsu class outside
                  #    the candidate; over every committed crop the
                  #    nearest real pixel to any of those composites
                  #    still stands 2.4x the tolerance clear.
                  #  - the FLUSH hypothesis is OFF here. One band divided
                  #    into cells by assertion rather than by anything
                  #    visible makes every solid rectangle a piece (a
                  #    square is an O, a 4:1 bar an I), which the
                  #    threshold can afford because its solid class is
                  #    nearly always the piece, and the band cannot
                  #    because the band is where everything the threshold
                  #    REFUSED arrives. Measured with it on: the box's own
                  #    inner WELL reads 'O' at every panel/well shade pair
                  #    tried (0.223-0.307), with no piece in the box at
                  #    all; a pale caption BAR above the piece — which the
                  #    band's upper class keeps INSTEAD of the piece,
                  #    since nothing says the furniture sits below it —
                  #    reads a confident 'I' on 150 of 396 bar geometries
                  #    while the box holds some other piece; a lone
                  #    rectangle is named on 162 of 540 geometries. It
                  #    costs a flush skin its PALE O and I, and it costs
                  #    the evidence nothing: of the 530 crops named across
                  #    every committed window, not one is named that way.
                  # The pass is asked only where the threshold came back
                  # empty-handed, so every crop that was readable before is
                  # byte-identical. Measured over the committed windows:
                  # 530 of 557 crops named against 452, no crop's name
                  # changed, and on tests/fixtures/spawn_latency the
                  # entering-piece accelerator gains an episode it had no
                  # evidence for (16 frames to hint -> 2).
                  # WHICH WAY IT ERRS is the same way the rest of this
                  # module does, and the 27 crops still unread are the
                  # evidence: a box mid-deal with the outgoing piece's
                  # panel sliding over the incoming one, and a box under
                  # the game's end-of-round summary. Neither holds a
                  # piece, and a band a rule cannot resolve into exactly
                  # one tetromino keeps the silence the threshold gave
                  # it. That direction is not a preference here: a
                  # preview feeds the 2-ply lookahead AND names the
                  # half-visible piece entering the board, so a
                  # confident wrong box is a confident wrong hint about
                  # a piece the user is watching.
                  # MEASURED END TO END, every committed window replayed
                  # through CoachEngine, before -> after (frames hinted /
                  # PIECE_LOCKED / BOARD_RESET / rejected at the gate /
                  # frames hinting a piece other than the committed
                  # falling one):
                  #   absorbed_piece      66 / 1 / 1 /  5 / 0  (identical)
                  #   ghost_beside_stack  31 / 0 / 1 /  5 / 0  (identical)
                  #   ghost_session       93 / 3 / 0 /  1 / 0  (identical)
                  #   live_session        75 / 4 / 0 / 17 / 0  (identical)
                  #   pale_piece          45 / 0 / 1 / 14 / 0  (identical)
                  #   spawn_latency      120->134 / 2 / 2 / 37 / 0
                  # Only the window whose box the band pass opened moves,
                  # and only by carrying a hint on 14 more frames
                  # (OCCLUDED 31 -> 17). Sighting -> hint distances are
                  # unchanged elsewhere (absorbed_piece 15, ghost_session
                  # 2, live_session 2), and the extra pass costs ~1 ms,
                  # on the frames the threshold could not read and
                  # nowhere else — app.py reads the box only when its
                  # pixels change.
  state.py        # GameState tracker: keeps the committed stack as the one
                  # authoritative memory (never None), feeds explain_grid, and
                  # debounces (2 consistent frames) before committing. UNEXPLAINED
                  # frames touch nothing (line-clear animations, torn frames);
                  # PIECE_LOCKED fires when a lock is structurally verified, not at
                  # touchdown; BOARD_RESET only after several consecutive identical
                  # unexplainable frames (new game, garbage, mid-game attach);
                  # PIECE_UNNAMED when a name the preview hint supplied is
                  # ruled out by a later frame — the name is withdrawn and
                  # the consumer takes the hint off the screen (rule (4)).
                  # A resync adopts the observed board as the stack MINUS
                  # the piece in flight (strip_piece_in_flight): the one
                  # component that rests on nothing AND touches row 0,
                  # budgeted at a tetromino (and at four cells it must BE
                  # one). It is the one path that commits a board it
                  # cannot diff, and a mid-game attach in a game that
                  # parks spawns at the top edge is the ordinary case. A
                  # whole piece is held back too, not just a clipped one:
                  # freezing the falling piece in is what made its own
                  # descent read as stack cells vanishing, so the tracker
                  # reset and re-absorbed it one row lower, the whole way
                  # down (58 frames, ~3.9 s with no hint;
                  # tests/fixtures/absorbed_piece).
                  # Row 0 is where the rule STOPS, because floating does
                  # not imply in flight: naive gravity makes settled cells
                  # float, since clearing a row drops everything above it
                  # onto a row with holes in it, and the repo's own
                  # Board.drop turns an ordinary position into a board
                  # with four real locked cells hanging at rows 9-10.
                  # Deleting those hands the solver a phantom hole and
                  # then announces a piece the player does not have, as
                  # the same cells read back as added (measured over
                  # solver self-play: 666 of 47874 settled boards, 2.16%
                  # of post-clear ones, 1014 cells). Settled content
                  # cannot be at row 0 with air under it — the same
                  # picture grid.py vouches for as one piece in flight —
                  # and anything lower is adopted, floating or not. A
                  # piece absorbed lower down is not stranded: its next
                  # descending frame takes it back out (_carried_piece),
                  # which is the loop-breaker that works however the piece
                  # got in. "Maximal component" also holds only on the
                  # board the capture can SEE, so a component adjacent to
                  # an unobservable cell is never held back: the panel
                  # over cols 8-9 of rows 0-1 splits a column stacked to
                  # the top from a piece locked beside it, and the inner
                  # half then floats.
                  # The tracker also remembers the preview: the last known
                  # reading and the one it replaced. The piece that LEAVES
                  # the preview is the piece entering the board, which is the
                  # only evidence there is for naming a fragment the top edge
                  # has cut in half (explain_grid's entering_hint above).
                  # Measured on the live session: the O is named on frame 61
                  # instead of 74 — 13 frames, ~0.9 s, that used to carry no
                  # hint at all — and frame 74's independent structural
                  # reading agrees it is an O. That evidence is about ONE
                  # deal, and three rules hold it there — a flip says "the
                  # piece that was here has been dealt" and never says
                  # WHEN, so all three are about DATING it.
                  # (0) A reading is the box's CONTENT only once a second
                  # consecutive readable capture agrees with it — the same
                  # debounce every other observation here goes through.
                  # The preview is read by the same vision as the board and
                  # one frame of it can be wrong, and one misread capture
                  # is not one bad flip but TWO: X -> W and then W -> X.
                  # The second is the dangerous one — it names W, a piece
                  # the game never dealt, and applies it to whatever
                  # fragment is parked at the top edge, which the
                  # structural rules had refused to name. (SPEC used to
                  # claim rule (2) covered this flicker; it only bounds how
                  # long W survives past the NEXT lock, and does nothing
                  # about the piece already in flight, which is renamed on
                  # the spot.) The cost is one capture of latency on a real
                  # flip, and the hint's age carries it honestly: a flip is
                  # dated from the capture the change was FIRST seen on,
                  # not from the one that confirms it.
                  # NOTE what none of these rules can do: a preview that
                  # reads WRONG consistently names the clipped fragment
                  # wrong, and no frame can contradict it while two cells
                  # at row 0 fit five pieces. Measured with every O in the
                  # committed window misread as an S: 12 frames, 0.80 s,
                  # of a confidently wrong hint on the one piece the
                  # preview named, ending as the O's second row descends,
                  # with the same committed stack, locks and resets as the
                  # honest replay. The claim to make is not "never a wrong
                  # hint" — it is that a wrong hint is bounded by the
                  # ambiguity of the frame, retracted the moment the frame
                  # can tell, expired with its deal, and never evidence.
                  # (1) A flip counts only when the GAP it is read across
                  # is shorter than a tenure. The box is unreadable in
                  # bursts, and a burst covering one previewed piece's
                  # whole tenure moves its value on twice, so the next flip
                  # arrives a deal late (X -> [Y never read] -> Z names X,
                  # which is by then on the board — the "confused two
                  # pieces" failure the user reported seeing once). A piece
                  # that came and went inside the gap held the box for the
                  # whole of it, so a gap shorter than one tenure cannot
                  # hide a deal; MAX_PREVIEW_GAP (12 captures) is that
                  # budget, set between the shortest tenure in evidence
                  # (18 captures, the pale T on spawn_latency 00198-00215,
                  # whose flip at 00216 is refused) and the longest gap the
                  # win depends on (6 captures, the end-of-round wipe at
                  # 00152-00157, whose flip at 00158 names the O it dealt).
                  # The gap is counted in CAPTURES, not in frames the
                  # board's confidence gate accepted: app.py returns before
                  # tracker.update() on every rejection, so a clock running
                  # on accepted frames alone stops for exactly the events
                  # that blank the box (a wipe, a flash, an animation —
                  # they reject the board too), and the flip on the far
                  # side of one is dated against whatever was last
                  # accepted, seconds earlier. GameStateTracker.
                  # observe_preview takes the rejected captures for that
                  # reason, and the hint's age (rule (2)) counts them too.
                  # Measured: before it, the headline flip at 00158 was
                  # dated against 00126, 32 captures and one deal earlier.
                  # (2) A hint survives a lock only while it is YOUNGER
                  # than that lock's own debounce. The preview reports a
                  # deal by changing as the new piece spawns, which is the
                  # frame the commit is debouncing, so the hint of the deal
                  # a lock BEGINS is always the young one; an older hint
                  # names the piece that just LOCKED. Counting locks
                  # instead (what this did) cannot tell the two apart —
                  # both show exactly one lock since the hint was set — and
                  # with no expiry at all, one unreadable burst (or one
                  # flickered frame: X -> W -> X leaves the hint naming W,
                  # a piece never dealt) names every later top-edge
                  # fragment for the rest of the session.
                  # (3) A resync drops a hint from BEFORE the world it is
                  # adopting — what the preview shows still holds, what was
                  # dealt into THIS board does not — but KEEPS one set
                  # inside the run of frames it is resyncing onto, which is
                  # this board's own deal. That last step assumes the new
                  # world continues the same piece SEQUENCE: true of a
                  # field wipe, of garbage and of a mid-game attach, false
                  # of a new game or a restart, where the departing name
                  # belongs to the old game and the first piece dealt has
                  # nothing to do with it. No frame separates the two, so
                  # the clause is bounded rather than safe: the kept hint
                  # is a hypothesis, rule (4) takes the name back on the
                  # first frame that contradicts it, rule (5) takes it
                  # back on the clock when the new game's first piece
                  # PARKS and no frame ever contradicts anything, and it
                  # decides nothing structural. That clause is the wait the user
                  # reported: the game wipes the field, deals an O, the box
                  # flips O -> I on the frame the O's first two cells reach
                  # the top edge, and the reset four frames later threw
                  # away the hint naming the very piece it was resyncing
                  # ONTO. (4) And a frame that REFUTES a hint both drops
                  # the hypothesis (hint_refuted, above) and RETRACTS the
                  # name that hypothesis already committed: the frame that
                  # contradicts a hint normally names no replacement (the
                  # piece has shown a second cell, which rules the hinted
                  # name out while still fitting several others), so it is
                  # OCCLUDED, and OCCLUDED holds the committed snapshot —
                  # the refuted name stayed on screen for the rest of the
                  # piece's tenure at the top edge, 14-17 frames in this
                  # game. The tracker therefore keeps which committed name
                  # came from a hint, tests it against every later frame's
                  # entering_names, and on a contradiction commits the
                  # falling piece back to None and fires PIECE_UNNAMED. The
                  # retraction is NOT debounced: the debounce exists to
                  # stop a torn frame committing something, and withdrawing
                  # a claim is the safe direction. A name structure has
                  # produced — or confirmed — stops being a hypothesis and
                  # is never retracted.
                  # (5) And a hint expires with the deal it reports
                  # whether or not anything REVEALS that deal. Rules (2),
                  # (3) and (4) all wait for an event — a lock, a resync,
                  # a frame that contradicts the name — and the cases
                  # nothing else can see are exactly the ones where no
                  # event comes: a hold swap (below), a restart whose
                  # first piece the old game's preview named (rule (3)),
                  # or simply a piece PARKED at the top edge showing two
                  # cells that fit the name whatever it really is. Without
                  # a clock of its own the hint had no expiry at all in
                  # those cases: measured, 300 frames — twenty seconds —
                  # of a swapped-in piece wearing the previous piece's
                  # name, and it would have held for the rest of the
                  # session. MAX_HINT_AGE (24 captures) is that budget,
                  # and like every other one here it is set between two
                  # measurements: above the longest a hint has
                  # legitimately held a name on screen (15 captures, the O
                  # on spawn_latency 00159-00173, the shape rule taking
                  # over on the 16th) and above the longest tenure in the
                  # evidence (18), so no hint is ever cut short of the
                  # window it exists to cover; and low enough that a name
                  # nothing can contradict stands for 1.6 s instead of
                  # forever. Expiring it WITHDRAWS the name too, by rule
                  # (4)'s machinery and for rule (4)'s reason: the frame
                  # under it is OCCLUDED, which holds the committed
                  # snapshot, so dropping the hypothesis alone would leave
                  # the name exactly where it was. It is a ceiling, not a
                  # detector — where to stop believing a hint nothing has
                  # confirmed, not how to notice the swap.
                  # What none of them can SEE is a HOLD swap: the held
                  # piece comes out of the hold box, so the piece at the
                  # top edge changes with no lock to expire the hint and
                  # no preview flip to re-date it, and the swapped-in
                  # piece wears the name the hint gave the one it
                  # replaced. Nothing in the capture contradicts that
                  # while its visible cells still fit the name (two cells
                  # side by side fit an O whether or not they are one),
                  # and position rules nothing out either — the game this
                  # was measured on drags pieces rather than stepping
                  # them. The hold box is not captured at all, only the
                  # board and the preview, so there is no signal to read;
                  # reading one would mean capturing a third region, which
                  # is a scope decision and not a patch. So the swap is
                  # still not DETECTED, and what it costs is now bounded
                  # at both ends: rule (4) takes the name back on the
                  # first frame that contradicts it, rule (5) takes it
                  # back after 1.6 s when no frame ever does, and nothing
                  # structural was decided on it either way. Pinned in
                  # test_a_hold_swap_wears_the_hint_until_the_new_piece_
                  # contradicts_it and test_a_hint_nothing_can_contradict_
                  # expires_rather_than_stand, as a bounded residual and
                  # not a solved case.
                  # Measured over tests/fixtures/spawn_latency (161 frames,
                  # 124 accepted): OCCLUDED 43 -> 31, frames with a hint on
                  # screen 108 -> 120, and the per-piece sighting -> hint
                  # distance 17 / 14 / 16 -> 17 / 2 / 16 frames. The 2 is
                  # the commit debounce, not a wait, and the O is named
                  # with two of its four cells still above the capture. The
                  # two that do not move are the two with no sound evidence
                  # (one where no flip has been seen at all, one where the
                  # flip is a deal late), which is the shape of the whole
                  # thing: an accelerator on a signal that is usually
                  # present (the preview reads on 69% of this session's
                  # frames), never a dependency — where it is absent the
                  # coach holds exactly as before. Every other committed
                  # window is unmoved: pale_piece, ghost_beside_stack,
                  # ghost_session, absorbed_piece, live_session and
                  # roas_stacker keep the same frames hinted, the same
                  # LOCKED and BOARD_RESET counts, and no episode whose
                  # hint changes target mid-flight.
                  # unobservable_cells: the tracker discards the capture's
                  # reading there and carries a BELIEF for those cells instead —
                  # seeded empty at bootstrap/resync, moved only by an explained
                  # transition (a lock merges in what it can see). An OCCLUDED
                  # frame holds state and never counts toward a reset, so a piece
                  # resting under the panel cannot wipe the board.
capture/
  screen.py       # mss-based capture of a screen rect at native (Retina) scale;
                  # handles logical-vs-pixel coordinate scaling. Protocol/interface
                  # so tests can inject frames from PNG files instead.
                  # A screen shot of the board region contains whatever is
                  # ON TOP of it, and this tool's own overlay window is —
                  # every committed fixture has the coach's hint in it,
                  # and reading it back as board content is what
                  # vision.grid._own_paint_layer exists to undo. Excluding
                  # the overlay's own window from the shot is the real
                  # fix and is platform-specific work that belongs here;
                  # until it lands the classifier carries it.
overlay/
  window.py       # PySide6 frameless, transparent, always-on-top, click-through
                  # window (WA_TransparentForMouseEvents, WindowStaysOnTopHint,
                  # WindowTransparentForInput, NoDropShadowWindowHint) positioned
                  # exactly over the board region.
  renderer.py     # Draw target placement: 4 cell outlines + subtle fill; distinct
                  # color (configurable); optional arrow/rotation count badge.
                  # What is painted here comes back round in the next
                  # capture, so HintStyle.fill_opacity is IMPORTED from
                  # vision.grid.HINT_FILL_OPACITY rather than written
                  # twice: the painter and the classifier that has to
                  # recognize the composite again cannot be allowed to
                  # drift. (That direction, overlay -> vision, and not the
                  # other, because vision/ must stay importable with no
                  # display and overlay/ needs PySide6.)
region_select.py  # Full-screen dim + drag-rectangle picker (Qt), returns rect in
                  # logical coords; run twice (board, next box).
app.py            # Main loop wiring: capture -> vision -> state -> solve -> overlay.
                  # Precompute: while piece A falls, assume it lands on target and
                  # pre-solve piece B; on lock, flip hint instantly; if observed
                  # board != predicted, re-solve from observed.
                  # The loop is a FEEDBACK loop, not a pipeline: the
                  # overlay it draws is on screen when the next capture is
                  # taken. CoachConfig.hint_color therefore goes to the
                  # classifier as well as to the renderer, so the coach
                  # looks for the paint it actually drew (see
                  # vision.grid._own_paint_layer).
                  # compute_overlap_mask: when the next-piece region overlaps
                  # the board region (a preview floated over the top corner),
                  # project next_rect into the board's unit square (board-
                  # relative fractions, so Retina-agnostic) and name the cells
                  # whose center falls inside. Those cells are UNOBSERVABLE,
                  # not empty: the engine drops their reading (after confidence
                  # is judged on the grid, before the tracker and debug view)
                  # and declares them to BOTH stateful components — the
                  # tracker, which treats them as unknown, and the classifier,
                  # whose background estimate and top-row prior must not be
                  # fed a UI panel's pixels. Computed once from the fixed
                  # session rects; empty (a no-op) when the next box is drawn
                  # outside the board.
                  # _solver_board: what the solver is handed for those cells. A
                  # covered cell resting directly on the stack (or the floor) is
                  # where the stack plausibly continues up out of sight, so it
                  # goes across FILLED and the hint stays out of space the coach
                  # cannot see; covered cells with air under them go across as
                  # believed. Filling them all unconditionally is wrong: a
                  # column whose top row is filled is one Board.drop rejects, so
                  # it would cost the columns under the panel in every board
                  # state to guard a case that only arises near top-out.
                  # Tradeoff: a real piece crossing the covered corner cannot be
                  # named while several tetrominoes fit its visible part (the
                  # frame is OCCLUDED: hint held, nothing committed, no reset),
                  # and a lock there commits only the cells actually seen.
cli.py            # `tetris-coach` entry point: select regions, start loop; flags
                  # for poll rate, colors, and a terminal debug view (per
                  # committed frame: observed grid, falling piece, next piece,
                  # confidence, events). A *graphical* debug window is deferred
                  # to the live phase..
```

## Key design decisions (already made — do not relitigate)

1. **Bitboards** (one int per row) for the solver. Target: full 2-ply search
   < 50 ms in CPython. Benchmark in tests; if over budget, optimize before adding
   features. Numba is a fallback, not a default dependency.
2. **Dellacherie evaluation** (6 features, fixed classic weights) rather than
   height/holes/bumpiness-only. One-piece + next-piece search (2-ply), no deeper.
3. **Precompute-ahead loop** (app.py above) so perceived latency ≈ one repaint.
4. **User-selected regions** rather than auto-detection of the board. Auto-detect is
   a possible later enhancement, not v1.
5. **PySide6** for region picker + overlay. **mss** for capture. **OpenCV + NumPy**
   for vision. Python 3.11+.
6. Vision works on *occupancy*, not colors, for game-agnosticism. Colors only help
   piece identification as a secondary signal.

## What can be built & tested headless (Linux CI / cloud)

- All of `core/`, `solver/`: pure logic. Unit tests + a self-play simulation test
  (solver should survive ≥ 1000 pieces on random sequences without topping out —
  Dellacherie's algorithm averages far more) + a pytest-benchmark for search time.
- All of `vision/`: test against generated synthetic board screenshots (render
  boards with varied palettes/cell sizes/gridlines/backgrounds via Pillow) plus a
  few real screenshots committed under `tests/fixtures/` (small PNGs of e.g.
  Jstris/Tetr.io boards).
- `capture/screen.py` and `overlay/` are macOS-only at runtime: keep them thin,
  isolate behind interfaces, write them to spec, exclude from headless test runs
  (import-guard so the package imports cleanly without a display).

## Definition of done for the headless phase

- `pytest` green, including vision fixture tests and solver self-play.
- Benchmark report in `BENCH.md`: search time p50/p95 on M-class and CI hardware.
- `python -m tetris_coach.cli --demo` runs a terminal demo: simulated game where
  the solver plays itself, printing the board — proves the whole non-GUI stack.
- Code formatted (ruff), typed (mypy clean on core/solver at least).
