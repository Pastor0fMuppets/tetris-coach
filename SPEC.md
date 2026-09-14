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
                  # as one. Panel completions are tried FIRST, so a session
                  # with a NEXT panel reads exactly as it did before the
                  # rule existed. A clipped piece's bounding box starts
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
                  # rename it the moment it descends into view. It only
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
                  # rather than guessed. Each cell is sampled at its CENTER
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
                  # never deals.
  state.py        # GameState tracker: keeps the committed stack as the one
                  # authoritative memory (never None), feeds explain_grid, and
                  # debounces (2 consistent frames) before committing. UNEXPLAINED
                  # frames touch nothing (line-clear animations, torn frames);
                  # PIECE_LOCKED fires when a lock is structurally verified, not at
                  # touchdown; BOARD_RESET only after several consecutive identical
                  # unexplainable frames (new game, garbage, mid-game attach).
                  # A resync adopts the observed board as the stack MINUS
                  # the visible part of a piece the top edge cut in half
                  # (strip_entering_piece): a single component reaching row
                  # 0 that satisfies the entering-piece rule against the
                  # rest of the board. It is the one path that commits a
                  # board it cannot diff, and a mid-game attach in a game
                  # that parks spawns at the top edge is the ordinary case.
                  # A fully visible piece is still absorbed (nothing in one
                  # memoryless board says it is in flight).
                  # The tracker also remembers the preview: the last known
                  # reading and the one it replaced. The piece that LEAVES
                  # the preview is the piece entering the board, which is the
                  # only evidence there is for naming a fragment the top edge
                  # has cut in half (explain_grid's entering_hint above).
                  # Measured on the live session: the O is named on frame 61
                  # instead of 74 — 13 frames, ~0.9 s, that used to carry no
                  # hint at all — and frame 74's independent structural
                  # reading agrees it is an O.
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
overlay/
  window.py       # PySide6 frameless, transparent, always-on-top, click-through
                  # window (WA_TransparentForMouseEvents, WindowStaysOnTopHint,
                  # WindowTransparentForInput, NoDropShadowWindowHint) positioned
                  # exactly over the board region.
  renderer.py     # Draw target placement: 4 cell outlines + subtle fill; distinct
                  # color (configurable); optional arrow/rotation count badge.
region_select.py  # Full-screen dim + drag-rectangle picker (Qt), returns rect in
                  # logical coords; run twice (board, next box).
app.py            # Main loop wiring: capture -> vision -> state -> solve -> overlay.
                  # Precompute: while piece A falls, assume it lands on target and
                  # pre-solve piece B; on lock, flip hint instantly; if observed
                  # board != predicted, re-solve from observed.
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
