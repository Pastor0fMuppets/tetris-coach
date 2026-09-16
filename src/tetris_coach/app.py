"""Main loop wiring: capture -> vision -> state -> solve -> overlay.

Precompute-ahead: while piece A falls toward its target, assume it lands
there and pre-solve piece B on the predicted board; when A locks, the hint
for B flips instantly. If the observed board disagrees with the prediction,
re-solve from the observed board.

This module is headless-importable; GUI/capture objects are only created
inside :func:`run`.

Threading: in the live app the QTimer on the Qt GUI thread only *schedules*
ticks; the blocking capture->vision->solve pass (:class:`FrameWorker`, which
wraps the synchronous :class:`CoachEngine`) runs on a single worker thread,
and its result crosses back to the GUI thread through a queued signal so the
overlay always repaints there. Ticks never overlap: timer fires while the
worker is busy are skipped.

Hint stability: the overlay is a TRAINING aid, so what it shows has to be
followable. A target chosen for a piece is therefore HELD for as long as
that piece is in flight, and the engine only solves again when an input
actually changed — see :data:`HINT_SWITCH_MARGIN` and
:meth:`CoachEngine._steady_hint` for the rule and the direction it errs in.

Two trackers, one policy: WHAT is falling, over what board, is read by a
:class:`~tetris_coach.vision.readers.FrameVision` — the colour-first
tracker by default, the shipped shape-matching one behind
``--tracker shape`` — and everything in this module that decides what the
user SEES (the stability rule above, the stale-hint withdrawal, the board
handed to the solver, the precompute) is written against the one
:class:`~tetris_coach.vision.readers.FrameReading` both of them produce.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .capture.screen import FrameSource, Rect
from .core.board import DEFAULT_HEIGHT, FULL_ROW, WIDTH, Board
from .solver.evaluate import DELLACHERIE, evaluate_drop
from .solver.search import TOP_OUT_SCORE, Move, best_move, enumerate_drops
from .vision.colour_palette import CELL_MARGIN, PAINT_PIXEL_TOL
from .vision.colour_tracker import ColourTracker
from .vision.grid import (
    DEFAULT_HINT_COLOR,
    DEFAULT_NEXT_HINT_COLOR,
    GridClassifier,
    OwnPaint,
)
from .vision.readers import (
    FRESH_EVENTS,
    ColourVision,
    FrameReading,
    FrameVision,
    ShapeVision,
    VisionEvent,
    render_debug_frame,
)
from .vision.state import GameStateTracker

if TYPE_CHECKING:  # pragma: no cover - overlay/ pulls in PySide6; app must not
    from .overlay.renderer import Fractions, HintStyle


@dataclass
class CoachConfig:
    poll_rate: float = 15.0  # frames per second
    hint_color: str = DEFAULT_HINT_COLOR
    # Colour of the SECOND hint: where the piece after this one goes, on
    # the board this hint's placement leaves behind. It is drawn dashed
    # rather than faint (see overlay/renderer.py), so the two are told
    # apart by stroke even where their colours sit over similar pieces.
    next_hint_color: str = DEFAULT_NEXT_HINT_COLOR
    # Show that second hint at all. Some players want one target only.
    show_next_hint: bool = True
    # Opacity of the fill inside the CURRENT hint's cells. 0 -- the
    # default -- means the hint is an outline, drawn entirely in the outer
    # band of each cell, which is the part of a cell no reader samples: the
    # coach then cannot read its own drawing back at all
    # (tests/test_hint_invisibility.py measures exactly that, over every
    # committed window). Raising it RE-OPENS that path, and the reader is
    # told the same number so that what it un-composites is what was
    # painted -- the painter and the reader may not disagree about the
    # blend. It applies to the current hint only: the reader can be told
    # one paint colour, so only one hint may carry a fill.
    hint_fill_opacity: float = 0.0
    # The confidence floor the SHAPE reader's gate applies to its own
    # occupancy reading (vision/grid.py). The colour reader gates on
    # different evidence and ignores this: a frame whose cells are not
    # drawn flat, one with more than a tetromino resting on nothing, one
    # carrying cells that are neither the board's ground nor content, or
    # one showing a completed row, is refused outright.
    min_confidence: float = 0.15
    # Board height in rows (width is always 10). The single source the
    # stateful components (classifier, tracker, overlay) are seeded from;
    # everything downstream derives the height from its data.
    rows: int = DEFAULT_HEIGHT
    debug: bool = False
    # Directory to save captured frames into as PNGs (debugging aid for
    # region/scaling/tracking problems); None disables dumping.
    dump_dir: str | None = None
    # How long a hint may stay on screen after vision stops being able to
    # justify it. A one-frame glitch must not make the overlay flicker, but
    # a hint the coach cannot currently see the board behind is worse than
    # no hint: measured on a real session, a game-over screen and a "ROW
    # CLEARED" reward popup each left a stale placement painted over them,
    # once for 326 frames (21.7 s).
    #
    # Chosen by measurement, not taste. Rejected runs across every committed
    # window and two live sessions: 31 (a board wipe at game start), 28, 21
    # (the reward popup), 15, 14, 8, 7, 5, 4 ... and then 326, which is the
    # game-over screen. Everything legitimate is under ~31 frames and the
    # piece is still there behind it, so the hint should survive; the
    # pathological case is an order of magnitude longer. 45 frames (3 s at
    # the default poll rate) sits above every real obscuration measured and
    # far below the one that is not.
    max_stale_frames: int = 45
    # How many CONSECUTIVE frames from the start of the session to dump.
    # Consecutive matters: the tracker diffs each frame against the last
    # committed one, so only a contiguous run can replay live tracking
    # offline. ~700 frames is ~47 s at the default 15 fps.
    dump_limit: int = 700
    # Which tracker reads the frames. "colour" names a piece by the colour
    # it is drawn in, cell by cell, with the board re-derived from every
    # frame; "shape" is the shipped one, which matches binary occupancy
    # against one committed stack memory. Both are wired through the same
    # reading (vision/readers.py), so everything the user sees is decided
    # by the same policy either way.
    #
    # The default is the colour reader because of what the two put ON
    # SCREEN over the eight committed windows, replayed through this very
    # class (`python -m tetris_coach.race`, pinned in
    # tests/test_engine_race.py): over 474 frames it drew 0 hints for the
    # wrong piece against the shipped reader's 7, was never late (median 0
    # frames from a piece appearing to its hint, against 1 and a worst case
    # of 3), never moved a target mid-flight against 3, and left the
    # overlay blank on 4 frames against 32.
    #
    # "shape" stays reachable as an escape hatch (--tracker shape). The two
    # fail on different evidence -- the colour reader refuses a frame whose
    # cells are not drawn flat, the shipped one a frame its confidence gate
    # dislikes -- so a theme or a game the colour rules cannot read is a
    # flag away from the old behaviour rather than a rebuild.
    tracker: str = "colour"


# The overlay loop exits after this many CONSECUTIVE failed ticks (~3 s at
# the default poll rate): a persistently failing capture/vision path (e.g.
# a bad region, a disconnected display) must not spin forever.
MAX_CONSECUTIVE_TICK_FAILURES = 45


# How much better a freshly solved placement must be, in Dellacherie score
# units, before the overlay is allowed to move off the one it is already
# showing for the SAME piece over the SAME board. A hint that moves while
# the learner is looking at it is worse than useless — they cannot follow
# it, and the tool exists to build placement intuition — so a near-tie is
# resolved in favour of the target already on screen.
#
# One unit is one row of landing height, or one row/column transition; a
# hole is 4. Measured over 4000 reachable boards of solver self-play: when
# the upcoming piece becomes readable the 2-ply answer moves the placement
# on 22% of them, and the score it gains has median 2.0 (p25 1.0, p90 8.0).
# At 1.0 the margin suppresses 32% of those moves — every one worth a
# single transition or less — and keeps the rest, and no margin this size
# can ever suppress a switch that avoids a hole.
HINT_SWITCH_MARGIN = 1.0


def rescore(board: Board, move: Move, next_piece: str | None) -> Move | None:
    """``move``'s placement, dropped and scored again on ``board``.

    A :class:`Move`'s score is only meaningful for the board and the
    lookahead it was computed with, so deciding whether to keep a standing
    hint means putting it and its challenger on the same footing: the same
    board, the same upcoming piece. Same rotation, same column, re-dropped.

    Returns ``None`` when that placement is no longer legal at all.

    The arithmetic is :func:`~tetris_coach.solver.search.best_move`'s, spelled
    with its public parts (this ply's evaluation, plus the best reply to the
    board it leaves), so the two cannot drift apart about what a placement is
    worth — pinned by ``test_hint_stability.test_rescore_agrees_with_the_solver``.
    """
    for rotation, col, result in enumerate_drops(board, move.piece):
        if col != move.col or rotation.index != move.rotation.index:
            continue
        score = evaluate_drop(result, rotation, DELLACHERIE)
        if next_piece is not None:
            reply = best_move(result.board, next_piece)
            score += reply.score if reply is not None else TOP_OUT_SCORE
        return Move(
            piece=move.piece,
            rotation=rotation,
            col=col,
            row=result.landing_row,
            score=score,
            lines_cleared=result.lines_cleared,
            board=result.board,
        )
    return None


def _next_box_fractions(
    board_rect: Rect, next_rect: Rect | None
) -> tuple[float, float, float, float] | None:
    """The next-piece capture projected into the board's unit square.

    ``(x0, y0, x1, y1)`` as shares of the board region, or ``None`` when
    there is no next box or it lies wholly outside the board — the general
    case, a preview drawn in its own corner of the screen.

    Fractions rather than pixels because the capture may be Retina-scaled
    and the overlay window is measured in points: only ratios mean the
    same thing in both. Two callers need this projection and they must not
    disagree about it — :func:`compute_overlap_mask`, which asks which
    board CELLS the panel hides from the reader, and
    :func:`preview_keep_out`, which asks which PIXELS the overlay may not
    paint in because that second capture will read them back.
    """
    if next_rect is None or board_rect.width <= 0 or board_rect.height <= 0:
        return None
    fx0 = (next_rect.left - board_rect.left) / board_rect.width
    fx1 = (next_rect.left + next_rect.width - board_rect.left) / board_rect.width
    fy0 = (next_rect.top - board_rect.top) / board_rect.height
    fy1 = (next_rect.top + next_rect.height - board_rect.top) / board_rect.height
    # No overlap with the board's unit square [0, 1] x [0, 1].
    if fx1 <= 0.0 or fx0 >= 1.0 or fy1 <= 0.0 or fy0 >= 1.0:
        return None
    return (fx0, fy0, fx1, fy1)


def preview_keep_out(board_rect: Rect, next_rect: Rect | None) -> Fractions | None:
    """The part of the board the overlay may not paint in, as fractions.

    THE OVERLAY WINDOW IS SIZED TO THE WHOLE BOARD, and in the games this
    tool was built for the NEXT panel floats inside the board rectangle
    (that is why :func:`compute_overlap_mask` exists at all). So a hint on
    one of those cells is painted straight across the box that
    :meth:`FrameWorker.run_tick` grabs a moment later as ``next_image`` —
    a SECOND capture, read by a second reader, which the board reader's
    invisibility property says nothing about.

    And that reader cannot be taught its way out of it. It works at pixel
    level on a crop with no cell grid of its own, so it recognizes this
    tool's translucent FILL (a colour short of the hint's own magnitude,
    :data:`~tetris_coach.vision.colour_preview.PAINT_FILL_MAX`) and
    nothing else. The outline is full-opacity hint colour, which is not
    that — and could not be made that, because widening the rule to pure
    hint colour would swallow a NEXT piece the game happens to render in
    it. Worse, the paint is OPAQUE: recognizing it would only tell the
    reader that the piece it is naming has bars through it, not what was
    underneath. Measured on ``tests/fixtures/spawn_latency``, an O-hint on
    the four covered cells takes ``identify_preview`` from a piece to
    ``None`` on every frame tried, for the solid hint and the dashed one
    alike.

    Losing the next piece is not a cosmetic failure: hints drop to 1 ply,
    ``_precompute_next(None)`` clears the second hint and the instant flip
    on lock, and — because the next piece is one of the inputs the hint is
    solved for — the hint MOVES, which moves the paint, which can make the
    box readable again. That is the stutter feedback loop this project has
    already paid for twice.

    So the overlay keeps out of that rectangle entirely, and the property
    is the same one the outline design rests on: the reader does not have
    to recognize the paint, because the paint is not there. What the user
    loses is the part of a hint that lay under the game's own opaque
    panel, where the board is not visible to them either.
    """
    fractions = _next_box_fractions(board_rect, next_rect)
    if fractions is None:
        return None
    fx0, fy0, fx1, fy1 = fractions
    return (max(0.0, fx0), max(0.0, fy0), min(1.0, fx1), min(1.0, fy1))


def compute_overlap_mask(
    board_rect: Rect,
    next_rect: Rect | None,
    rows: int,
    width: int = WIDTH,
) -> frozenset[tuple[int, int]]:
    """Board cells whose SAMPLED PATCH falls under the next-piece preview box.

    Some games (e.g. ROAS Stacker) float the NEXT preview on top of the
    top corner of the playfield, inside the region the user must select as
    the board (pieces spawn and move through the top rows). The cells under
    that box therefore show the NEXT piece, not the board — a second
    tetromino's worth of added cells every frame, which turns the frame
    UNEXPLAINED and eventually trips a spurious BOARD_RESET.

    This returns the ``(row, col)`` cells the capture cannot observe. They
    are *unknown*, NOT empty — vision has no evidence about the board
    there, and asserting emptiness is its own bug (a piece resting across
    the boundary reads as a broken tetromino; a stack that grows into the
    corner reads as free space). Everything downstream treats them as
    unknown: see :class:`~tetris_coach.vision.state.GameStateTracker` for
    the belief the committed stack carries, and :meth:`CoachEngine._solver_board`
    for what the solver is handed.

    Geometry is done in board-relative fractions so it is Retina-agnostic
    (the capture may be scaled; only ratios matter): the next box is
    projected into the board rectangle's unit square, and a cell is masked
    when the box reaches the part of it vision actually READS -- the
    central patch both readers sample, inset :data:`CELL_MARGIN` on every
    side.

    That is the criterion rather than centre-in-rect, which this used to
    use, because the panel corrupts a cell exactly when it reaches the
    patch: a panel that only grazes a cell's border changes nothing about
    the mean read off its middle, and one that covers a quarter of the cell
    changes that mean whether or not it has reached the centre. Measured on
    the 8 committed live ROAS Stacker frames, a board rectangle that leaves
    the panel covering a column's cells without covering their centres puts
    two extra content cells in the reading -- and with the airborne budget
    at exactly 4 that refused EVERY frame of the session, silently, with
    the coach never speaking. Centre-in-rect had no slack to give; the
    patch test has a quarter of a cell of it on each side, and none of the
    committed windows' masks change (``tests/test_truth_oracle.py`` pins
    that this and the oracle's own copy still agree).

    When ``next_rect`` is ``None`` or does not overlap ``board_rect`` (the
    general case: a next box drawn in a separate area outside the board),
    the mask is empty and downstream behavior is unchanged.
    """
    fractions = _next_box_fractions(board_rect, next_rect)
    if fractions is None:
        return frozenset()
    fx0, fy0, fx1, fy1 = fractions
    masked: set[tuple[int, int]] = set()
    for r in range(rows):
        top, bottom = (r + CELL_MARGIN) / rows, (r + 1 - CELL_MARGIN) / rows
        if fy1 <= top or fy0 >= bottom:
            continue
        for c in range(width):
            left, right = (c + CELL_MARGIN) / width, (c + 1 - CELL_MARGIN) / width
            if fx0 < right and fx1 > left:
                masked.add((r, c))
    return frozenset(masked)


def selection_warning(
    unobservable_cells: frozenset[tuple[int, int]],
    width: int = WIDTH,
    tracker: str = CoachConfig.tracker,
) -> str | None:
    """A note for the user when the two rectangles hide what vision needs.

    One selection costs more than it looks: a next-piece box that covers
    the board's ENTIRE top row. It is an ordinary mis-selection, not an
    exotic one — any game whose NEXT queue is a horizontal bar across the
    top of the playfield lands here if the board rectangle is drawn around
    it — and what it costs depends on which tracker is reading.

    With ``--tracker shape`` it is FATAL and silently so: the board's
    background colour is bootstrapped from the top row's cells (see
    :mod:`~tetris_coach.vision.grid`), and with all of them behind a panel
    there is no sample to bootstrap from, so every frame is refused —
    correctly, and the user is left watching a coach that never says
    anything.

    With the colour reader it is lossy instead. The background is
    re-estimated from every cell that reads empty rather than from the top
    row (:meth:`~tetris_coach.vision.colour_palette.Palette.update_background`),
    so frames still read — but this game parks a new piece at the top edge
    until the player drags it down, and while it is up there it is behind
    the panel. Measured on the same geometry: the coach says nothing at all
    until the piece descends out of row 0, and then hints it at once.

    Returns None when the selection is fine (the common case, including
    the corner-preview geometry the mask exists for).
    """
    if not all((0, c) in unobservable_cells for c in range(width)):
        return None
    if tracker == "shape":
        return (
            "tetris-coach: the next-piece box covers the whole top row of the "
            "board region, which is where the board's background color is read "
            "from; no frame can be classified. Restart and draw the board "
            "rectangle below the next-piece bar, or the next-piece rectangle "
            "outside the board."
        )
    return (
        "tetris-coach: the next-piece box covers the whole top row of the "
        "board region, so a piece that has entered the board but not yet "
        "descended out of that row cannot be seen at all, and no hint will "
        "be shown for it until it does. Restart and draw the board rectangle "
        "below the next-piece bar, or the next-piece rectangle outside the "
        "board."
    )


def fill_warning(config: CoachConfig) -> str | None:
    """A note for the user when they have asked for a hint the reader can see.

    The hint is an outline in the outer band of each cell, which is the
    part of a cell neither reader samples, so by default nothing this tool
    paints can reach a reading -- measured over every committed capture
    window in ``tests/test_hint_invisibility.py``. A FILL is inside the
    patch, and then a correct reading depends on the rule that recognizes
    the paint and takes it back out again being right about this theme,
    this background and whatever piece is underneath. It has been wrong
    before, three times, and each time the user saw it as a hint that
    flickered or pointed at a piece they did not have.

    So the option stays and says so once, at startup, rather than being
    quietly equivalent.
    """
    if config.hint_fill_opacity <= 0.0:
        return None
    return (
        f"tetris-coach: --hint-fill {config.hint_fill_opacity:g} draws a "
        "translucent fill inside the hint, which is the one part of the "
        "overlay the coach can read back as board content. It is taken out "
        "again by recognizing it, rather than by never being there; if the "
        "hint flickers or names a piece you do not have, run without the flag."
    )


def hint_color_conflict(config: CoachConfig) -> str | None:
    """Why this pair of hint colours cannot be drawn safely, or ``None``.

    ``hint_styles`` guards one direction -- only the CURRENT hint may
    carry a fill, because the reader is told one paint colour -- but
    nothing stopped the SECOND hint's colour from being the first's.

    That matters only when a fill is configured, and then it matters a
    lot. ``colour_palette.own_paint_states`` decides PAINTED from a ring
    of hint colour around a clean patch, and the dashed outline rings its
    cells just as the solid one does (each side band is 3/5 painted,
    well past ``PAINT_EDGE_SHARE``). PAINTED means UN-COMPOSITE: subtract
    a blend that is not there, and the board's own bare ground comes out
    as a colour ``alpha/(1-alpha)`` of the way from the background AWAY
    from the hint -- which is content, which floats, which is a piece the
    player has not got. Measured on a bare-background frame with nothing
    drawn on it but a dashed second hint in ``--hint-color``'s own colour:
    the four cells read PAINTED and the engine names a phantom O falling
    at (8,4),(8,5),(9,4),(9,5). With ``--hint-fill 0`` the same frame
    reads clean, which is why the default is safe.

    The rule cannot tell the two hints apart, because nothing in the cell
    says which one painted it -- so the colours have to differ by more
    than the reader's own per-channel tolerance
    (``colour_palette.PAINT_PIXEL_TOL``), and that is checked against the
    reader's number rather than a number of this function's own.

    A colour this module cannot parse as hex turns the recognition rule
    OFF (see ``OwnPaint.for_hint_color``), so a first colour it cannot
    parse is safe -- there is no PAINTED state to reach. A second colour
    it cannot parse is not: the painter will happily draw ``"cyan"`` and
    nothing here can prove it is far enough away, so with a fill on it is
    refused rather than guessed at.
    """
    if config.hint_fill_opacity <= 0.0 or not config.show_next_hint:
        return None
    first = OwnPaint.for_hint_color(config.hint_color)
    if first is None:
        return None  # the recognition rule is off; there is no PAINTED to reach
    second = OwnPaint.for_hint_color(config.next_hint_color)
    if second is not None:
        apart = max(abs(a - b) for a, b in zip(first.color, second.color, strict=True))
        if apart > PAINT_PIXEL_TOL:
            return None
    return (
        f"tetris-coach: --next-hint-color {config.next_hint_color} is too close to "
        f"--hint-color {config.hint_color} to be drawn alongside --hint-fill "
        f"{config.hint_fill_opacity:g}. The fill is the one part of the overlay the "
        "coach reads back, and it recognizes it by the colour of the outline round "
        "it -- so a second hint in the same colour is read as a fill that was never "
        "painted, and un-compositing it turns bare board into a piece you have not "
        "got. Pick a second colour further away (more than "
        f"{PAINT_PIXEL_TOL:g} per channel, as a hex colour), or drop --hint-fill."
    )


def hint_styles(config: CoachConfig) -> tuple[HintStyle, HintStyle | None]:
    """How the two hints are drawn, for this configuration.

    Split out of :func:`run`, which cannot run off macOS, so that what the
    flags MEAN is testable headlessly -- the same reason
    ``cli.coach_config`` exists. What is decided here is only WHICH styles;
    where the paint lands is ``overlay.renderer``'s pure geometry, and
    whether it can be read back is measured in
    ``tests/test_hint_invisibility.py``.

    The second style is ``None`` when the session asked for one target
    only, which is how the overlay is told to draw nothing rather than
    being handed a style it should ignore.

    Only the CURRENT hint can carry a fill. The reader is told one paint
    colour (``CoachEngine._own_paint``), so a fill in the second colour
    would be a blend nothing could un-composite -- and a fill is the one
    thing here a reader can see at all.

    And a pair the reader could not tell apart is REFUSED rather than
    drawn (:func:`hint_color_conflict`). ``cli.main`` checks the same
    thing and exits with the message, so no invocation reaches this;
    raising is for the programmatic caller, where a startup error is the
    honest outcome and a phantom tetromino is not.
    """
    from .overlay.renderer import HintStyle

    conflict = hint_color_conflict(config)
    if conflict is not None:
        raise ValueError(conflict)
    current = HintStyle(
        color=config.hint_color,
        dashed=False,
        fill_opacity=config.hint_fill_opacity,
    )
    if not config.show_next_hint:
        return current, None
    return current, HintStyle(color=config.next_hint_color, dashed=True)


def make_vision(
    config: CoachConfig,
    unobservable_cells: frozenset[tuple[int, int]],
    own_paint: OwnPaint | None,
) -> FrameVision:
    """The tracker ``config.tracker`` names, wired for this session.

    Both readers are handed the same three session facts, because both can
    be got wrong in the same three ways: which board cells the game's own
    NEXT panel floats over (they are unknown, not empty), what this tool's
    own hint paint looks like coming back round through the capture, and
    how tall the board is.
    """
    if config.tracker == "colour":
        return ColourVision(
            rows=config.rows,
            unobservable_cells=unobservable_cells,
            own_paint=own_paint,
            debug=config.debug,
        )
    if config.tracker == "shape":
        return ShapeVision(
            rows=config.rows,
            min_confidence=config.min_confidence,
            unobservable_cells=unobservable_cells,
            own_paint=own_paint,
            debug=config.debug,
        )
    raise ValueError(f"unknown tracker {config.tracker!r} (expected 'colour' or 'shape')")


class CoachEngine:
    """GUI-free part of the loop: frame in, hint (Move or None) out.

    Owns the hint policy — which placement is on screen, when it may move
    and when it comes down — plus the solver calls and the precompute
    cache. What is falling and what the board is comes from a
    :class:`~tetris_coach.vision.readers.FrameVision`; the runner (macOS
    overlay loop or a test harness) feeds it frames.
    """

    def __init__(
        self,
        config: CoachConfig | None = None,
        unobservable_cells: frozenset[tuple[int, int]] | None = None,
        vision: FrameVision | None = None,
    ) -> None:
        self.config = config or CoachConfig()
        # Board cells the next-piece preview floats over (see
        # compute_overlap_mask): the capture reads the NEXT piece there, not
        # the board, so their captured value is discarded and the tracker is
        # told they are unknown. Empty by default, so a headless engine and
        # the common non-overlapping next box are unchanged.
        self._unobservable_cells: frozenset[tuple[int, int]] = unobservable_cells or frozenset()
        # This tool's overlay is ON SCREEN when the next frame is captured,
        # so every reader has to be told what its own paint looks like or it
        # reads the hint back as board content. It is the configured color,
        # not the default, or a session run with --hint-color would paint
        # one thing and look for another.
        # The opacity is the CONFIGURED one, not the module default: with
        # no fill (the default) there is nothing to un-composite and the
        # rules say so, and with one there is exactly the blend that was
        # painted. Handing over a number the painter does not use is how a
        # rule that rewrites cells goes wrong -- measured, in
        # tests/fixtures/hint_stutter.
        self._own_paint = OwnPaint.for_hint_color(
            self.config.hint_color, opacity=self.config.hint_fill_opacity
        )
        self.vision: FrameVision = vision or make_vision(
            self.config, self._unobservable_cells, self._own_paint
        )
        self.current_hint: Move | None = None
        self._predicted_board: Board | None = None
        self._precomputed: Move | None = None
        # The (piece, stack, next piece) the standing hint was solved for.
        # A frame whose reading matches it decided nothing new, so nothing
        # is solved and — the point — the target on screen cannot move.
        self._solved_for: tuple[str | None, tuple[int, ...], str | None] | None = None
        # The most recent frame's reading, for a caller that wants to know
        # what the hint it was handed was drawn on (the debug view, the
        # replay harness in race/engine.py). Never read by the policy.
        self.last_reading: FrameReading | None = None
        # Consecutive frames vision has refused. The hint is withdrawn
        # once this passes config.max_stale_frames (see process_frame),
        # and the user is told once why (see _report_silence).
        self._stale_frames = 0
        self._told_about_silence = False

    @property
    def tracker(self) -> GameStateTracker | ColourTracker:
        """The tracker reading this session's frames.

        A :class:`~tetris_coach.vision.state.GameStateTracker` under
        ``--tracker shape`` and a
        :class:`~tetris_coach.vision.colour_tracker.ColourTracker` under
        the default; they answer to different things, and code that reaches
        past the engine for one of them is code that knows which it wants.
        """
        return self.vision.tracker

    @property
    def _hint_is_provisional(self) -> bool:
        """Was the standing hint solved with less than the full 2-ply view?

        True while it was computed without an upcoming piece — the instant
        flip to a precompute, or a frame whose preview box said nothing.
        Not a flag but a reading of :attr:`_solved_for`: the upcoming piece
        recorded there is what the hint actually used, so the frame the box
        becomes readable is a changed input and the refinement happens
        through the ordinary re-solve (under the stability margin).
        """
        return (
            self.current_hint is not None
            and self._solved_for is not None
            and self._solved_for[2] is None
        )

    @property
    def second_hint(self) -> Move | None:
        """Where the NEXT piece goes, if the hint on screen is followed.

        The engine already solves this placement: :meth:`_precompute_next`
        assumes the standing hint is taken, and pre-solves the upcoming
        piece on the board that would leave, so that the moment the
        current piece locks the hint for the next one flips instantly.
        This is that same answer, shown a piece early.

        It is CONDITIONAL, and everything here is about not letting it
        mislead. A second target is only honest while the board it was
        computed on is the board the player is about to produce, so it is
        derived from the state rather than remembered:

        * There has to be a hint on screen. Without one there is nothing
          the second placement is conditional ON -- during a lock gap the
          engine pre-solves against the SETTLED board instead
          (:meth:`_update_hint`), which is a different prediction and not
          one to draw.
        * The precompute has to belong to the hint on screen. Every path
          that moves the hint calls :meth:`_precompute_next` immediately
          after, so this holds by construction; comparing the boards says
          so out loud, and the moment the player puts the piece somewhere
          else the next frame re-solves both and the second hint follows
          the first rather than pointing at a board that never happened.
        * The hint must clear no lines. A cleared row SHIFTS every row
          above it down, so the predicted board's coordinates and the
          screen's stop being the same coordinates, and a placement drawn
          at the predicted row would point at the wrong row of the board
          the player is looking at. There is no fix for that short of
          drawing it after the clear, so it is not drawn.
        * The two placements may not share a cell. They cannot: the
          predicted board has the first hint's cells filled and no drop
          lands in a filled cell. Checked anyway, because two hints
          overlapping would also put one hint's rotation badge in a cell
          belonging to the other, and a guarantee the renderer relies on
          is worth one set intersection a frame.
        """
        hint = self.current_hint
        upcoming = self._precomputed
        if hint is None or upcoming is None:
            return None
        if self._predicted_board != hint.board:
            return None
        if hint.lines_cleared:
            return None
        if not set(upcoming.cells).isdisjoint(hint.cells):
            return None
        return upcoming

    @property
    def classifier(self) -> GridClassifier:
        """The shipped reader's occupancy classifier (``--tracker shape``)."""
        vision = self.vision
        if isinstance(vision, ShapeVision):
            return vision.classifier
        raise AttributeError(
            "the colour tracker reads cells by colour and has no occupancy "
            "classifier; construct the engine with CoachConfig(tracker='shape')"
        )

    def process_frame(
        self,
        board_image: np.ndarray,
        next_image: np.ndarray | None,
    ) -> Move | None:
        """Digest one captured frame pair; return the hint to display."""
        reading = self.vision.read(board_image, next_image)
        self.last_reading = reading
        for note in reading.notes:
            print(note, flush=True)
        if not reading.accepted:
            self._stale_frames += 1
            if self._stale_frames > self.config.max_stale_frames:
                # Vision has not been able to justify this placement for
                # long enough that the board behind it has probably moved
                # on. Take it down rather than let it sit there being
                # confidently wrong; the tracker's own state is untouched,
                # so a readable frame re-solves immediately.
                self._withdraw()
                self._report_silence(reading)
            return self.current_hint  # a brief hold rides out a glitch
        self._stale_frames = 0
        self._told_about_silence = False
        self._update_hint(reading)
        return self.current_hint

    def _report_silence(self, reading: FrameReading) -> None:
        """Say once why the coach has stopped talking, when it stays stopped.

        A refused frame is invisible to the user: the overlay simply has
        nothing on it, and the one thing they can see is a tool that does
        not work. Every way vision can fail wholesale looks identical from
        the outside -- a board rectangle that is not over the board, a
        selection that includes the game's NEXT panel, a theme neither
        reader can read, a game that has been closed -- so the frame's own
        reason for refusing is the only thing that tells them apart, and it
        costs nothing to print it.

        Once per silence, at the same point the hint comes down, and reset
        by the first frame that reads: a message per frame at 15 fps is a
        different way of telling the user nothing.
        """
        if self._told_about_silence:
            return
        self._told_about_silence = True
        seconds = self._stale_frames / max(self.config.poll_rate, 1e-9)
        because = reading.refused_because or "no reason given"
        print(
            f"tetris-coach: no frame has been readable for {self._stale_frames} "
            f"frames ({seconds:.0f} s) -- {because}. The hint is off screen "
            "until one is. If this does not clear, the board rectangle may "
            "not be over the board, or may take in the game's own panels; "
            "restart to re-select the regions.",
            file=sys.stderr,
            flush=True,
        )

    def _withdraw(self) -> None:
        """Take the hint off the screen, and forget what was planned on it."""
        self.current_hint = None
        self._solved_for = None
        self._predicted_board = None
        self._precomputed = None

    def _update_hint(self, reading: FrameReading) -> None:
        """Decide what is on screen after this frame.

        The whole hint policy, and it is short on purpose: solve when an
        INPUT changed, hold otherwise, and let the target move only when
        the piece it was drawn for is no longer the piece in play.
        """
        if reading.inputs == self._solved_for:
            return  # nothing vision says has changed: the target stands
        piece = reading.falling_piece
        board = self._solver_board(reading.stack_rows)
        if piece is None:
            # No piece to advise on: a lock gap, a piece whose name has
            # been withdrawn, a board just re-anchored. Showing a placement
            # for a piece the player does not have is the failure this
            # tool's own user reported, so the coach says nothing.
            self._withdraw()
            if VisionEvent.PIECE_LOCKED in reading.events and reading.next_piece is not None:
                # Lock gap (piece locked, next spawn not yet visible):
                # pre-solve the upcoming piece on the settled board so the
                # spawn flips instantly through the validation guard below.
                self._predicted_board = board
                self._precomputed = best_move(board, reading.next_piece)
            self._solved_for = reading.inputs
            return
        # A hint is free to move when the piece it was chosen for is gone:
        # there is nothing on screen the user is still being asked to
        # follow. Everything else — the board shifting under the same
        # piece, the preview becoming readable — is held to the margin.
        fresh = (
            self.current_hint is None
            or self.current_hint.piece != piece
            or bool(FRESH_EVENTS.intersection(reading.events))
        )
        solved_with = reading.next_piece
        if (
            fresh
            and self._precomputed is not None
            and self._predicted_board == board
            and self._precomputed.piece == piece
        ):
            # Prediction held: flip to the precomputed hint instantly.
            self.current_hint = self._precomputed
            # It was solved 1-ply (the piece after it was unknown when it
            # was computed), so the reading is recorded as having named no
            # upcoming piece: the next frame that can see one counts as a
            # changed input and refines this to the full 2-ply answer —
            # under the margin, which is exactly the near-tie that rule
            # exists for.
            solved_with = None
        else:
            self.current_hint = self._steady_hint(board, piece, reading.next_piece, fresh)
        self._solved_for = (piece, reading.stack_rows, solved_with)
        self._precompute_next(reading.next_piece)

    def _steady_hint(
        self,
        board: Board,
        piece: str,
        next_piece: str | None,
        fresh: bool,
    ) -> Move | None:
        """The placement to show: the best one, unless the standing one is close.

        A hint that moves while the learner is looking at it is worse than
        useless — they cannot follow it, and the tool exists to build
        placement intuition — so a challenger has to beat the target
        already on screen by more than :data:`HINT_SWITCH_MARGIN` to take
        its place. Both are judged on THIS board with THIS lookahead (see
        :func:`rescore`), so the comparison is between two placements and
        not between two moments.

        ``fresh`` says the piece on screen is no longer the piece in play,
        and then there is nothing to be steady about: the new answer wins
        outright.
        """
        candidate = best_move(board, piece, next_piece)
        standing = self.current_hint
        if fresh or candidate is None or standing is None or standing.piece != piece:
            return candidate
        held = rescore(board, standing, next_piece)
        if held is None:
            return candidate  # the standing placement is no longer legal
        return candidate if candidate.score > held.score + HINT_SWITCH_MARGIN else held

    def _solver_board(self, stack_rows: tuple[int, ...]) -> Board:
        """The committed stack as the solver should see it.

        For an unobservable cell the reading holds no evidence, and the way
        that hurts is a hint planned INTO a cell the coach cannot see —
        worse than useless, since the overlay would draw it under the very
        panel that hides the board. A covered cell resting directly on the
        stack (or on the floor) is exactly where the stack plausibly
        continues up into the covered region, so the solver is handed those
        filled and keeps out.

        Covered cells with air under them are left empty. Filling every
        covered cell instead would permanently fill the top rows of the
        columns under the panel — and a column whose top row is filled is
        one :meth:`Board.drop` rejects outright, which would cost the user
        those columns for the entire session, in every board state, to
        guard a case that only arises near top-out.
        """
        unknown = self.vision.unknown_rows
        if not any(unknown):
            return Board(stack_rows)
        rows = list(stack_rows)
        for r in range(len(rows) - 1, -1, -1):
            support = rows[r + 1] if r + 1 < len(rows) else FULL_ROW  # the floor supports
            rows[r] |= unknown[r] & support
        return Board(tuple(rows))

    def _precompute_next(self, next_piece: str | None) -> None:
        """Assume the current hint is followed; pre-solve the next piece."""
        self._predicted_board = None
        self._precomputed = None
        if self.current_hint is None or next_piece is None:
            return
        predicted = self.current_hint.board
        self._predicted_board = predicted
        # The piece after next is unknown: 1-ply pre-solve (refined later
        # if the prediction misses).
        self._precomputed = best_move(predicted, next_piece)


@dataclass(frozen=True)
class TickResult:
    """Outcome of one capture->vision->solve tick."""

    hint: Move | None  # the hint to display; meaningful only when ok is True
    ok: bool  # the frame pair was captured and processed
    stop: bool  # the consecutive-failure cap was reached: shut the loop down
    # Where the piece after this one goes, on the board `hint` would leave
    # behind -- or None when the engine will not stand behind that
    # prediction (see CoachEngine.second_hint). Carried beside the hint
    # rather than asked for on the GUI thread: the engine is the worker
    # thread's, and a property read across threads is a race.
    second: Move | None = None


class FrameWorker:
    """Headless per-tick logic: grab a frame pair, digest it, account failures.

    This is exactly what runs on the worker thread in the live app; the Qt
    layer only schedules :meth:`run_tick` off the GUI thread and routes the
    returned :class:`TickResult` back to the overlay. Keeping it Qt-free
    lets tests drive the production tick logic without a display.
    """

    def __init__(
        self,
        engine: CoachEngine,
        frame_source: FrameSource,
        board_rect: Rect,
        next_rect: Rect | None,
    ) -> None:
        self.engine = engine
        self.frame_source = frame_source
        self.board_rect = board_rect
        self.next_rect = next_rect
        self.consecutive_failures = 0
        self._ticks = 0
        self._dump_disabled = False

    def _dump_frames(self, board_image: np.ndarray, next_image: np.ndarray | None) -> None:
        """Save captured frames as PNGs for offline inspection (debug aid).

        Every tick is saved, up to ``config.dump_limit`` frames. CONSECUTIVE
        frames are the point: the tracker explains each frame as a diff
        against the previous committed one, so a sampled dump (every Nth
        tick) can only ever replay as UNEXPLAINED and says nothing about
        live tracking. A contiguous run replays the real session offline.
        After the limit, every 100th tick is kept so a long session still
        leaves late evidence without unbounded disk growth.
        """
        dump_dir = self.engine.config.dump_dir
        self._ticks += 1
        if dump_dir is None or self._dump_disabled:
            return
        if self._ticks > self.engine.config.dump_limit and self._ticks % 100 != 0:
            return
        try:  # pragma: no cover - debug-only, Pillow is a dev dependency
            from pathlib import Path

            from PIL import Image

            out = Path(dump_dir)
            out.mkdir(parents=True, exist_ok=True)
            n = self._ticks
            Image.fromarray(board_image[:, :, ::-1]).save(out / f"board_{n:05d}.png")
            if next_image is not None:
                Image.fromarray(next_image[:, :, ::-1]).save(out / f"next_{n:05d}.png")
        except Exception:  # noqa: BLE001 - dumping must never break the loop
            self._dump_disabled = True  # give up quietly

    def run_tick(self) -> TickResult:
        """Run one blocking capture->vision->solve pass."""
        try:
            board_image = self.frame_source.grab(self.board_rect)
            next_image = (
                self.frame_source.grab(self.next_rect) if self.next_rect is not None else None
            )
            self._dump_frames(board_image, next_image)
            hint = self.engine.process_frame(board_image, next_image)
            second = self.engine.second_hint
        except Exception:  # noqa: BLE001 - one bad frame must not kill the loop
            self.consecutive_failures += 1
            if self.consecutive_failures == 1:
                # Log once per failure streak, never once per frame.
                traceback.print_exc()
                print(
                    "tetris-coach: frame processing failed; skipping frames.",
                    file=sys.stderr,
                )
            if self.consecutive_failures >= MAX_CONSECUTIVE_TICK_FAILURES:
                print(
                    f"tetris-coach: {self.consecutive_failures} consecutive frames "
                    "failed; exiting. Check that the selected regions still "
                    "cover the board and restart to re-select them.",
                    file=sys.stderr,
                )
                return TickResult(hint=None, ok=False, stop=True)
            return TickResult(hint=None, ok=False, stop=False)
        self.consecutive_failures = 0
        return TickResult(hint=hint, ok=True, stop=False, second=second)


def run(
    board_rect: Rect,
    next_rect: Rect | None,
    source: FrameSource | None = None,
    config: CoachConfig | None = None,
) -> None:  # pragma: no cover - macOS GUI loop
    """Start the capture/overlay loop (macOS only).

    The QTimer slot on the GUI thread only schedules work: each tick's
    blocking grab+vision+solve runs a :class:`FrameWorker` pass on a
    one-thread QThreadPool, and the result returns through an explicitly
    queued signal so the overlay repaints from the GUI thread. A busy flag
    (touched only on the GUI thread) skips timer fires while the worker is
    still on an earlier frame, so ticks never queue up behind a slow one.
    """
    from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
    from PySide6.QtWidgets import QApplication

    from .capture.screen import ScreenCapture
    from .overlay.window import OverlayWindow

    config = config or CoachConfig()
    # The next preview may float over the top corner of the selected board
    # region; name those unobservable cells once (the rects are fixed for
    # the session).
    unobservable_cells = compute_overlap_mask(board_rect, next_rect, config.rows)
    for warning in (
        selection_warning(unobservable_cells, tracker=config.tracker),
        fill_warning(config),
    ):
        if warning is not None:
            print(warning, file=sys.stderr)
    engine = CoachEngine(config, unobservable_cells=unobservable_cells)
    frame_source: FrameSource = source if source is not None else ScreenCapture()
    worker = FrameWorker(engine, frame_source, board_rect, next_rect)

    app = QApplication.instance() or QApplication([])
    current_style, second_style = hint_styles(config)
    window = OverlayWindow(
        board_rect,
        current_style,
        rows=config.rows,
        second_style=second_style,
        # The next box may lie under this window; the overlay may not paint
        # there, or the coach blinds its own preview reader.
        keep_out=preview_keep_out(board_rect, next_rect),
    )
    window.show()

    class TickSignals(QObject):
        finished = Signal(object)  # carries a TickResult

    class TickTask(QRunnable):
        """One tick on the pool thread; the pool auto-deletes it after run."""

        def __init__(self, signals: TickSignals) -> None:
            super().__init__()
            self._signals = signals

        def run(self) -> None:
            self._signals.finished.emit(worker.run_tick())

    pool = QThreadPool()
    pool.setMaxThreadCount(1)
    busy = False

    def schedule_tick() -> None:
        nonlocal busy
        if busy:
            return  # the worker is still on an earlier frame: skip this tick
        busy = True
        pool.start(TickTask(signals))

    def on_tick_finished(result: TickResult) -> None:
        nonlocal busy
        busy = False
        if result.stop:
            timer.stop()
            app.quit()
            return
        if result.ok:
            window.set_hint(result.hint, result.second)

    signals = TickSignals()
    # Explicitly queued: the signal is emitted from the pool thread, and the
    # slot repaints the overlay, which must only happen on the GUI thread.
    signals.finished.connect(on_tick_finished, Qt.ConnectionType.QueuedConnection)

    timer = QTimer()
    timer.timeout.connect(schedule_tick)
    timer.start(max(1, int(1000 / config.poll_rate)))
    app.exec()
    pool.waitForDone()


__all__ = [
    "HINT_SWITCH_MARGIN",
    "MAX_CONSECUTIVE_TICK_FAILURES",
    "CoachConfig",
    "CoachEngine",
    "FrameWorker",
    "TickResult",
    "compute_overlap_mask",
    "fill_warning",
    "hint_color_conflict",
    "hint_styles",
    "make_vision",
    "preview_keep_out",
    "render_debug_frame",
    "rescore",
    "run",
    "selection_warning",
]
