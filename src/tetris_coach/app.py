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
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .capture.screen import FrameSource, Rect
from .core.board import DEFAULT_HEIGHT, FULL_ROW, WIDTH, Board
from .solver.evaluate import DELLACHERIE, evaluate_drop
from .solver.search import TOP_OUT_SCORE, Move, best_move, enumerate_drops
from .vision.grid import GridClassifier, OwnPaint
from .vision.pieces_vision import FallingPiece, identify_next
from .vision.state import GameEvent, GameStateTracker, Snapshot


@dataclass
class CoachConfig:
    poll_rate: float = 15.0  # frames per second
    hint_color: str = "#00e5ff"
    min_confidence: float = 0.15
    # Board height in rows (width is always 10). The single source the
    # stateful components (classifier, tracker, overlay) are seeded from;
    # everything downstream derives the height from its data.
    rows: int = DEFAULT_HEIGHT
    debug: bool = False
    # Directory to save captured frames into as PNGs (debugging aid for
    # region/scaling/tracking problems); None disables dumping.
    dump_dir: str | None = None
    # How many CONSECUTIVE frames from the start of the session to dump.
    # Consecutive matters: the tracker diffs each frame against the last
    # committed one, so only a contiguous run can replay live tracking
    # offline. ~700 frames is ~47 s at the default 15 fps.
    dump_limit: int = 700


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


def render_debug_frame(
    occupancy: NDArray[np.bool_],
    confidence: float,
    committed: Snapshot,
    falling: FallingPiece | None,
    events: list[GameEvent],
) -> str:
    """Terminal debug view of one committed frame: what vision sees.

    The rows x 10 grid is the raw observed occupancy; the falling piece and
    next piece come from the tracker's committed view of the same frame.
    (A graphical debug window is deferred to the live phase; see SPEC.md.)
    """
    grid = str(Board.from_grid(occupancy))
    falling_txt = "-"
    if falling is not None:
        falling_txt = (
            f"{falling.piece} rot{falling.rotation_index} @ (row {falling.row}, col {falling.col})"
        )
    events_txt = ", ".join(event.name for event in events) or "-"
    return (
        f"{grid}\n"
        f"falling: {falling_txt}  next: {committed.next_piece or '-'}  "
        f"confidence: {confidence:.2f}  events: {events_txt}"
    )


def compute_overlap_mask(
    board_rect: Rect,
    next_rect: Rect | None,
    rows: int,
    width: int = WIDTH,
) -> frozenset[tuple[int, int]]:
    """Board cells whose center falls under the next-piece preview box.

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
    when its center ``((c + 0.5) / width, (r + 0.5) / rows)`` lies inside
    that projection. Center-in-rect tolerates a slightly loose next
    selection without masking a cell merely grazed at its border.

    When ``next_rect`` is ``None`` or does not overlap ``board_rect`` (the
    general case: a next box drawn in a separate area outside the board),
    the mask is empty and downstream behavior is unchanged.
    """
    if next_rect is None or board_rect.width <= 0 or board_rect.height <= 0:
        return frozenset()
    fx0 = (next_rect.left - board_rect.left) / board_rect.width
    fx1 = (next_rect.left + next_rect.width - board_rect.left) / board_rect.width
    fy0 = (next_rect.top - board_rect.top) / board_rect.height
    fy1 = (next_rect.top + next_rect.height - board_rect.top) / board_rect.height
    # No overlap with the board's unit square [0, 1] x [0, 1].
    if fx1 <= 0.0 or fx0 >= 1.0 or fy1 <= 0.0 or fy0 >= 1.0:
        return frozenset()
    masked: set[tuple[int, int]] = set()
    for r in range(rows):
        cy = (r + 0.5) / rows
        if not (fy0 <= cy <= fy1):
            continue
        for c in range(width):
            cx = (c + 0.5) / width
            if fx0 <= cx <= fx1:
                masked.add((r, c))
    return frozenset(masked)


def selection_warning(
    unobservable_cells: frozenset[tuple[int, int]],
    width: int = WIDTH,
) -> str | None:
    """A note for the user when the two rectangles hide what vision needs.

    Only one selection is fatal rather than merely lossy: a next-piece box
    that covers the board's ENTIRE top row. The board's background color
    is bootstrapped from the top row's cells (see
    :mod:`~tetris_coach.vision.grid`), so with all of them behind a panel
    there is no sample to bootstrap from and every frame is refused —
    correctly, but silently, and the user is left watching a coach that
    never says anything. It is an ordinary mis-selection, not an exotic
    one: any game whose NEXT queue is a horizontal bar across the top of
    the playfield lands here if the board rectangle is drawn around it.

    Returns None when the selection is fine (the common case, including
    the corner-preview geometry the mask exists for).
    """
    if all((0, c) in unobservable_cells for c in range(width)):
        return (
            "tetris-coach: the next-piece box covers the whole top row of the "
            "board region, which is where the board's background color is read "
            "from; no frame can be classified. Restart and draw the board "
            "rectangle below the next-piece bar, or the next-piece rectangle "
            "outside the board."
        )
    return None


class CoachEngine:
    """GUI-free part of the loop: frame in, hint (Move or None) out.

    Owns the state tracker, the solver calls, and the precompute cache;
    the runner (macOS overlay loop or a test harness) feeds it frames.
    """

    def __init__(
        self,
        config: CoachConfig | None = None,
        unobservable_cells: frozenset[tuple[int, int]] | None = None,
    ) -> None:
        self.config = config or CoachConfig()
        # Board cells the next-piece preview floats over (see
        # compute_overlap_mask): the capture reads the NEXT piece there, not
        # the board, so their captured value is discarded and the tracker is
        # told they are unknown. Empty by default, so a headless engine and
        # the common non-overlapping next box are unchanged.
        self._unobservable_cells: frozenset[tuple[int, int]] = unobservable_cells or frozenset()
        # The one place config.rows fans out to the stateful components;
        # both hold board-shaped state before the first frame exists, so
        # their row count cannot come from data.
        self.tracker = GameStateTracker(
            confirm_frames=2,
            rows=self.config.rows,
            unobservable_cells=self._unobservable_cells,
        )
        # Stateful board classifier: its background memory keeps boards
        # readable when the stack legally reaches the visible top row
        # (where the per-frame top-row estimate inverts) and gates solid
        # overlays whose color is not the board's background (a bright
        # pause panel on a dark theme must never read as a board wipe).
        # The covered cells go in HERE as well as into the blanking below:
        # the classifier estimates the board's background from the top row,
        # and a preview box parked on two top-row cells otherwise poisons
        # that estimate on every frame and — via the top-row cap — rejects
        # every frame, so the memory never anchors and the session is
        # deadlocked (the diagnosed ROAS Stacker failure).
        # The hint color goes in too, because this tool's overlay is ON
        # SCREEN when the next frame is captured: the coach reads its own
        # paint back as board content unless the classifier is told what
        # that paint looks like. It is the configured color, not the
        # default, or a session run with --hint-color would paint one
        # thing and look for another.
        # ...and the preview reader is told the same thing, for the same
        # reason: the box floats over the top corner of the playfield, so
        # a hint drawn in that corner is drawn over the BOX, where a
        # tetromino of our own paint would otherwise read as the piece
        # the game is about to deal.
        self._own_paint = OwnPaint.for_hint_color(self.config.hint_color)
        self.classifier = GridClassifier(
            rows=self.config.rows,
            min_confidence=self.config.min_confidence,
            unobservable_cells=self._unobservable_cells,
            own_paint=self._own_paint,
        )
        self.current_hint: Move | None = None
        self._predicted_board: Board | None = None
        self._precomputed: Move | None = None
        # True while current_hint came from the 1-ply precompute and should
        # be refined to 2-ply once the upcoming piece is known.
        self._hint_is_provisional = False
        # Preview-vision cache: the preview image is byte-identical on most
        # frames (a piece stays in the box ~15 frames), so identify_next
        # only runs when the pixels actually change.
        self._last_next_image: np.ndarray | None = None
        self._last_next_piece: str | None = None
        # Debug status-line throttling: (confidence, occupied, gate) of the
        # last line printed, plus a frame counter for the periodic reprint.
        self._debug_last_status: tuple[float, int, str] | None = None
        self._debug_frames = 0

    def process_frame(
        self,
        board_image: np.ndarray,
        next_image: np.ndarray | None,
    ) -> Move | None:
        """Digest one captured frame pair; return the hint to display."""
        occupancy, confidence = self.classifier.classify(board_image)
        rejected = confidence < self.config.min_confidence
        if self.config.debug:
            # Always-on compact status so a silently rejected or
            # never-committing stream is still diagnosable: print on any
            # change, and at least every 30 frames (~2 s).
            self._debug_frames += 1
            occupied = int(occupancy.sum())
            gate = "REJECTED" if rejected else "ok"
            status = (round(confidence, 2), occupied, gate)
            if status != self._debug_last_status or self._debug_frames % 30 == 0:
                self._debug_last_status = status
                kind = self.tracker.last_kind
                print(
                    f"[vision] frame {self._debug_frames}: confidence {confidence:.2f} "
                    f"(gate {gate} at {self.config.min_confidence}), "
                    f"occupied {occupied}/{occupancy.size}, last frame kind "
                    f"{kind.name if kind is not None else '-'}",
                    flush=True,
                )
        if rejected:
            # The gate judged the BOARD image. The preview box is a
            # different region with its own readability, and the tracker's
            # flip rules date a deal in CAPTURES — so a rejected capture
            # still has to reach the preview clock, or a wipe (which
            # rejects the board and blanks the box alike) stops that clock
            # for its whole duration and the flip on the far side is dated
            # against a frame seconds earlier. Board state is untouched.
            self.tracker.observe_preview(self._identify_next_cached(next_image))
            return self.current_hint  # keep showing the last good hint
        # Drop what the capture read under the preview box: confidence above
        # was judged on the full grid, but neither the tracker nor the debug
        # view below may take the NEXT piece for board content. The tracker
        # knows those cells are unknown rather than empty (unobservable_cells).
        occupancy = self._blanked(occupancy)
        next_piece = self._identify_next_cached(next_image)

        previous = self.tracker.committed
        events = self.tracker.update(occupancy, next_piece)
        committed = self.tracker.committed
        if self.config.debug and (events or committed != previous):
            print(
                render_debug_frame(occupancy, confidence, committed, self.tracker.falling, events)
            )

        if GameEvent.BOARD_RESET in events:
            self._predicted_board = None
            self._precomputed = None
            self.current_hint = None

        if GameEvent.PIECE_UNNAMED in events:
            # The preview named the piece entering, and that name is over:
            # this frame ruled it out, or it stood unconfirmed for longer
            # than a hint may (a hold swap or a restart puts a different
            # piece under the name with nothing to contradict it, so the
            # clock is the only thing that ends it). Take the hint off the
            # screen rather than leaving a placement for a piece the
            # player does not have: showing nothing is what the coach does
            # for any piece it cannot name, and the frames after this name
            # it from shape as soon as they can.
            # The precompute goes too — it was solved for a board that
            # assumed this hint would be followed.
            self.current_hint = None
            self._hint_is_provisional = False
            self._predicted_board = None
            self._precomputed = None

        if GameEvent.PIECE_LOCKED in events or GameEvent.PIECE_SPAWNED in events:
            board = self._solver_board(committed.stack_rows)
            piece = committed.falling_piece
            if piece is None:
                # Lock gap (piece locked, next spawn not yet visible):
                # pre-solve the upcoming piece on the settled board so the
                # spawn flips instantly through the validation guard below.
                self.current_hint = None
                self._hint_is_provisional = False
                self._predicted_board = None
                self._precomputed = None
                if committed.next_piece is not None:
                    self._predicted_board = board
                    self._precomputed = best_move(board, committed.next_piece)
            else:
                if (
                    self._precomputed is not None
                    and self._predicted_board is not None
                    and board == self._predicted_board
                    and self._precomputed.piece == piece
                ):
                    # Prediction held: flip to the precomputed hint instantly.
                    self.current_hint = self._precomputed
                    self._hint_is_provisional = True
                else:
                    self.current_hint = best_move(board, piece, committed.next_piece)
                    # Provisional means "computed with less than full 2-ply
                    # information": refine once the preview becomes readable.
                    self._hint_is_provisional = committed.next_piece is None
                self._precompute_next(committed.next_piece)
        elif (
            self._hint_is_provisional
            and self.current_hint is not None
            and committed.falling_piece == self.current_hint.piece
            and committed.next_piece is not None
        ):
            # Quiet frame: upgrade the instant 1-ply hint to the full 2-ply
            # answer now that the upcoming piece is known.
            board = self._solver_board(committed.stack_rows)
            refined = best_move(board, self.current_hint.piece, committed.next_piece)
            if refined is not None:
                self.current_hint = refined
            self._hint_is_provisional = False
            self._precompute_next(committed.next_piece)
        return self.current_hint

    def _blanked(self, occupancy: NDArray[np.bool_]) -> NDArray[np.bool_]:
        """Occupancy with the unobservable cells cleared.

        Clearing is not a claim that those cells are empty — it discards a
        reading that is about the game's UI, not the board. What they
        actually hold is carried as a belief by the tracker, which is told
        which cells they are.

        Returns a copy when any cell is cleared so a cached classifier
        output is never mutated in place; the array is returned unchanged
        (no copy) when there is nothing to clear.
        """
        if not self._unobservable_cells:
            return occupancy
        blanked = occupancy.copy()
        rows, cols = blanked.shape
        for r, c in self._unobservable_cells:
            if 0 <= r < rows and 0 <= c < cols:
                blanked[r, c] = False
        return blanked

    def _solver_board(self, stack_rows: tuple[int, ...]) -> Board:
        """The committed stack as the solver should see it.

        For an unobservable cell the committed stack holds a belief, and
        the way a wrong belief hurts is a hint planned INTO a cell the
        coach cannot see — worse than useless, since the overlay would draw
        it under the very panel that hides the board. A covered cell
        resting directly on the stack (or on the floor) is exactly where
        the stack plausibly continues up into the covered region, so the
        solver is handed those filled and keeps out.

        Covered cells with air under them are handed over as believed.
        Filling every covered cell instead would permanently fill the top
        rows of the columns under the panel — and a column whose top row is
        filled is one :meth:`Board.drop` rejects outright, which would cost
        the user those columns for the entire session, in every board state,
        to guard a case that only arises near top-out.
        """
        unknown = self.tracker.unknown_rows
        if not any(unknown):
            return Board(stack_rows)
        rows = list(stack_rows)
        for r in range(len(rows) - 1, -1, -1):
            support = rows[r + 1] if r + 1 < len(rows) else FULL_ROW  # the floor supports
            rows[r] |= unknown[r] & support
        return Board(tuple(rows))

    def _identify_next_cached(self, next_image: np.ndarray | None) -> str | None:
        """identify_next, skipped when the preview pixels did not change."""
        if next_image is None:
            return None
        if self._last_next_image is not None and np.array_equal(next_image, self._last_next_image):
            return self._last_next_piece
        piece = identify_next(next_image, own_paint=self._own_paint)
        # Copy: capture sources may reuse the frame buffer between grabs.
        self._last_next_image = np.array(next_image, copy=True)
        self._last_next_piece = piece
        return piece

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
        return TickResult(hint=hint, ok=True, stop=False)


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
    from .overlay.renderer import HintStyle
    from .overlay.window import OverlayWindow

    config = config or CoachConfig()
    # The next preview may float over the top corner of the selected board
    # region; name those unobservable cells once (the rects are fixed for
    # the session).
    unobservable_cells = compute_overlap_mask(board_rect, next_rect, config.rows)
    warning = selection_warning(unobservable_cells)
    if warning is not None:
        print(warning, file=sys.stderr)
    engine = CoachEngine(config, unobservable_cells=unobservable_cells)
    frame_source: FrameSource = source if source is not None else ScreenCapture()
    worker = FrameWorker(engine, frame_source, board_rect, next_rect)

    app = QApplication.instance() or QApplication([])
    window = OverlayWindow(board_rect, HintStyle(color=config.hint_color), rows=config.rows)
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
            window.set_hint(result.hint)

    signals = TickSignals()
    # Explicitly queued: the signal is emitted from the pool thread, and the
    # slot repaints the overlay, which must only happen on the GUI thread.
    signals.finished.connect(on_tick_finished, Qt.ConnectionType.QueuedConnection)

    timer = QTimer()
    timer.timeout.connect(schedule_tick)
    timer.start(max(1, int(1000 / config.poll_rate)))
    app.exec()
    pool.waitForDone()
