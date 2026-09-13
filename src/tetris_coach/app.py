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
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .capture.screen import FrameSource, Rect
from .core.board import DEFAULT_HEIGHT, Board
from .solver.search import Move, best_move
from .vision.grid import GridClassifier
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
    # Directory to save the first captured frames into as PNGs (debugging
    # aid for region/scaling problems); None disables dumping.
    dump_dir: str | None = None


# The overlay loop exits after this many CONSECUTIVE failed ticks (~3 s at
# the default poll rate): a persistently failing capture/vision path (e.g.
# a bad region, a disconnected display) must not spin forever.
MAX_CONSECUTIVE_TICK_FAILURES = 45


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


class CoachEngine:
    """GUI-free part of the loop: frame in, hint (Move or None) out.

    Owns the state tracker, the solver calls, and the precompute cache;
    the runner (macOS overlay loop or a test harness) feeds it frames.
    """

    def __init__(self, config: CoachConfig | None = None) -> None:
        self.config = config or CoachConfig()
        # The one place config.rows fans out to the stateful components;
        # both hold board-shaped state before the first frame exists, so
        # their row count cannot come from data.
        self.tracker = GameStateTracker(confirm_frames=2, rows=self.config.rows)
        # Stateful board classifier: its background memory keeps boards
        # readable when the stack legally reaches the visible top row
        # (where the per-frame top-row estimate inverts) and gates solid
        # overlays whose color is not the board's background (a bright
        # pause panel on a dark theme must never read as a board wipe).
        self.classifier = GridClassifier(
            rows=self.config.rows, min_confidence=self.config.min_confidence
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
            return self.current_hint  # keep showing the last good hint
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

        if GameEvent.PIECE_LOCKED in events or GameEvent.PIECE_SPAWNED in events:
            board = Board(committed.stack_rows)
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
            board = Board(committed.stack_rows)
            refined = best_move(board, self.current_hint.piece, committed.next_piece)
            if refined is not None:
                self.current_hint = refined
            self._hint_is_provisional = False
            self._precompute_next(committed.next_piece)
        return self.current_hint

    def _identify_next_cached(self, next_image: np.ndarray | None) -> str | None:
        """identify_next, skipped when the preview pixels did not change."""
        if next_image is None:
            return None
        if self._last_next_image is not None and np.array_equal(next_image, self._last_next_image):
            return self._last_next_piece
        piece = identify_next(next_image)
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
        self._dumped = 0

    def _dump_frames(self, board_image: np.ndarray, next_image: np.ndarray | None) -> None:
        """Save captured frames as PNGs for offline inspection (debug aid)."""
        dump_dir = self.engine.config.dump_dir
        if dump_dir is None or self._dumped >= 12:
            return
        try:  # pragma: no cover - debug-only, Pillow is a dev dependency
            from pathlib import Path

            from PIL import Image

            out = Path(dump_dir)
            out.mkdir(parents=True, exist_ok=True)
            n = self._dumped
            Image.fromarray(board_image[:, :, ::-1]).save(out / f"board_{n:03d}.png")
            if next_image is not None:
                Image.fromarray(next_image[:, :, ::-1]).save(out / f"next_{n:03d}.png")
            self._dumped += 1
        except Exception:  # noqa: BLE001 - dumping must never break the loop
            self._dumped = 12  # give up quietly

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
    engine = CoachEngine(config)
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
