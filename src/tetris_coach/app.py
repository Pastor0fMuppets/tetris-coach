"""Main loop wiring: capture -> vision -> state -> solve -> overlay.

Precompute-ahead: while piece A falls toward its target, assume it lands
there and pre-solve piece B on the predicted board; when A locks, the hint
for B flips instantly. If the observed board disagrees with the prediction,
re-solve from the observed board.

This module is headless-importable; GUI/capture objects are only created
inside :func:`run`.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .capture.screen import FrameSource, Rect
from .core.board import Board
from .solver.search import Move, best_move
from .vision.grid import classify_grid
from .vision.pieces_vision import FallingPiece, identify_next
from .vision.state import GameEvent, GameStateTracker, Snapshot


@dataclass
class CoachConfig:
    poll_rate: float = 15.0  # frames per second
    hint_color: str = "#00e5ff"
    min_confidence: float = 0.15
    debug: bool = False


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

    The 20x10 grid is the raw observed occupancy; the falling piece and
    next piece come from the tracker's committed view of the same frame.
    (A graphical debug window is deferred to the live phase; see SPEC.md.)
    """
    grid = str(Board.from_grid(occupancy))
    falling_txt = "-"
    if falling is not None:
        falling_txt = (
            f"{falling.piece} rot{falling.rotation_index} "
            f"@ (row {falling.row}, col {falling.col})"
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
        self.tracker = GameStateTracker(confirm_frames=2)
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

    def process_frame(
        self,
        board_image: np.ndarray,
        next_image: np.ndarray | None,
    ) -> Move | None:
        """Digest one captured frame pair; return the hint to display."""
        occupancy, confidence = classify_grid(board_image)
        if confidence < self.config.min_confidence:
            return self.current_hint  # keep showing the last good hint
        next_piece = self._identify_next_cached(next_image)

        previous = self.tracker.committed
        events = self.tracker.update(occupancy, next_piece)
        committed = self.tracker.committed
        if self.config.debug and (events or committed != previous):
            print(
                render_debug_frame(
                    occupancy, confidence, committed, self.tracker.falling, events
                )
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
        if self._last_next_image is not None and np.array_equal(
            next_image, self._last_next_image
        ):
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


def run(
    board_rect: Rect,
    next_rect: Rect | None,
    source: FrameSource | None = None,
    config: CoachConfig | None = None,
) -> None:  # pragma: no cover - macOS GUI loop
    """Start the capture/overlay loop (macOS only)."""
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from .capture.screen import ScreenCapture
    from .overlay.renderer import HintStyle
    from .overlay.window import OverlayWindow

    config = config or CoachConfig()
    engine = CoachEngine(config)
    frame_source: FrameSource = source if source is not None else ScreenCapture()

    app = QApplication.instance() or QApplication([])
    window = OverlayWindow(board_rect, HintStyle(color=config.hint_color))
    window.show()

    consecutive_failures = 0

    def tick() -> None:
        nonlocal consecutive_failures
        try:
            board_image = frame_source.grab(board_rect)
            next_image = frame_source.grab(next_rect) if next_rect is not None else None
            hint = engine.process_frame(board_image, next_image)
        except Exception:  # noqa: BLE001 - one bad frame must not kill the loop
            consecutive_failures += 1
            if consecutive_failures == 1:
                # Log once per failure streak, never once per frame.
                traceback.print_exc()
                print(
                    "tetris-coach: frame processing failed; skipping frames.",
                    file=sys.stderr,
                )
            if consecutive_failures >= MAX_CONSECUTIVE_TICK_FAILURES:
                print(
                    f"tetris-coach: {consecutive_failures} consecutive frames "
                    "failed; exiting. Check that the selected regions still "
                    "cover the board and restart to re-select them.",
                    file=sys.stderr,
                )
                timer.stop()
                app.quit()
            return
        consecutive_failures = 0
        window.set_hint(hint)

    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(max(1, int(1000 / config.poll_rate)))
    app.exec()
