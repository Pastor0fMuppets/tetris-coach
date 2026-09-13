"""Main loop wiring: capture -> vision -> state -> solve -> overlay.

Precompute-ahead: while piece A falls toward its target, assume it lands
there and pre-solve piece B on the predicted board; when A locks, the hint
for B flips instantly. If the observed board disagrees with the prediction,
re-solve from the observed board.

This module is headless-importable; GUI/capture objects are only created
inside :func:`run`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .capture.screen import FrameSource, Rect
from .core.board import Board
from .solver.search import Move, best_move
from .vision.grid import classify_grid
from .vision.pieces_vision import identify_next, split_grid
from .vision.state import GameEvent, GameStateTracker


@dataclass
class CoachConfig:
    poll_rate: float = 15.0  # frames per second
    hint_color: str = "#00e5ff"
    min_confidence: float = 0.15
    debug: bool = False


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

    def process_frame(
        self,
        board_image: np.ndarray,
        next_image: np.ndarray | None,
    ) -> Move | None:
        """Digest one captured frame pair; return the hint to display."""
        occupancy, confidence = classify_grid(board_image)
        if confidence < self.config.min_confidence:
            return self.current_hint  # keep showing the last good hint
        stack_grid, falling = split_grid(occupancy)
        next_piece = identify_next(next_image) if next_image is not None else None

        events = self.tracker.update(stack_grid, falling, next_piece)
        committed = self.tracker.committed
        if committed is None:
            return self.current_hint

        if GameEvent.BOARD_RESET in events:
            self._predicted_board = None
            self._precomputed = None
            self.current_hint = None

        if GameEvent.PIECE_LOCKED in events or GameEvent.PIECE_SPAWNED in events:
            board = Board(committed.stack_rows)
            piece = committed.falling_piece
            if piece is None:
                self.current_hint = None
                self._hint_is_provisional = False
            elif (
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
                self._hint_is_provisional = False
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

    def tick() -> None:
        board_image = frame_source.grab(board_rect)
        next_image = frame_source.grab(next_rect) if next_rect is not None else None
        hint = engine.process_frame(board_image, next_image)
        window.set_hint(hint)

    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(max(1, int(1000 / config.poll_rate)))
    app.exec()
