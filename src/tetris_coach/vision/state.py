"""Debounced game-state tracking.

Raw per-frame observations are noisy: line-clear animations, piece spawn
flashes, and mid-transition captures produce transient states. The tracker
therefore requires ``confirm_frames`` (default 2) consecutive identical
observations before committing a state change and emitting events.

An observation's identity is (stack rows, falling piece name, next piece
name) — the falling piece's *position* is deliberately excluded, since it
changes every frame while the piece drops.

Events:

- ``PIECE_SPAWNED``: a (new) falling piece appeared.
- ``PIECE_LOCKED``: the stack grew consistently with the previous falling
  piece locking (+4 cells, minus 10 per cleared line).
- ``BOARD_RESET``: the stack changed in a way no lock explains (board
  cleared, new game, or a game-over animation wiping the field).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray

from .pieces_vision import FallingPiece


class GameEvent(Enum):
    PIECE_SPAWNED = auto()
    PIECE_LOCKED = auto()
    BOARD_RESET = auto()


@dataclass(frozen=True)
class Snapshot:
    """Committed, debounced game state."""

    stack_rows: tuple[int, ...]  # bitmask per row, row 0 = top
    falling_piece: str | None
    next_piece: str | None

    @property
    def stack_cells(self) -> int:
        return sum(row.bit_count() for row in self.stack_rows)


def _rows_from_grid(grid: NDArray[np.bool_]) -> tuple[int, ...]:
    rows, cols = grid.shape
    return tuple(
        int(sum(1 << c for c in range(cols) if grid[r, c])) for r in range(rows)
    )


def _is_lock_consistent(old_cells: int, new_cells: int) -> bool:
    """True if the cell-count change matches a piece lock with 0-4 clears."""
    for cleared in range(5):
        if new_cells == old_cells + 4 - 10 * cleared:
            return True
    return False


class GameStateTracker:
    """Tracks committed game state across debounced frames."""

    def __init__(self, confirm_frames: int = 2) -> None:
        if confirm_frames < 1:
            raise ValueError("confirm_frames must be >= 1")
        self._confirm_frames = confirm_frames
        self._committed: Snapshot | None = None
        self._pending: Snapshot | None = None
        self._pending_count = 0
        self._last_falling: FallingPiece | None = None

    @property
    def committed(self) -> Snapshot | None:
        return self._committed

    @property
    def falling(self) -> FallingPiece | None:
        """Most recent raw falling-piece observation (not debounced)."""
        return self._last_falling

    def update(
        self,
        stack_grid: NDArray[np.bool_],
        falling: FallingPiece | None,
        next_piece: str | None,
    ) -> list[GameEvent]:
        """Feed one frame's observation; returns events committed this frame."""
        self._last_falling = falling
        candidate = Snapshot(
            stack_rows=_rows_from_grid(stack_grid),
            falling_piece=falling.piece if falling is not None else None,
            next_piece=next_piece,
        )

        if candidate == self._committed:
            self._pending = None
            self._pending_count = 0
            return []

        if candidate == self._pending:
            self._pending_count += 1
        else:
            self._pending = candidate
            self._pending_count = 1

        if self._pending_count < self._confirm_frames:
            return []

        previous = self._committed
        self._committed = candidate
        self._pending = None
        self._pending_count = 0
        return self._diff(previous, candidate)

    @staticmethod
    def _diff(old: Snapshot | None, new: Snapshot) -> list[GameEvent]:
        events: list[GameEvent] = []
        if old is None:
            if new.falling_piece is not None:
                events.append(GameEvent.PIECE_SPAWNED)
            return events

        stack_changed = new.stack_rows != old.stack_rows
        locked = False
        if stack_changed:
            if old.falling_piece is not None and _is_lock_consistent(
                old.stack_cells, new.stack_cells
            ):
                locked = True
                events.append(GameEvent.PIECE_LOCKED)
            else:
                events.append(GameEvent.BOARD_RESET)

        if new.falling_piece is not None and (
            old.falling_piece is None
            or new.falling_piece != old.falling_piece
            or locked
            or GameEvent.BOARD_RESET in events
        ):
            events.append(GameEvent.PIECE_SPAWNED)
        return events
