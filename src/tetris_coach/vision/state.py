"""Debounced game-state tracking anchored on a committed stack memory.

The tracker keeps ONE authoritative piece of memory: the committed stack,
as one bitmask per board row (the row count is set at construction,
default 20) — never ``None``. Every frame the falling piece is
DERIVED by :func:`~.pieces_vision.explain_grid` as a set difference
against that memory, never guessed geometrically from a single frame.

The committed stack changes only through an explicit, debounced,
structurally verified transition (a FALLING/QUIET/LOCKED commit, or a
board-reset resync). A frame the classifier cannot explain produces no
candidate, touches nothing, and the last committed state is held — there
is no best-effort commit path.

An observation's identity is (stack rows, falling piece name, next piece
name) — the falling piece's *position* is deliberately excluded, since it
changes every frame while the piece drops.

Events:

- ``PIECE_SPAWNED``: a (new) falling piece appeared.
- ``PIECE_LOCKED``: a lock was structurally verified — the piece's cells
  became immutable-and-explained (rows cleared, or the next spawn
  appeared). NOTE the deliberate semantics: vision cannot distinguish
  lock delay from lock without the game's timer, so PIECE_LOCKED fires at
  the verification point, not at touchdown. A piece resting on the stack
  stays FALLING (and the hint stays up) until the lock is revealed.
- ``BOARD_RESET``: ``reset_confirm_frames`` consecutive IDENTICAL
  unexplainable frames — a stable new world memory cannot explain (new
  game, garbage rising, mid-game attach). The tracker re-anchors on the
  observed board. Clear animations never trip it (their frames morph, and
  the settled board is explained as a lock); a one-frame glitch never
  does (the counter resets on the next coherent frame).

A resync adopts the observed board as the stack, minus the visible part
of a piece the board region's top edge has cut in half: that fragment is
evidence of a piece still in flight, and freezing it into the stack is
the corruption that made every hint after the first one wrong.

``unobservable_cells`` names board cells the capture can never read (a
game UI panel floating over the playfield). The observed value there is
meaningless and is discarded; what the committed stack holds for those
cells is a BELIEF, seeded empty at bootstrap/resync and thereafter moved
only by an explained transition (a lock merging its hidden half in, a
clear shifting rows through). An OCCLUDED frame — a piece the covered
region is hiding — is coherent: it holds the committed state and, unlike
an unexplainable frame, never counts toward a board reset.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray

from ..core.board import DEFAULT_HEIGHT, WIDTH
from .pieces_vision import FallingPiece, FrameKind, explain_grid, strip_entering_piece


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


def _rows_from_grid(grid: NDArray[np.bool_]) -> tuple[int, ...]:
    rows, cols = grid.shape
    return tuple(int(sum(1 << c for c in range(cols) if grid[r, c])) for r in range(rows))


def _rows_from_cells(cells: frozenset[tuple[int, int]] | None, rows: int) -> tuple[int, ...]:
    """``(row, col)`` cells as one bitmask per row; cells off the board drop."""
    out = [0] * rows
    for r, c in cells or ():
        if 0 <= r < rows and 0 <= c < WIDTH:
            out[r] |= 1 << c
    return tuple(out)


class GameStateTracker:
    """Tracks committed game state across debounced frames."""

    def __init__(
        self,
        confirm_frames: int = 2,
        reset_confirm_frames: int = 4,
        max_missing_cells: int = 2,
        rows: int = DEFAULT_HEIGHT,
        unobservable_cells: frozenset[tuple[int, int]] | None = None,
    ) -> None:
        if confirm_frames < 1:
            raise ValueError("confirm_frames must be >= 1")
        if reset_confirm_frames < 1:
            raise ValueError("reset_confirm_frames must be >= 1")
        if rows < 1:
            raise ValueError("rows must be >= 1")
        self._confirm_frames = confirm_frames
        # Board cells the capture can never read, as one bitmask per row.
        # Public so the consumer that hands a board to the solver can apply
        # its own policy to the same cells (see CoachEngine._solver_board).
        self.unknown_rows: tuple[int, ...] = _rows_from_cells(unobservable_cells, rows)
        self._reset_confirm_frames = reset_confirm_frames
        self._max_missing_cells = max_missing_cells
        # Bootstrap IS the ordinary rule set: a fresh game diffs cleanly
        # from the empty snapshot; a mid-game attach resyncs via the reset
        # rule. The committed snapshot is never None. ``rows`` exists ONLY
        # for this bootstrap: the empty committed stack must exist before
        # any frame, so its length cannot be derived from data; every later
        # commit takes its length from the observed frame.
        self._committed = Snapshot((0,) * rows, None, None)
        self._pending: Snapshot | None = None
        self._pending_kind: FrameKind | None = None
        self._pending_count = 0
        self._last_falling: FallingPiece | None = None
        # How explain_grid classified the most recent update() frame; for
        # debug/status display only, never for control flow.
        self.last_kind: FrameKind | None = None
        self._unexplained_rows: tuple[int, ...] | None = None
        self._unexplained_count = 0

    @property
    def committed(self) -> Snapshot:
        return self._committed

    @property
    def falling(self) -> FallingPiece | None:
        """Last *coherent* raw falling-piece observation (position included).

        Not necessarily from the current frame: unexplained frames and
        lock transitions do not overwrite it.
        """
        return self._last_falling

    def update(self, occupancy: NDArray[np.bool_], next_piece: str | None) -> list[GameEvent]:
        """Feed one frame's full occupancy grid; returns committed events."""
        # Whatever the capture read in an unobservable cell is not board
        # content: discard it here so no rule can mistake it for one. The
        # frame is then all-zero there, which explain_grid reads as "no
        # evidence" (not "empty") because it is told which cells those are.
        rows = tuple(
            o & ~u for o, u in zip(_rows_from_grid(occupancy), self.unknown_rows, strict=True)
        )
        explanation = explain_grid(
            rows,
            self._committed.stack_rows,
            self._last_falling,
            max_missing_cells=self._max_missing_cells,
            unknown_rows=self.unknown_rows,
        )
        self.last_kind = explanation.kind

        if explanation.kind is FrameKind.OCCLUDED:
            # Coherent: the covered region is hiding a piece. Hold the
            # committed state (and any pending transition), and — the whole
            # point — do NOT let a stationary hidden piece reach the reset
            # debounce, which would wipe the board it is resting on.
            self._unexplained_rows = None
            self._unexplained_count = 0
            return []

        if explanation.kind is FrameKind.UNEXPLAINED:
            if rows == self._unexplained_rows:
                self._unexplained_count += 1
            else:
                self._unexplained_rows = rows
                self._unexplained_count = 1
            if self._unexplained_count >= self._reset_confirm_frames:
                # Re-anchor on the observed board. Unobservable cells were
                # blanked above, so the belief is seeded EMPTY: a resync is
                # usually a fresh game, and there is no evidence for
                # anything else. (Seeding them filled instead would be a
                # different lie, and a permanent one — a board whose top
                # rows are filled has columns nothing can be dropped into.)
                # A piece the top edge has cut in half is the one thing on
                # the frame that is demonstrably not settled, and a resync
                # is exactly when one is likely to be sitting there (an
                # attach mid-game, in a game that parks spawns at the top
                # edge). It is left out rather than frozen into the stack.
                self._committed = Snapshot(
                    strip_entering_piece(rows, self.unknown_rows), None, next_piece
                )
                self._pending = None
                self._pending_kind = None
                self._pending_count = 0
                self._unexplained_rows = None
                self._unexplained_count = 0
                self._last_falling = None
                return [GameEvent.BOARD_RESET]
            # Pending is untouched: a torn frame between the confirmations
            # of a real transition must not restart its count.
            return []

        self._unexplained_rows = None
        self._unexplained_count = 0
        if explanation.kind is FrameKind.FALLING and explanation.falling is not None:
            # LOCKED frames must NOT overwrite this until they commit: the
            # L1/C1 anchors need the pre-lock position to re-derive the
            # same candidate on the confirmation frame.
            self._last_falling = explanation.falling

        candidate = Snapshot(
            stack_rows=explanation.stack_rows,
            falling_piece=(explanation.falling.piece if explanation.falling is not None else None),
            next_piece=next_piece,
        )

        if candidate == self._committed:
            self._pending = None
            self._pending_kind = None
            self._pending_count = 0
            return []

        if candidate == self._pending:
            self._pending_count += 1
            self._pending_kind = explanation.kind
        else:
            self._pending = candidate
            self._pending_kind = explanation.kind
            self._pending_count = 1

        if self._pending_count < self._confirm_frames:
            return []

        previous = self._committed
        kind = self._pending_kind
        self._committed = candidate
        self._pending = None
        self._pending_kind = None
        self._pending_count = 0
        if kind is FrameKind.LOCKED:
            # The pre-lock piece is absorbed into the stack; from now on
            # the anchor is the residual spawn (or nothing).
            self._last_falling = explanation.falling
        return self._events(previous, candidate, kind)

    @staticmethod
    def _events(old: Snapshot, new: Snapshot, kind: FrameKind | None) -> list[GameEvent]:
        events: list[GameEvent] = []
        if kind is FrameKind.LOCKED:
            events.append(GameEvent.PIECE_LOCKED)
            if new.falling_piece is not None:
                # The common lock+spawn commit: a fresh spawn even when the
                # name equals the locked piece's.
                events.append(GameEvent.PIECE_SPAWNED)
        elif new.falling_piece is not None and new.falling_piece != old.falling_piece:
            # None -> name, or a hold swap changing the name. Same-name
            # hold swaps are invisible and correctly quiet; next-piece-only
            # changes commit quietly with no events.
            events.append(GameEvent.PIECE_SPAWNED)
        return events
