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

A resync adopts the observed board as the stack, minus the piece in
flight — a component that rests on nothing AND touches row 0, where
settled content cannot be. Freezing a piece in flight into the stack is
the corruption that made every hint after the first one wrong: the
piece's own descent then reads as stack cells vanishing, which nothing
can explain, so the tracker resets again and re-absorbs it one row lower,
the whole way down. Floating LOWER DOWN is not evidence of anything —
naive gravity leaves settled cells hanging over holes after a clear — so
it is adopted like the rest of the board (see ``strip_piece_in_flight``).
Everything the resync does not hold back is adopted unconditionally: an
earlier "re-anchoring on the board already believed fires no event"
short-circuit existed only because the strip removed real content, and
with the strip narrowed no unexplainable frame can reach it (measured:
0 of 82249 frames whose strip lands exactly on the committed stack are
UNEXPLAINED; every one is FALLING or OCCLUDED, neither of which resets).
Left in, it was a livelock: a real floating remnant the strip deleted
made the resync a no-op for good, and the tracker never adopted it.

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
from .pieces_vision import FallingPiece, FrameKind, explain_grid, strip_piece_in_flight


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
        # Preview memory, for naming a piece the top edge has cut in half.
        # The last non-None preview reading, and the one it replaced.
        self._preview_piece: str | None = None
        self._entering_hint: str | None = None
        # Whether the PREVIOUS frame could read the preview box. A flip is
        # only news of the deal happening NOW when the box was readable on
        # both sides of it (see update()).
        self._preview_read = False
        # Frames since the hint was set. A hint is evidence about ONE deal,
        # so it must not outlive it, and its age is how the rules below
        # tell the flip reporting the deal now committing from a flip left
        # over from a deal ago (see update()).
        self._hint_age = 0

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
        # The piece that LEAVES the preview is the piece entering the board:
        # a game deals the previewed piece and shows the one after it. That
        # is the only evidence anything has about the name of a piece the
        # board region's top edge has cut in half — 1-3 cells at row 0 fit
        # several tetrominoes, and this session's game parks them there for
        # seconds. Only a change of a KNOWN preview counts: the preview
        # reading is None whenever the box is mid-animation or unreadable,
        # and None -> X says nothing about what was dealt.
        # ...and only a flip the box is in a position to be REPORTING. A
        # flip says "the piece that was here has been dealt"; it does not
        # say WHEN. The box goes unreadable in bursts (mid-animation,
        # mid-flash, a pale piece against a pale ground), and across a
        # burst long enough to cover a whole tenure the box's value has
        # moved on TWICE: X -> [Y never read] -> Z reads as a flip naming
        # X, a whole deal after X entered — the "confused two pieces"
        # failure, and it is refused. Two consecutive readable frames
        # cannot straddle two deals (a tenure is ~14-22 frames here), so
        # the flip is sound exactly when the previous frame read the box
        # too. Measured on the spawn_latency window: the T dealt at 00198
        # sat unread in the box for 18 frames, and the flip at 00216
        # carried the I before it.
        previously_read = self._preview_read
        self._preview_read = next_piece is not None
        self._hint_age += 1
        if next_piece is not None and next_piece != self._preview_piece:
            if self._preview_piece is not None and previously_read:
                self._entering_hint = self._preview_piece
                self._hint_age = 0
            self._preview_piece = next_piece
        explanation = explain_grid(
            rows,
            self._committed.stack_rows,
            self._last_falling,
            max_missing_cells=self._max_missing_cells,
            unknown_rows=self.unknown_rows,
            entering_hint=self._entering_hint,
        )
        self.last_kind = explanation.kind
        if explanation.hint_refuted:
            # A preview reading is a HYPOTHESIS, and this frame falsified
            # it: the piece coming in from above showed cells no placement
            # of the hinted piece fits. Dropped here, before anything is
            # decided on it, so the rest of this frame — and every frame
            # until the next flip — reads from shape alone. A piece two
            # cells wide fits an O as well as an L, but its next row down
            # does not, so a misnamed piece contradicts itself within a
            # frame or two of descending and costs a briefly withheld
            # hint rather than a confidently wrong one.
            self._entering_hint = None

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
                # A piece at the top edge — entering, or just spawned, and
                # floating — is the one thing on the frame that is
                # demonstrably not settled, and a resync is exactly when
                # one is likely to be sitting there (an attach mid-game, in
                # a game that parks spawns at the top edge). It is left out
                # rather than frozen into the stack. Everything else the
                # frame shows is adopted, floating or not: a piece absorbed
                # lower down is taken back out by _carried_piece on its next
                # descending frame, while content deleted here is gone.
                resynced = strip_piece_in_flight(rows, self.unknown_rows)
                self._committed = Snapshot(resynced, None, next_piece)
                self._pending = None
                self._pending_kind = None
                self._pending_count = 0
                self._unexplained_rows = None
                self._unexplained_count = 0
                self._last_falling = None
                # A resync is a new world (a new game, garbage, a mid-game
                # attach). What the preview shows still holds, but which
                # piece was dealt into THIS board does not.
                self._entering_hint = None
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
            #
            # A name the entering hint supplied is NOT an observation: the
            # frame showed 1-3 cells that several tetrominoes fit, and the
            # preview picked one. Good enough to hint on, and not good
            # enough to become evidence — explain_grid uses this piece's
            # NAME to refuse a lock ("a piece cannot change identity
            # between flight and lock"), so a misnamed entering piece would
            # block its own lock and, four identical frames later, reset
            # the board. In this game a piece goes straight from the top
            # edge to a hard drop, so the correction the ordinary rules
            # apply on the way down never gets to run.
            self._last_falling = None if explanation.hinted_name else explanation.falling

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
            # ...and the deal the hint describes is over unless the hint
            # IS this deal's. A lock is the game dealing again, and the
            # preview reports that deal by changing at the moment the new
            # piece spawns — which is the frame this commit is debouncing,
            # so the hint of the deal now beginning is already set and its
            # age is at most the debounce (plus the one frame the flip may
            # lead the lock becoming visible by). An OLDER hint is one
            # whose own deal has just ended under it: it names the piece
            # that locked, not the fragment now at the top edge, and left
            # standing it would name every later fragment too — the
            # preview is unreadable in bursts, and one burst spanning a
            # whole tenure would poison every top-edge naming after it.
            # Counting locks instead of frames could not tell the two
            # apart: both show exactly one lock since the hint was set.
            if self._hint_age > self._confirm_frames:
                self._entering_hint = None
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
