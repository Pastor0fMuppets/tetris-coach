"""What one frame told the coach, and the two ways of reading a frame.

The engine in :mod:`tetris_coach.app` decides what to DRAW -- which
placement to show, when it may move, and when it has to come down. What it
draws on top of (which piece is in flight, over what settled board, with
what coming next) is read here, and there are two readers because there are
two trackers:

``ShapeVision``   the shipped reader: a confidence-gated occupancy grid
                  (:mod:`.grid`) explained as a set difference against one
                  committed stack memory (:mod:`.state`, :mod:`.pieces_vision`).
``ColourVision``  the adopted reader: every frame read on its own from the
                  rendered colour of each cell (:mod:`.colour_tracker`).

Internally they share nothing. They meet exactly here: one
:class:`FrameReading` per frame, saying whether the frame could be read at
all, what is falling, what the settled board is, what the box holds, and
which transitions this frame is. The engine's hint policy is written
against this and nothing else, so BOTH trackers get the same hint
stability, the same stale-hint withdrawal, the same treatment of the cells
the preview panel covers and the same board handed to the solver -- each of
which was added in answer to a real user report, and none of which a change
of tracker may quietly drop.

A reading is what the coach knows AFTER the frame, not a diff: ``accepted``
False means this frame told it nothing new (the gate refused it, or the
capture is not showing a board), and the other fields then carry what still
stands from the last frame that did.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ..core.board import DEFAULT_HEIGHT, WIDTH, Board
from .colour_tracker import ColourTracker
from .colour_tracker import Event as ColourEvent
from .grid import HINT_PAINT, GridClassifier, OwnPaint
from .pieces_vision import FallingPiece, identify_next
from .state import GameEvent, GameStateTracker, Snapshot

Cell = tuple[int, int]


class VisionEvent(Enum):
    """A transition a frame reports, in the vocabulary the engine uses.

    Both trackers raise their own events; these are the ones the hint
    policy asks about. ``PIECE_UNNAMED`` and ``BOARD_RESET`` belong to the
    committed-memory design and have no counterpart in the colour reader
    (which has no memory to retract or re-anchor); ``LINES_CLEARED`` is the
    other way round. The engine reads a reading's STATE for what to show
    and its events only for whether the target may move, so a vocabulary
    one reader never speaks costs the other nothing.
    """

    PIECE_SPAWNED = auto()
    PIECE_LOCKED = auto()
    PIECE_UNNAMED = auto()
    BOARD_RESET = auto()
    LINES_CLEARED = auto()


#: Events after which a hint is free to move: the piece in play has
#: changed, so there is no target on screen for the user to still be
#: following. Every other re-solve is held to :data:`~tetris_coach.app.HINT_SWITCH_MARGIN`.
FRESH_EVENTS = frozenset({VisionEvent.PIECE_SPAWNED, VisionEvent.PIECE_LOCKED})

_SHAPE_EVENTS = {
    GameEvent.PIECE_SPAWNED: VisionEvent.PIECE_SPAWNED,
    GameEvent.PIECE_LOCKED: VisionEvent.PIECE_LOCKED,
    GameEvent.PIECE_UNNAMED: VisionEvent.PIECE_UNNAMED,
    GameEvent.BOARD_RESET: VisionEvent.BOARD_RESET,
}

_COLOUR_EVENTS = {
    ColourEvent.PIECE_SPAWNED: VisionEvent.PIECE_SPAWNED,
    ColourEvent.PIECE_LOCKED: VisionEvent.PIECE_LOCKED,
    ColourEvent.LINES_CLEARED: VisionEvent.LINES_CLEARED,
}


@dataclass(frozen=True)
class FrameReading:
    """Everything the engine is told about one captured frame."""

    accepted: bool  # vision could read this frame at all
    falling_piece: str | None  # the piece in flight, named
    stack_rows: tuple[int, ...]  # the settled board, one bitmask per row
    next_piece: str | None
    events: tuple[VisionEvent, ...] = ()
    #: Lines a debug session should print for this frame. Built only when
    #: the reader is constructed with ``debug=True``; empty otherwise, so
    #: an ordinary session formats nothing it will not show.
    notes: tuple[str, ...] = field(default=())

    @property
    def inputs(self) -> tuple[str | None, tuple[int, ...], str | None]:
        """What a hint is a function of: the piece, the board, what is next."""
        return (self.falling_piece, self.stack_rows, self.next_piece)


class FrameVision(Protocol):
    """A tracker, as the engine uses one."""

    #: Board cells the capture can never read, one bitmask per row. Fixed
    #: for the session: which cells the game's own UI floats over.
    unknown_rows: tuple[int, ...]

    #: The tracker doing the reading, for debug and for a caller that knows
    #: which reader it asked for. Nothing in the engine's hint policy may
    #: touch it: that is the whole point of the reading above.
    tracker: GameStateTracker | ColourTracker

    def read(
        self, board_image: NDArray[np.uint8], next_image: NDArray[np.uint8] | None
    ) -> FrameReading:
        """Digest one captured frame pair."""


def rows_from_cells(cells: frozenset[Cell] | None, rows: int) -> tuple[int, ...]:
    """``(row, col)`` cells as one bitmask per row; cells off the board drop."""
    out = [0] * rows
    for r, c in cells or ():
        if 0 <= r < rows and 0 <= c < WIDTH:
            out[r] |= 1 << c
    return tuple(out)


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


class ShapeVision:
    """The shipped reader: confidence gate, occupancy grid, committed stack.

    This is the whole vision half of what ``CoachEngine.process_frame``
    used to do inline, moved behind :class:`FrameReading` and otherwise
    untouched -- the gate, the blanking of the covered cells, the preview
    cache and the debug output are the same code they were.
    """

    def __init__(
        self,
        rows: int = DEFAULT_HEIGHT,
        min_confidence: float = 0.15,
        unobservable_cells: frozenset[Cell] | None = None,
        own_paint: OwnPaint | None = HINT_PAINT,
        debug: bool = False,
    ) -> None:
        self._min_confidence = min_confidence
        self._unobservable_cells = unobservable_cells or frozenset()
        self._own_paint = own_paint
        self._debug = debug
        # Stateful board classifier: its background memory keeps boards
        # readable when the stack legally reaches the visible top row
        # (where the per-frame top-row estimate inverts) and gates solid
        # overlays whose color is not the board's background (a bright
        # pause panel on a dark theme must never read as a board wipe).
        # The covered cells go in HERE as well as into the blanking below:
        # the classifier estimates the board's background from the top row,
        # and a preview box parked on two top-row cells otherwise poisons
        # that estimate on every frame and -- via the top-row cap -- rejects
        # every frame, so the memory never anchors and the session is
        # deadlocked (the diagnosed ROAS Stacker failure).
        # The hint color goes in too, because this tool's overlay is ON
        # SCREEN when the next frame is captured: the coach reads its own
        # paint back as board content unless the classifier is told what
        # that paint looks like.
        self.classifier = GridClassifier(
            rows=rows,
            min_confidence=min_confidence,
            unobservable_cells=self._unobservable_cells,
            own_paint=own_paint,
        )
        self.tracker = GameStateTracker(
            confirm_frames=2,
            rows=rows,
            unobservable_cells=self._unobservable_cells,
        )
        # Preview-vision cache: the preview image is byte-identical on most
        # frames (a piece stays in the box ~15 frames), so identify_next
        # only runs when the pixels actually change.
        self._last_next_image: np.ndarray | None = None
        self._last_next_piece: str | None = None
        # Debug status-line throttling: (confidence, occupied, gate) of the
        # last line built, plus a frame counter for the periodic reprint.
        self._debug_last_status: tuple[float, int, str] | None = None
        self._debug_frames = 0

    @property
    def unknown_rows(self) -> tuple[int, ...]:
        return self.tracker.unknown_rows

    def read(
        self, board_image: NDArray[np.uint8], next_image: NDArray[np.uint8] | None
    ) -> FrameReading:
        occupancy, confidence = self.classifier.classify(board_image)
        rejected = confidence < self._min_confidence
        notes: list[str] = []
        if self._debug:
            # Always-on compact status so a silently rejected or
            # never-committing stream is still diagnosable: report on any
            # change, and at least every 30 frames (~2 s).
            self._debug_frames += 1
            occupied = int(occupancy.sum())
            gate = "REJECTED" if rejected else "ok"
            status = (round(confidence, 2), occupied, gate)
            if status != self._debug_last_status or self._debug_frames % 30 == 0:
                self._debug_last_status = status
                kind = self.tracker.last_kind
                notes.append(
                    f"[vision] frame {self._debug_frames}: confidence {confidence:.2f} "
                    f"(gate {gate} at {self._min_confidence}), "
                    f"occupied {occupied}/{occupancy.size}, last frame kind "
                    f"{kind.name if kind is not None else '-'}"
                )
        if rejected:
            # The gate judged the BOARD image. The preview box is a
            # different region with its own readability, and the tracker's
            # flip rules date a deal in CAPTURES -- so a rejected capture
            # still has to reach the preview clock, or a wipe (which
            # rejects the board and blanks the box alike) stops that clock
            # for its whole duration and the flip on the far side is dated
            # against a frame seconds earlier. Board state is untouched.
            self.tracker.observe_preview(self._identify_next_cached(next_image))
            held = self.tracker.committed
            return FrameReading(
                accepted=False,
                falling_piece=held.falling_piece,
                stack_rows=held.stack_rows,
                next_piece=held.next_piece,
                notes=tuple(notes),
            )
        # Drop what the capture read under the preview box: confidence above
        # was judged on the full grid, but neither the tracker nor the debug
        # view below may take the NEXT piece for board content. The tracker
        # knows those cells are unknown rather than empty (unobservable_cells).
        occupancy = self._blanked(occupancy)
        next_piece = self._identify_next_cached(next_image)

        previous = self.tracker.committed
        events = self.tracker.update(occupancy, next_piece)
        committed = self.tracker.committed
        if self._debug and (events or committed != previous):
            notes.append(
                render_debug_frame(occupancy, confidence, committed, self.tracker.falling, events)
            )
        return FrameReading(
            accepted=True,
            falling_piece=committed.falling_piece,
            stack_rows=committed.stack_rows,
            next_piece=committed.next_piece,
            events=tuple(_SHAPE_EVENTS[event] for event in events),
            notes=tuple(notes),
        )

    def _blanked(self, occupancy: NDArray[np.bool_]) -> NDArray[np.bool_]:
        """Occupancy with the unobservable cells cleared.

        Clearing is not a claim that those cells are empty -- it discards a
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


class ColourVision:
    """The adopted reader: :class:`~.colour_tracker.ColourTracker` per frame.

    Thin by design. The tracker already answers every question a reading
    asks -- it reads the board and the NEXT box itself, it knows the cells
    the panel covers, and it says outright when the capture is not showing
    a board (``board_visible``) rather than reporting furniture as a stack.
    What is added here is the session-level wiring the engine owes it: the
    cells the preview covers, this tool's own hint colour, and the cache
    that keeps an unchanged preview box from being read twice.
    """

    def __init__(
        self,
        rows: int = DEFAULT_HEIGHT,
        unobservable_cells: frozenset[Cell] | None = None,
        own_paint: OwnPaint | None = HINT_PAINT,
        debug: bool = False,
    ) -> None:
        self._debug = debug
        self._debug_frames = 0
        self._debug_last: tuple[bool, str | None, str | None, tuple[int, ...]] | None = None
        self.tracker = ColourTracker(
            rows=rows,
            cols=WIDTH,
            unobservable_cells=unobservable_cells,
            paint=own_paint,
        )
        # The covered cells are unknown, not empty: the tracker classifies
        # nothing there, and the engine's solver board decides what to
        # believe about them (see CoachEngine._solver_board).
        self.unknown_rows = rows_from_cells(unobservable_cells, rows)
        # Preview cache, for the same reason the shipped reader has one:
        # the box is byte-identical on most frames. Handing the tracker no
        # crop is exactly handing it an unchanged one -- an unreadable or
        # absent box leaves its last reading standing -- so a duplicate
        # frame is skipped rather than read again.
        self._last_next_image: np.ndarray | None = None

    def read(
        self, board_image: NDArray[np.uint8], next_image: NDArray[np.uint8] | None
    ) -> FrameReading:
        crop = next_image
        if (
            crop is not None
            and self._last_next_image is not None
            and np.array_equal(crop, self._last_next_image)
        ):
            crop = None
        report = self.tracker.update(board_image, crop)
        if report.board_visible and next_image is not None:
            # Only after a frame the tracker actually read: a refused frame
            # never reaches the box, so caching its crop would skip the
            # reading altogether the next time the same pixels arrive.
            self._last_next_image = np.array(next_image, copy=True)
        piece = None if report.falling is None else report.falling.piece
        notes: list[str] = []
        if self._debug:
            self._debug_frames += 1
            state = (report.board_visible, piece, report.next_piece, report.stack_rows)
            if state != self._debug_last or self._debug_frames % 30 == 0:
                self._debug_last = state
                cells = 0 if report.falling is None else len(report.falling.cells)
                events = ", ".join(event.name for event in report.events) or "-"
                notes.append(
                    f"[vision] frame {self._debug_frames}: "
                    f"board {'read' if report.board_visible else 'UNREADABLE'}, "
                    f"falling {piece or '-'} ({cells} cells), "
                    f"next {report.next_piece or '-'}, events {events}\n"
                    f"{Board(report.stack_rows)}"
                )
        return FrameReading(
            accepted=report.board_visible,
            falling_piece=piece,
            stack_rows=report.stack_rows,
            next_piece=report.next_piece,
            events=tuple(_COLOUR_EVENTS[event] for event in report.events),
            notes=tuple(notes),
        )


__all__ = [
    "FRESH_EVENTS",
    "ColourVision",
    "FrameReading",
    "FrameVision",
    "ShapeVision",
    "VisionEvent",
    "render_debug_frame",
    "rows_from_cells",
]
