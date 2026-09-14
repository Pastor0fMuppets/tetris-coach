"""Replay of 96 CONSECUTIVE frames from the session that produced Bug 1.

``tests/fixtures/live_session/`` holds ticks 00040-00135 of a real failing
ROAS Stacker session (10 x 12, light theme) with its NEXT preview floating
over board cells (0,8), (0,9), (1,8) and (1,9). See the fixtures' README
for the geometry and the frame-by-frame narrative; the frames are RGB PNGs
and the capture pipeline hands the engine BGR, so each is flipped on load.

Consecutive is the whole point: the tracker explains every frame as a diff
against the previous committed one, so only a contiguous run replays what
the live session actually did.

What it did, measured on these frames BEFORE the top-edge rule:

    UNEXPLAINED 34 of 96   BOARD_RESET 7   PIECE_LOCKED 0
    spawns named 2         frames with a hint 35
    committed stack, last frame:  row 0 = ...####...  <- PHANTOM

At frame 59 the I piece hard-drops to the floor AND the next piece appears
with only two cells inside the board region (its top half is above the
capture). 4 + 2 added cells is not two tetrominoes, so the frame went
UNEXPLAINED; four identical unexplainable frames later the reset debounce
re-anchored on the observed board and committed the 2-cell fragment as
stack content. From there the board handed to the solver had blocks in it
that do not exist — the user's report that the first hint was right and
every hint after it was random.

AFTER (same frames, same gate):

    UNEXPLAINED 0          BOARD_RESET 0   PIECE_LOCKED 4
    spawns named 4         frames with a hint 62
    committed stack, last frame:  exactly the board in the image

32 of the 96 frames are still REJECTED by the confidence gate before the
tracker ever sees them (this game's light theme reads at 0.08 while a
piece is the only thing on the board). That is a separate defect; it costs
resolution, not correctness — a rejected frame holds the last good hint —
and this replay pins it as-is so a vision change that alters it is seen.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from tetris_coach.app import CoachConfig, CoachEngine, compute_overlap_mask
from tetris_coach.capture.screen import Rect
from tetris_coach.solver.search import Move
from tetris_coach.vision.pieces_vision import FrameKind
from tetris_coach.vision.state import GameEvent

FIXTURES = Path(__file__).parent / "fixtures" / "live_session"
ROWS = 12
GATE = CoachConfig().min_confidence

SESSION_BOARD = Rect(left=303, top=313, width=477, height=579)
SESSION_NEXT = Rect(left=686, top=316, width=94, height=94)
COVERED = frozenset({(0, 8), (0, 9), (1, 8), (1, 9)})

# The board as it really stands on the last frame of the window: an I on
# the floor at cols 0-3, a second I beside it at cols 4-7, the O at cols
# 0-1 above them and the last I locked across cols 2-5.
FINAL_STACK = (0,) * 9 + (0b0000000011, 0b0000111111, 0b0011111111)


def load(name: str) -> np.ndarray:
    """One captured frame as the capture pipeline would hand it over (BGR)."""
    return np.asarray(Image.open(FIXTURES / name))[:, :, ::-1]


def frame_numbers() -> list[str]:
    return [p.stem.split("_")[1] for p in sorted(FIXTURES.glob("board_*.png"))]


class Tick:
    """What one replayed frame did."""

    def __init__(
        self,
        number: str,
        confidence: float,
        kind: FrameKind | None,
        events: list[GameEvent],
        stack_rows: tuple[int, ...],
        falling: str | None,
        hint: Move | None,
    ) -> None:
        self.number = number
        self.confidence = confidence
        self.kind = kind
        self.events = events
        self.stack_rows = stack_rows
        self.falling = falling
        self.hint = hint

    @property
    def accepted(self) -> bool:
        return self.confidence >= GATE


@lru_cache(maxsize=1)
def replay() -> tuple[Tick, ...]:
    """Feed the whole window through CoachEngine, wired as app.run wires it."""
    covered = compute_overlap_mask(SESSION_BOARD, SESSION_NEXT, rows=ROWS)
    engine = CoachEngine(CoachConfig(rows=ROWS), unobservable_cells=covered)
    update = engine.tracker.update
    seen: list[GameEvent] = []

    def spy(occupancy, next_piece):  # type: ignore[no-untyped-def]
        events = update(occupancy, next_piece)
        seen.extend(events)
        return events

    engine.tracker.update = spy  # type: ignore[method-assign]
    ticks: list[Tick] = []
    for number in frame_numbers():
        board = load(f"board_{number}.png")
        preview = FIXTURES / f"next_{number}.png"
        seen.clear()
        # The classifier is stateful, so this must read the same frame the
        # engine is about to digest — classify() is a pure function of the
        # image plus the memory, and the engine re-runs it identically.
        _occupancy, confidence = engine.classifier.classify(board)
        hint = engine.process_frame(board, load(preview.name) if preview.exists() else None)
        committed = engine.tracker.committed
        ticks.append(
            Tick(
                number=number,
                confidence=confidence,
                kind=engine.tracker.last_kind,
                events=list(seen),
                stack_rows=committed.stack_rows,
                falling=committed.falling_piece,
                hint=hint,
            )
        )
    return tuple(ticks)


def tick(number: str) -> Tick:
    return next(t for t in replay() if t.number == number)


def test_the_session_geometry_is_the_readme_geometry() -> None:
    assert compute_overlap_mask(SESSION_BOARD, SESSION_NEXT, rows=ROWS) == COVERED
    assert len(frame_numbers()) == 96
    assert frame_numbers()[0] == "00040"
    assert frame_numbers()[-1] == "00135"


def test_no_frame_is_unexplained_and_the_board_never_resets() -> None:
    # The headline: the session survives frame 59 instead of dying there.
    kinds = Counter(t.kind for t in replay() if t.kind is not None)
    assert kinds[FrameKind.UNEXPLAINED] == 0, "an unexplainable frame is back"
    events = Counter(e for t in replay() for e in t.events)
    assert events[GameEvent.BOARD_RESET] == 0, "spurious reset"
    assert events[GameEvent.PIECE_LOCKED] == 4
    assert events[GameEvent.PIECE_SPAWNED] == 4


def test_the_committed_stack_never_holds_a_phantom_cell() -> None:
    # The corruption itself: a piece the top edge cut in half must never
    # reach the committed stack. Nothing is ever committed above row 9,
    # and the stack only ever grows (this window clears no lines), so an
    # absorbed fragment cannot hide behind a later commit either.
    previous: tuple[int, ...] | None = None
    for t in replay():
        assert all(row == 0 for row in t.stack_rows[:9]), f"{t.number}: {t.stack_rows}"
        if previous is not None:
            assert all(old & ~new == 0 for old, new in zip(previous, t.stack_rows, strict=True)), (
                f"{t.number}: committed cells vanished"
            )
        previous = t.stack_rows
    assert replay()[-1].stack_rows == FINAL_STACK


def test_frame_59_is_a_lock_plus_a_piece_entering_from_above() -> None:
    # The exact frame that used to kill the session: the I hard-drops to
    # the floor and the next piece appears with two cells on the grid.
    before = tick("00058")
    assert not before.accepted  # the descent frames are below the gate
    assert tick("00059").kind is FrameKind.LOCKED
    assert tick("00059").accepted
    # Debounced: the lock commits on its second consecutive frame, and the
    # committed stack is the I on the floor and nothing else.
    assert tick("00060").stack_rows == (0,) * 11 + (0b1111,)
    assert GameEvent.PIECE_LOCKED in tick("00060").events


def test_the_entering_piece_holds_rather_than_being_guessed() -> None:
    # Frames 59-67 are byte-identical: this game has no gravity, so the
    # spawned piece SITS at the top edge. Two cells side by side fit an O,
    # an S, a Z, a J and an L, so the frames are coherent but nameless —
    # held, never guessed, and never counted toward a reset.
    held = [tick(f"000{n}") for n in range(61, 74)]
    assert all(t.kind is FrameKind.OCCLUDED for t in held)
    assert all(t.stack_rows == (0,) * 11 + (0b1111,) for t in held)
    assert all(t.falling is None for t in held)


def test_pieces_are_named_and_hinted_after_frame_59() -> None:
    # The user-visible payoff: the coach keeps coaching. The O names itself
    # as soon as its second row clears the top edge (frame 74), and every
    # piece named from there on is hinted.
    assert tick("00074").kind is FrameKind.FALLING
    assert tick("00075").falling == "O"
    named = [t for t in replay() if t.falling is not None and t.number > "00059"]
    assert {t.falling for t in named} == {"O", "I"}
    assert all(t.hint is not None and t.hint.piece == t.falling for t in named)
    assert sum(t.hint is not None for t in replay()) == 62


def test_the_preview_corner_is_never_read_as_board_content() -> None:
    # Unchanged by this fix, and load-bearing for it: the cells the NEXT
    # box floats over are unknown, not empty, and never enter the stack.
    for t in replay():
        for r, c in COVERED:
            assert not t.stack_rows[r] >> c & 1


def test_the_confidence_gate_still_rejects_the_same_frames() -> None:
    # Pinned so a vision change shows up here: 32 frames never reach the
    # tracker at all. Every one of them keeps the last good hint on screen.
    rejected = [t for t in replay() if not t.accepted]
    assert len(rejected) == 32
    assert all(t.kind is not FrameKind.UNEXPLAINED for t in rejected)
