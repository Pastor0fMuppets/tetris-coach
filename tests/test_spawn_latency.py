"""What the coach shows while a piece is still entering, measured.

``tests/fixtures/spawn_latency/`` holds 161 CONSECUTIVE ticks (00080-00240)
of the session the user was watching when he said: "The hint doesn't show
until the piece has fully dropped both rows onto the board. Ideally we
would store the piece from the next piece viewer to speed things up so it
can show when the new piece is only partially visible as it is falling."
See the fixtures' README for the geometry and for the cost measured over
the whole 705-frame session it came from.

The wait is deliberate, not a bug: this game parks a spawn at the top edge
of the board region, where 1-3 of its cells are on the grid and the rest
is above the capture, and two cells side by side fit an O, an S, a Z, a J
and an L alike. ``explain_grid`` holds rather than guess, the frame reads
OCCLUDED, and nothing is on screen until the piece's second row descends.

This module replays the window through ``CoachEngine`` and MEASURES that
wait, per entering piece, so the accelerator can be judged against it
rather than against a description of it. The three episodes here, and
what each has to work with:

    frames 00086-00102  the window OPENS on a piece already at the top
                        edge. No preview CHANGE has been seen (the box
                        holds an O from the first frame, and None -> X
                        says nothing about what was dealt), so there is
                        no evidence and the rule holds: 17 frames.
    frames 00161-00174  the preview flips O -> I on 00158, which is the
                        game saying the O is what it just dealt — the
                        very fragment sitting at the top edge. THIS is
                        the one the preview can name early.
    frames 00217-00232  the flip here is I -> O and it is a deal LATE:
                        the T that was dealt sat in the preview box from
                        00198 to 00215 unread (it is drawn in a pale blue
                        that scores 0.319 against the box background,
                        under the 0.35 uniformity floor), so the name the
                        flip carries is the piece BEFORE it. Naming the
                        fragment from it would be the "confused two
                        pieces" failure; it must stay a 16-frame wait.
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
from tetris_coach.vision.pieces_vision import FrameKind, identify_next
from tetris_coach.vision.state import GameEvent

FIXTURES = Path(__file__).parent / "fixtures" / "spawn_latency"
ROWS = 12
GATE = CoachConfig().min_confidence

SESSION_BOARD = Rect(left=204, top=311, width=480, height=577)
SESSION_NEXT = Rect(left=587, top=309, width=99, height=102)
COVERED = frozenset({(0, 8), (0, 9), (1, 8), (1, 9)})

# The frames the confidence gate never lets through: the window opens
# mid-animation, and 00127-00157 is the game's end-of-round wipe.
REJECTED = [*(f"000{n}" for n in range(80, 86)), *(f"00{n}" for n in range(127, 158))]


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
        observed: np.ndarray,
        stack_rows: tuple[int, ...],
        falling: str | None,
        next_piece: str | None,
        read_next: str | None,
        hint: Move | None,
    ) -> None:
        self.number = number
        self.confidence = confidence
        self.kind = kind
        self.events = events
        self.observed = observed  # the occupancy the tracker was handed
        self.stack_rows = stack_rows
        self.falling = falling
        self.next_piece = next_piece  # committed
        self.read_next = read_next  # what the preview reader said this frame
        self.hint = hint

    @property
    def accepted(self) -> bool:
        return self.confidence >= GATE

    @property
    def entering_cells(self) -> list[int]:
        """Columns of row 0 the committed stack does NOT hold — a piece in."""
        return [c for c in range(10) if self.observed[0, c] and not self.stack_rows[0] >> c & 1]


@lru_cache(maxsize=1)
def replay() -> tuple[Tick, ...]:
    """Feed the whole window through CoachEngine, wired as app.run wires it."""
    covered = compute_overlap_mask(SESSION_BOARD, SESSION_NEXT, rows=ROWS)
    engine = CoachEngine(CoachConfig(rows=ROWS), unobservable_cells=covered)
    update = engine.tracker.update
    seen: list[GameEvent] = []
    handed: list[np.ndarray] = []
    read: list[str | None] = []

    def spy(occupancy, next_piece):  # type: ignore[no-untyped-def]
        handed.append(np.array(occupancy, copy=True))
        read.append(next_piece)
        events = update(occupancy, next_piece)
        seen.extend(events)
        return events

    engine.tracker.update = spy  # type: ignore[method-assign]
    ticks: list[Tick] = []
    for number in frame_numbers():
        board = load(f"board_{number}.png")
        preview = FIXTURES / f"next_{number}.png"
        seen.clear()
        handed.clear()
        read.clear()
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
                observed=handed[0] if handed else np.zeros((ROWS, 10), dtype=bool),
                stack_rows=committed.stack_rows,
                falling=committed.falling_piece,
                next_piece=committed.next_piece,
                read_next=read[0] if read else None,
                hint=hint,
            )
        )
    return tuple(ticks)


def tick(number: str) -> Tick:
    return next(t for t in replay() if t.number == number)


def entering_episodes() -> list[tuple[str, str | None, int | None]]:
    """``(first sighting, first hint, frames between)`` per entering piece.

    A SIGHTING is the first accepted frame showing cells at row 0 that the
    committed stack does not hold and that nothing on screen names yet — a
    piece entering, with no hint for it. The episode closes on the first
    accepted frame carrying a hint for a named falling piece. This is the
    user's own measure: what he sees is the gap between the piece
    appearing and something appearing on it.
    """
    episodes: list[tuple[str, str | None, int | None]] = []
    opened: tuple[int, str] | None = None
    for index, t in enumerate(replay()):
        if not t.accepted:
            continue
        named = t.falling is not None and t.hint is not None
        if named:
            if opened is not None:
                episodes.append((opened[1], t.number, index - opened[0]))
                opened = None
        elif t.entering_cells and opened is None:
            opened = (index, t.number)
        elif opened is not None and not t.entering_cells:
            episodes.append((opened[1], None, None))  # gone without ever being named
            opened = None
    if opened is not None:
        episodes.append((opened[1], None, None))
    return episodes


def test_the_window_is_the_readme_geometry() -> None:
    assert compute_overlap_mask(SESSION_BOARD, SESSION_NEXT, rows=ROWS) == COVERED
    assert len(frame_numbers()) == 161
    assert frame_numbers()[0] == "00080"
    assert frame_numbers()[-1] == "00240"


def test_the_window_replays_without_an_unexplainable_surprise() -> None:
    # The baseline this measurement stands on. Two resets, both real: the
    # window opens mid-game on a board no memory explains, and the game
    # wipes the field at 00127-00157 and deals a new one.
    events = Counter(e for t in replay() for e in t.events)
    assert events[GameEvent.BOARD_RESET] == 2
    assert events[GameEvent.PIECE_LOCKED] == 2
    assert events[GameEvent.PIECE_SPAWNED] == 4
    assert [t.number for t in replay() if not t.accepted] == REJECTED
    kinds = Counter(t.kind for t in replay() if t.accepted and t.kind is not None)
    assert kinds[FrameKind.UNEXPLAINED] == 8  # the two resets' confirmation runs


def test_a_piece_entering_from_above_is_held_nameless() -> None:
    # The wait, per piece, measured. Each of these is a piece the user can
    # see on his screen with nothing drawn on it.
    assert entering_episodes() == [
        ("00086", "00103", 17),
        ("00161", "00175", 14),
        ("00217", "00233", 16),
    ]
    # ...and what it looks like while it runs: a coherent frame that names
    # nothing, held rather than guessed, for 12 frames of the second one.
    waiting = [tick(f"00{n}") for n in range(162, 174)]
    assert all(t.kind is FrameKind.OCCLUDED for t in waiting)
    assert all(t.falling is None and t.hint is None for t in waiting)
    # Two cells, side by side, sliding across the top edge as the player
    # moves a piece he has been given nothing to move.
    assert [t.entering_cells for t in waiting] == [[4, 5]] * 8 + [[5, 6]] * 4


def test_the_wait_is_a_third_of_the_frames_the_gate_accepts() -> None:
    # The aggregate the README reports over the whole session (164 of 705
    # OCCLUDED, 23%), on the committed window: every one of these is a
    # frame where the coach knows a piece is there and cannot name it.
    accepted = [t for t in replay() if t.accepted]
    kinds = Counter(t.kind for t in accepted if t.kind is not None)
    assert len(accepted) == 124
    assert kinds[FrameKind.OCCLUDED] == 43
    assert sum(1 for t in replay() if t.hint is not None) == 108


def test_the_preview_is_read_on_most_frames_and_never_on_the_pale_T() -> None:
    # The signal's availability, and its measured ceiling. The tracker is
    # handed a name on 101 of the 124 frames that reach it; the 18 it is
    # not (00198-00215) are the frames the T sits in the box, drawn in a
    # pale blue that scores 0.319 against the box's background — under the
    # 0.35 uniformity floor the board reader and the preview reader share,
    # so the mask keeps nothing and there is no shape to read. That gap is
    # why the third episode's flip arrives a deal late.
    assert sum(1 for t in replay() if t.read_next is not None) == 101
    assert all(identify_next(load(f"next_00{n}.png")) is None for n in range(198, 216))
    assert identify_next(load("next_00197.png")) == "I"
    assert identify_next(load("next_00216.png")) == "O"


def test_the_flip_that_names_the_second_piece_is_on_the_frame_it_enters() -> None:
    # The evidence the accelerator has to work with, in this window. The
    # game wipes the field, deals an O, and the preview flips O -> I on
    # 00158 — the same frame the O's first two cells appear at the top
    # edge. The shape rule agrees on 00174, when its second row descends.
    assert tick("00157").next_piece == "O"  # committed before the wipe
    assert [tick(f"00{n}").entering_cells for n in range(158, 162)] == [[4, 5]] * 4
    assert tick("00174").kind is FrameKind.FALLING
    assert tick("00175").falling == "O"
