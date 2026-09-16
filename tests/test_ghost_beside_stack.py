"""Replay of 48 CONSECUTIVE frames where the coach read its own overlay back.

``tests/fixtures/ghost_beside_stack/`` holds ticks 00138-00185 of a real
ROAS Stacker session (10 x 12, light theme) with its NEXT preview floating
over board cells (0,8), (0,9), (1,8) and (1,9). The frames are RGB PNGs and
the capture pipeline hands the engine BGR, so each is flipped on load. The
rects are the ghost_session/absorbed_piece pair's neighbours but NOT equal
to them, and different again from ``tests/fixtures/live_session`` — see the
README.

Consecutive is the whole point: the tracker explains every frame as a diff
against the previous committed one, so only a contiguous run replays what
the live session actually did.

The user's report was "the overlay moves around a lot for the same piece
and turn". Measured on these frames BEFORE the paint rule:

    UNEXPLAINED 22 of 48   BOARD_RESET 3   PIECE_LOCKED 2 (both phantom)
    frames with a hint 18  hint targets for the one I piece: 2, plus blank

and it cycled every ~11 frames: FALLING -> LOCKED -> LOCKED -> UNEXPLAINED
x8 -> BOARD_RESET -> QUIET -> FALLING.

The engine was chasing its own tail. The I hangs near the top of the board
the whole window and never locks; what arrives at the bottom and then
vanishes is THIS TOOL'S HINT, painted over the game by
``overlay.renderer.draw_hint`` and still on screen when the next frame was
captured. Read as board content it is a tetromino that appears at the
floor, so the tracker committed a LOCK and re-solved; the re-solve moved
the hint to different columns, so next frame the "locked" cells had moved,
which is unexplainable; four of those reset the board, the reset cleared
the hint, the blank frame read clean, and the cycle started over. Row 11
read ``........##`` and ``....######`` on alternate ticks for exactly that
reason.

``vision.grid._ghost_layer`` could not save it. Its test 4 refuses any
candidate with a solid cell beside it, and this hint lands at cols 4-7 with
the stack at cols 8-9 — so the common case was thrown away to protect the
rare one (the pale periwinkle of ``tests/fixtures/roas_stacker``). The
column-matching signal the fixture README points at does not survive the
other windows either: on ``ghost_session`` 61-64 the same widget sits at
cols 2-3 while the piece it belongs to is at cols 4-5, because a hint marks
where the SOLVER wants the piece, not where the player is holding it.

AFTER (same frames, same gate), with ``_own_paint_layer`` recognizing the
paint by its color:

    UNEXPLAINED 4 of 48    BOARD_RESET 1   PIECE_LOCKED 0
    frames with a hint 31  hint targets for the one I piece: 1

The four unexplainable frames and the one reset are the window opening
mid-session on a board the tracker has never seen (it starts with an empty
stack), which is the ordinary mid-game attach; from frame 147 on nothing is
unexplainable. The I is tracked from 155 to the end of the window as the
player drags it down and left, and the hint stays on ``(11, 4)-(11, 7)``
for all 31 of those frames.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from tetris_coach.app import CoachConfig, CoachEngine, compute_overlap_mask
from tetris_coach.capture.screen import Rect
from tetris_coach.solver.search import Move
from tetris_coach.truth.windows import CAPTURED_FILL_OPACITY
from tetris_coach.vision.pieces_vision import FrameKind
from tetris_coach.vision.state import GameEvent

from .layers import NamedLayer, layers_named, pen_stroked_cells

FIXTURES = Path(__file__).parent / "fixtures" / "ghost_beside_stack"
ROWS = 12
GATE = CoachConfig().min_confidence

SESSION_BOARD = Rect(left=276, top=314, width=477, height=578)
SESSION_NEXT = Rect(left=659, top=317, width=98, height=95)
COVERED = frozenset({(0, 8), (0, 9), (1, 8), (1, 9)})

# The two places the coach painted its hint during this window, read off
# the PNGs. Both are a horizontal I on the floor; the solver moved it from
# cols 4-7 to cols 0-3 and back every few ticks precisely because its own
# paint kept changing the board it was solving.
HINT_RIGHT = frozenset({(11, 4), (11, 5), (11, 6), (11, 7)})
HINT_LEFT = frozenset({(11, 0), (11, 1), (11, 2), (11, 3)})

# The real board under all of it: two settled cells in the bottom-right
# corner, unchanged on every frame of the window. This row is the one the
# bug was visible in — it read "....######" whenever the hint was at cols
# 4-7 and "........##" whenever it was not.
FLOOR_ROW = "........##"


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
        observed: np.ndarray | None,
        stack_rows: tuple[int, ...],
        falling: str | None,
        hint: Move | None,
        layer: NamedLayer | None,
    ) -> None:
        self.number = number
        self.confidence = confidence
        self.kind = kind
        self.events = events
        self.observed = observed  # the occupancy the tracker was handed
        self.stack_rows = stack_rows
        self.falling = falling
        self.hint = hint
        self.layer = layer  # the third level this frame had removed, if any

    @property
    def accepted(self) -> bool:
        return self.confidence >= GATE

    def row(self, index: int) -> str:
        assert self.observed is not None
        return "".join("#" if self.observed[index, col] else "." for col in range(10))

    def committed(self, cell: tuple[int, int]) -> bool:
        row, col = cell
        return bool(self.stack_rows[row] >> col & 1)


@lru_cache(maxsize=1)
def replay() -> tuple[Tick, ...]:
    """Feed the whole window through CoachEngine, wired as app.run wires it."""
    covered = compute_overlap_mask(SESSION_BOARD, SESSION_NEXT, rows=ROWS)
    engine = CoachEngine(
        CoachConfig(rows=ROWS, tracker="shape", hint_fill_opacity=CAPTURED_FILL_OPACITY),
        unobservable_cells=covered,
    )
    update = engine.tracker.update
    seen: list[GameEvent] = []
    handed: list[np.ndarray] = []

    def spy(occupancy, next_piece):  # type: ignore[no-untyped-def]
        handed.append(np.array(occupancy, copy=True))
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
        # The classifier is stateful, so this must read the same frame the
        # engine is about to digest — classify() is a pure function of the
        # image plus the memory, and the engine re-runs it identically.
        with layers_named() as named:
            _occupancy, confidence = engine.classifier.classify(board)
            hint = engine.process_frame(board, load(preview.name) if preview.exists() else None)
        committed = engine.tracker.committed
        ticks.append(
            Tick(
                number=number,
                confidence=confidence,
                kind=engine.tracker.last_kind,
                events=list(seen),
                observed=handed[0] if handed else None,
                stack_rows=committed.stack_rows,
                falling=committed.falling_piece,
                hint=hint,
                layer=named[-1] if named else None,
            )
        )
    return tuple(ticks)


def test_the_session_geometry_is_the_readme_geometry() -> None:
    assert compute_overlap_mask(SESSION_BOARD, SESSION_NEXT, rows=ROWS) == COVERED
    assert len(frame_numbers()) == 48
    assert frame_numbers()[0] == "00138"
    assert frame_numbers()[-1] == "00185"


def test_the_window_really_does_contain_this_tool_s_own_overlay() -> None:
    """The premise, off the raw pixels: 24 of these frames have a hint on them.

    ``draw_hint`` strokes each cell outline with ``HintStyle.color`` at
    full opacity, so a frame carrying the overlay carries that exact RGB
    value. It alternates between two positions and is absent entirely on
    the frames in between — which is the flicker, seen at its source.
    """
    painted = {n: pen_stroked_cells(FIXTURES / f"board_{n}.png", ROWS) for n in frame_numbers()}
    assert set(painted.values()) == {frozenset(), HINT_RIGHT, HINT_LEFT}
    assert sum(1 for cells in painted.values() if cells) == 24
    assert painted["00143"] == frozenset()
    assert painted["00145"] == HINT_RIGHT
    assert painted["00147"] == HINT_LEFT


def test_the_overlay_is_named_on_exactly_the_frames_that_carry_it() -> None:
    # Nothing structural is asked of these cells: they are recognized by
    # their color, so a hint beside the stack is named exactly as readily
    # as one in open space. That is the whole fix — _ghost_layer refused
    # HINT_RIGHT (solid stack at cols 8-9 to its right) while naming
    # HINT_LEFT, which is what made the reading alternate.
    for t in replay():
        painted = pen_stroked_cells(FIXTURES / f"board_{t.number}.png", ROWS)
        named = t.layer.cells if t.layer is not None else frozenset()
        assert named == painted, f"{t.number}: named {sorted(named)} for {sorted(painted)}"
        if t.layer is not None:
            assert t.layer.rule == "own_paint", f"{t.number}: named by {t.layer.rule}"


def test_the_floor_row_no_longer_flickers() -> None:
    """The symptom, gone. Row 11 is the same on every frame the gate accepts.

    Before, it alternated between ``....######`` (the hint read as four
    locked cells) and ``........##`` (the real board) every two or three
    ticks. The two settled cells at cols 8-9 are all that is really there.
    """
    accepted = [t for t in replay() if t.accepted]
    assert len(accepted) == 43
    assert {t.row(11) for t in accepted} == {FLOOR_ROW}


def test_the_overlay_never_reaches_the_committed_stack() -> None:
    # A phantom lock is how it used to get in: two identical frames with
    # the hint read as content is exactly the tracker's debounce.
    for t in replay():
        for cell in HINT_RIGHT | HINT_LEFT:
            assert not t.committed(cell), f"{t.number}: {cell} committed"


def test_nothing_locks_and_the_board_settles_after_the_attach() -> None:
    kinds = [t.kind for t in replay() if t.accepted]
    assert kinds.count(FrameKind.UNEXPLAINED) == 4  # was 22
    events = [e for t in replay() for e in t.events]
    assert events.count(GameEvent.BOARD_RESET) == 1  # was 3
    assert GameEvent.PIECE_LOCKED not in events  # was 2, both phantom
    # The unexplainable frames are the attach at the head of the window,
    # not the oscillation: nothing after the resync is unexplainable.
    unexplained = [t.number for t in replay() if t.kind is FrameKind.UNEXPLAINED]
    assert unexplained == ["00143", "00144", "00145", "00146"]


def test_the_i_is_tracked_continuously_to_the_end_of_the_window() -> None:
    """One piece, one episode. The I never leaves the air in these frames.

    The player drags it down and left — row 1 cols 4-7, row 1 cols 3-6,
    row 2 cols 1-4, row 3 cols 0-3 — and it is still falling on the last
    frame. Before, the tracker lost it every ~11 frames to a phantom lock
    and a reset that adopted the board without it.
    """
    tracked = [t for t in replay() if t.falling is not None]
    assert [t.number for t in tracked] == [f"00{n}" for n in range(155, 186)]
    assert {t.falling for t in tracked} == {"I"}
    spawns = [e for t in replay() for e in t.events if e is GameEvent.PIECE_SPAWNED]
    assert len(spawns) == 1, "the I is picked up once, not once per cycle"


def test_one_stable_hint_target_for_the_whole_piece() -> None:
    """The user's actual complaint: the overlay moving for the same piece.

    Before, the hint for this single I alternated between leftmost column
    4 and leftmost column 0 — a jump across the whole board — and went
    blank in between, 5 changes over the window. Now there is one target
    and it is on screen from the moment the piece is picked up.
    """
    hinted = [t for t in replay() if t.hint is not None]
    assert [t.number for t in hinted] == [f"00{n}" for n in range(155, 186)]
    targets = {(t.hint.piece, t.hint.rotation.index, frozenset(t.hint.cells)) for t in hinted}
    assert targets == {("I", 0, HINT_RIGHT)}


def test_the_preview_corner_is_never_read_as_board_content() -> None:
    # Unchanged by this fix, and load-bearing for it: the cells the NEXT
    # box floats over are unknown, not empty, and never enter the stack.
    for t in replay():
        for cell in COVERED:
            assert not t.committed(cell)
