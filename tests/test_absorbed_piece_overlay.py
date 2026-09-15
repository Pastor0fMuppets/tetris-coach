"""The coach's own overlay in the absorbed-piece window, and what it cost there.

``tests/fixtures/absorbed_piece/`` is the 81-tick window (00280-00360) the
``_carried_piece`` rule was written from; see its README and
``tests/test_state.py``. This module is about a DIFFERENT thing in the same
frames: seven of them carry this tool's placement hint, painted over the
game by ``overlay.renderer.draw_hint`` and still on screen when the next
frame was captured.

It matters here for two opposite reasons.

It cost a phantom lock. On 296-297 the hint is a T on the floor at (10,3),
(11,2), (11,3), (11,4) while the real T — the pale periwinkle the whole
window is about — is still up at the top of the board. Read as content that
is a T arriving at the floor, and two identical frames is exactly the
tracker's debounce, so it committed a ``PIECE_LOCKED`` and re-solved,
moving the hint off the piece the player was actually holding. With
``vision.grid._own_paint_layer`` taking the paint out, those two frames
read FALLING, the hint holds, and confidence goes 0.372 -> 0.404.

And it is the window that could pay for the rule going wrong. The pale
periwinkle is the nearest thing in any committed fixture to the coach's own
paint — 23.77 uint8 units from the hint composite, three times the
tolerance — and deleting it would hand the solver a board with room in it
that does not exist. It is never touched, and nothing here may ever delete
a cell that does not literally have the hint's own color in it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from tetris_coach.app import CoachConfig, CoachEngine, compute_overlap_mask
from tetris_coach.capture.screen import Rect
from tetris_coach.solver.search import Move
from tetris_coach.vision.state import GameEvent

from .layers import NamedLayer, layers_named, pen_stroked_cells

FIXTURES = Path(__file__).parent / "fixtures" / "absorbed_piece"
ROWS = 12
GATE = CoachConfig().min_confidence

SESSION_BOARD = Rect(left=238, top=317, width=476, height=574)
SESSION_NEXT = Rect(left=620, top=314, width=97, height=96)

# The hint as it stands on the two frames the classifier can name: a T on
# the floor, nothing above it, no badge.
HINT_T = frozenset({(10, 3), (11, 2), (11, 3), (11, 4)})

# The frames whose widget carries a ROTATION BADGE — an opaque cell that
# matches no composite — and is therefore refused whole. The frame goes
# with it: a band the rules cannot resolve is a frame that must not be
# committed, so these report 0.0 and the tracker holds.
BADGE_FRAMES = frozenset({"00298", "00299", "00300", "00301", "00360"})


def frame_numbers() -> list[str]:
    return [p.stem.split("_")[1] for p in sorted(FIXTURES.glob("board_*.png"))]


class Tick:
    def __init__(
        self,
        number: str,
        confidence: float,
        observed: np.ndarray | None,
        events: list[GameEvent],
        hint: Move | None,
        layer: NamedLayer | None,
        falling: str | None,
        stack_rows: tuple[int, ...],
    ) -> None:
        self.number = number
        self.confidence = confidence
        self.observed = observed
        self.events = events
        self.hint = hint
        self.layer = layer
        self.falling = falling
        self.stack_rows = stack_rows

    def committed(self, cell: tuple[int, int]) -> bool:
        row, col = cell
        return bool(self.stack_rows[row] >> col & 1)

    @property
    def accepted(self) -> bool:
        return self.confidence >= GATE

    def row(self, index: int) -> str:
        assert self.observed is not None
        return "".join("#" if self.observed[index, col] else "." for col in range(10))


@lru_cache(maxsize=1)
def replay() -> tuple[Tick, ...]:
    """Feed the whole window through CoachEngine, wired as app.run wires it."""
    covered = compute_overlap_mask(SESSION_BOARD, SESSION_NEXT, rows=ROWS)
    engine = CoachEngine(CoachConfig(rows=ROWS), unobservable_cells=covered)
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
        board = np.asarray(Image.open(FIXTURES / f"board_{number}.png"))[:, :, ::-1]
        preview = FIXTURES / f"next_{number}.png"
        next_image = np.asarray(Image.open(preview))[:, :, ::-1] if preview.exists() else None
        seen.clear()
        handed.clear()
        with layers_named() as named:
            _occupancy, confidence = engine.classifier.classify(board)
            hint = engine.process_frame(board, next_image)
        ticks.append(
            Tick(
                number=number,
                confidence=confidence,
                observed=handed[0] if handed else None,
                events=list(seen),
                hint=hint,
                layer=named[-1] if named else None,
                falling=engine.tracker.committed.falling_piece,
                stack_rows=engine.tracker.committed.stack_rows,
            )
        )
    return tuple(ticks)


def tick(number: str) -> Tick:
    return next(t for t in replay() if t.number == number)


def test_the_window_carries_the_coach_s_overlay_on_seven_frames() -> None:
    painted = {n: pen_stroked_cells(FIXTURES / f"board_{n}.png", ROWS) for n in frame_numbers()}
    carrying = {n: cells for n, cells in painted.items() if cells}
    assert set(carrying) == {"00296", "00297"} | BADGE_FRAMES
    assert carrying["00296"] == HINT_T
    # A widget plus its badge is five cells; the badge rides the cell above
    # the widget's top-left one.
    assert all(len(painted[n]) == 5 for n in BADGE_FRAMES)


def test_nothing_is_deleted_that_does_not_carry_the_hint_s_own_color() -> None:
    """The safety property, checked against the raw PNGs on every frame.

    This is the one that protects the pale periwinkle: the rule may only
    ever remove cells that literally have ``#00e5ff`` painted on them.
    """
    for t in replay():
        named = t.layer.cells if t.layer is not None else frozenset()
        painted = pen_stroked_cells(FIXTURES / f"board_{t.number}.png", ROWS)
        assert named <= painted, f"{t.number}: deleted {sorted(named - painted)}"


def test_the_pale_periwinkle_is_never_deleted() -> None:
    # The piece that sits 23.77 from the hint composite — the nearest miss
    # in any committed fixture — read as content on the frames the window
    # opens with, exactly as it was before the rule.
    assert tick("00280").row(0) == "...###...."
    assert tick("00281").row(0) == "...###...."


def test_the_phantom_lock_on_the_hint_is_gone() -> None:
    # Before: the T-shaped hint on the floor read as content, two identical
    # frames tripped the debounce, PIECE_LOCKED fired and the re-solve
    # moved the hint off the piece the player was holding.
    for number in ("00296", "00297"):
        t = tick(number)
        assert t.layer is not None and t.layer.rule == "own_paint"
        assert t.layer.cells == HINT_T
        assert GameEvent.PIECE_LOCKED not in t.events
        assert t.confidence > 0.40  # was 0.372, with the paint in the split
    assert tick("00295").hint == tick("00296").hint == tick("00297").hint


def test_the_badge_frames_are_refused_whole() -> None:
    # The deliberate limit: nothing can name the opaque badge, so naming
    # the widget under it would leave an unexplainable added cell.
    for number in BADGE_FRAMES:
        assert tick(number).layer is None, f"{number}: the badge frame must be refused"


def test_a_widget_that_cannot_be_named_costs_the_frame_rather_than_the_board() -> None:
    """Refusing the WIDGET is not enough; the frame has to be refused too.

    These frames used to read 0.372 — above the gate — with the hint's
    cells left in as content. That is a T arriving at the floor while the
    real T is still at the top of the board, which the tracker cannot
    explain: four of them in a row tripped a BOARD_RESET, and the resync
    adopted the hint as stack (rows 10-11 gained 775 and 14 over the real
    771). The T was then lost for seven frames and re-spawned at 00308.

    The cells are known to be our own paint; what is unknown is what to DO
    with them, and a frame with an unresolvable band in it is exactly the
    frame that must not be committed. So it reports 0.0, the tracker holds
    and the piece stays in flight across the whole run.
    """
    for number in BADGE_FRAMES:
        assert tick(number).confidence == 0.0, f"{number}: read at {tick(number).confidence:.3f}"
    for number in ("00298", "00299", "00300", "00301"):
        assert tick(number).falling == "T", f"{number}: the T was dropped"
    events = [e for t in replay() for e in t.events]
    assert events.count(GameEvent.BOARD_RESET) == 1  # was 3
    assert events.count(GameEvent.PIECE_SPAWNED) == 2  # was 3: one was a re-pickup
    # The phantom itself: the reset used to adopt the widget, leaving rows
    # 8 and 9 holding 2 and 14 over a stack that is only two rows deep.
    for t in replay()[: frame_numbers().index("00310")]:
        assert t.stack_rows[8] == 0 and t.stack_rows[9] == 0, f"{t.number}: phantom stack rows"
