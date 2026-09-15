"""Replay of 61 CONSECUTIVE frames where a real piece read as background.

``tests/fixtures/pale_piece/`` holds ticks 00640-00700 of a ROAS Stacker
session (10 x 12, light theme) whose NEXT preview floats over board cells
(0,8), (0,9), (1,8) and (1,9). The frames are RGB PNGs and the capture
pipeline hands the engine BGR, so each is flipped on load. The rects are
this session's own — see the README, and do not borrow another window's.

A pale periwinkle T — RGB (206, 224, 251) against a (251, 252, 252)
board — drifts down the middle of the board for the whole window. It is
53 uint8 units from the background and scores 0.346, four thousandths
under :data:`MIN_SPREAD`, while the blue stack beside it scores 0.815.
Otsu's split is dominated by that blue, lands above 0.346, and the T
falls into the EMPTY class: not misread, ABSENT. Measured before:

    frames with a hint 0 of 61     falling piece tracked on 0
    observed row 11 "#........#"   (two settled periwinkle cells short)
    confidence 0.576               (confidently wrong, not unsure)

The two settled cells are the same failure standing still: (11,5) and
(11,8) are periwinkle too, so the board handed to the solver had two
holes in its floor that the game does not have.

No threshold can fix this. A ghost — the landing preview this module
deletes on purpose — scores 0.320 on this same theme, four thousandths
BELOW the piece, so a cut low enough to admit the T admits the ghost
with it. Only structure can tell them apart, which is what
``_ghost_layer`` is for, and the fix is to let the band reach it: a cell
between the background cluster and the floor is a CANDIDATE, the
structural rules name what they can, and what is left is content.

AFTER (same frames, same gate):

    frames with a hint 45 of 61    the T tracked as FALLING from 00656
    observed row 11 "#....#..##"   the real floor
    confidence 0.206               honest: the band against the ground

The 16 frames at the head are the window opening mid-session on a board
the tracker has never seen, and 00687 on is not the game at all: a
different application is on screen — an event-schedule page, once wrongly
recorded here as an end-of-round panel — correctly refused at 0.009, hint
held.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from tetris_coach.app import CoachConfig, CoachEngine, compute_overlap_mask
from tetris_coach.capture.screen import Rect
from tetris_coach.solver.search import Move
from tetris_coach.vision.grid import MIN_SPREAD, _cell_colors, _distance_scores
from tetris_coach.vision.pieces_vision import FrameKind
from tetris_coach.vision.state import GameEvent

FIXTURES = Path(__file__).parent / "fixtures" / "pale_piece"
ROWS = 12
GATE = CoachConfig().min_confidence

SESSION_BOARD = Rect(left=276, top=314, width=477, height=578)
SESSION_NEXT = Rect(left=659, top=317, width=98, height=95)
COVERED = frozenset({(0, 8), (0, 9), (1, 8), (1, 9)})

# The piece color that must be read as content, and the board it is drawn
# on, both read straight off the PNGs. 53 uint8 units apart — a real
# color difference, but a fifth of the blue stack's 294.
PERIWINKLE = (206, 224, 251)
BOARD_GROUND = (251, 252, 252)

# The T's three resting positions across the window: it is dragged down
# one row at a time, sitting at each for 15-16 ticks.
T_AT = {
    "high": frozenset({(1, 4), (2, 3), (2, 4), (2, 5)}),
    "middle": frozenset({(2, 4), (3, 3), (3, 4), (3, 5)}),
    "low": frozenset({(3, 4), (4, 3), (4, 4), (4, 5)}),
}

# Two cells of the settled stack are the same periwinkle, on every frame.
SETTLED_PERIWINKLE = frozenset({(11, 5), (11, 8)})
FLOOR_ROW = "#....#..##"

# The frames another application covers the capture region on: an
# event-schedule page, not the game. (Recorded here as the game's own
# end-of-round panel until the pixels of 00690 were looked at.)
NOT_THE_GAME = tuple(f"00{n}" for n in range(687, 701))


def load(name: str) -> np.ndarray:
    """One captured frame as the capture pipeline would hand it over (BGR)."""
    return np.asarray(Image.open(FIXTURES / name))[:, :, ::-1]


def frame_numbers() -> list[str]:
    return [p.stem.split("_")[1] for p in sorted(FIXTURES.glob("board_*.png"))]


def cells_colored(path: Path, rgb: tuple[int, int, int]) -> frozenset[tuple[int, int]]:
    """Cells of ``path`` whose most common pixel is ``rgb``.

    Ground truth from the raw image, independent of anything in
    ``vision.grid``: these pieces are drawn flat, so a cell's modal pixel
    IS its color.
    """
    image = np.asarray(Image.open(path).convert("RGB"))
    height, width, _ = image.shape
    found = set()
    for row in range(ROWS):
        for col in range(10):
            y0, y1 = int(row * height / ROWS), int((row + 1) * height / ROWS)
            x0, x1 = int(col * width / 10), int((col + 1) * width / 10)
            patch = image[y0:y1, x0:x1].reshape(-1, 3)
            values, counts = np.unique(patch, axis=0, return_counts=True)
            if tuple(int(v) for v in values[counts.argmax()]) == rgb:
                found.add((row, col))
    return frozenset(found)


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
    ) -> None:
        self.number = number
        self.confidence = confidence
        self.kind = kind
        self.events = events
        self.observed = observed  # the occupancy the tracker was handed
        self.stack_rows = stack_rows
        self.falling = falling
        self.hint = hint

    @property
    def accepted(self) -> bool:
        return self.confidence >= GATE

    def row(self, index: int) -> str:
        assert self.observed is not None
        return "".join("#" if self.observed[index, col] else "." for col in range(10))

    def occupied(self, cell: tuple[int, int]) -> bool:
        assert self.observed is not None
        return bool(self.observed[cell])

    def committed(self, cell: tuple[int, int]) -> bool:
        row, col = cell
        return bool(self.stack_rows[row] >> col & 1)


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
        board = load(f"board_{number}.png")
        preview = FIXTURES / f"next_{number}.png"
        seen.clear()
        handed.clear()
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
                observed=handed[0] if handed else None,
                stack_rows=committed.stack_rows,
                falling=committed.falling_piece,
                hint=hint,
            )
        )
    return tuple(ticks)


def test_the_session_geometry_is_the_readme_geometry() -> None:
    assert compute_overlap_mask(SESSION_BOARD, SESSION_NEXT, rows=ROWS) == COVERED
    assert len(frame_numbers()) == 61
    assert frame_numbers()[0] == "00640"
    assert frame_numbers()[-1] == "00700"


def test_the_piece_is_really_there_and_really_is_pale() -> None:
    """The premise, off the raw pixels rather than through the classifier.

    A whole T of periwinkle plus the two settled cells, on every frame of
    the window that is the game at all — and 53 uint8 units
    from the board it sits on, where the blue stack beside it is 294.
    """
    for number in frame_numbers():
        if number in NOT_THE_GAME:
            continue
        painted = cells_colored(FIXTURES / f"board_{number}.png", PERIWINKLE)
        assert painted - SETTLED_PERIWINKLE in set(T_AT.values()), f"{number}: {sorted(painted)}"
        assert SETTLED_PERIWINKLE <= painted, f"{number}: settled cells missing"
    near = float(np.linalg.norm(np.array(PERIWINKLE) - np.array(BOARD_GROUND)))
    far = float(np.linalg.norm(np.array((45, 46, 215)) - np.array(BOARD_GROUND)))
    assert 52.0 < near < 54.0
    assert 293.0 < far < 295.0


def test_no_threshold_could_have_seen_it() -> None:
    """Why this needed a structural fix and not a lower floor.

    The T's cells score 0.346 against the remembered background — under
    MIN_SPREAD, and inside the band a landing preview also lands in
    (0.320 on this theme, measured in ``_ghost_layer``'s own fixtures).
    Four thousandths apart: any cut that admits the piece admits the
    ghost, so the score cannot be what decides it.
    """
    covered = compute_overlap_mask(SESSION_BOARD, SESSION_NEXT, rows=ROWS)
    engine = CoachEngine(CoachConfig(rows=ROWS), unobservable_cells=covered)
    image = load("board_00660.png")
    engine.classifier.classify(image)  # anchor the background memory
    background = engine.classifier.background
    assert background is not None
    scores = _distance_scores(_cell_colors(image, ROWS, 10, margin=0.25), background)
    for cell in T_AT["middle"] | SETTLED_PERIWINKLE:
        assert abs(float(scores[cell]) - 0.346) < 0.005, f"{cell}: {float(scores[cell]):.3f}"
        assert float(scores[cell]) < MIN_SPREAD
    assert abs(float(scores[11, 0]) - 0.815) < 0.005, "the blue stack moved"


def test_every_accepted_frame_shows_the_whole_piece() -> None:
    """The bug itself: the T is in the occupancy the tracker is handed.

    Before, not one of these cells was ever occupied on any frame.
    """
    accepted = [t for t in replay() if t.accepted]
    assert len(accepted) == 47
    for t in accepted:
        here = [cells for cells in T_AT.values() if all(t.occupied(c) for c in cells)]
        assert len(here) == 1, f"{t.number}: the T is not in the reading"


def test_the_settled_periwinkle_reaches_the_floor_row() -> None:
    """The quiet half of the same bug: two stack cells the solver never had."""
    for t in replay():
        if not t.accepted:
            continue
        assert t.row(11) == FLOOR_ROW, f"{t.number}: {t.row(11)}"
    for cell in SETTLED_PERIWINKLE:
        assert replay()[-1].committed(cell), f"{cell} never reached the committed stack"


def test_the_piece_is_tracked_as_one_falling_t() -> None:
    """One piece, one episode, picked up once and never lost."""
    tracked = [t for t in replay() if t.falling is not None]
    assert [t.number for t in tracked] == [f"00{n}" for n in range(656, 701)]
    assert {t.falling for t in tracked} == {"T"}
    spawns = [e for t in replay() for e in t.events if e is GameEvent.PIECE_SPAWNED]
    assert len(spawns) == 1


def test_one_stable_hint_for_the_whole_descent() -> None:
    """The user-visible outcome: an overlay, and one that does not wander.

    Before: no hint on any of the 61 frames, the tail of a 90-frame
    (~6 s) dropout. The hint holds through the fourteen frames another
    application covers the region on, because a refused frame holds the
    last one.
    """
    hinted = [t for t in replay() if t.hint is not None]
    assert [t.number for t in hinted] == [f"00{n}" for n in range(656, 701)]
    targets = {(t.hint.piece, t.hint.rotation.index, frozenset(t.hint.cells)) for t in hinted}
    assert len(targets) == 1, f"the hint moved for one piece: {targets}"
    assert next(iter(targets))[0] == "T"


def test_nothing_phantom_locks_and_the_board_resets_only_on_the_attach() -> None:
    events = [e for t in replay() for e in t.events]
    assert GameEvent.PIECE_LOCKED not in events
    assert events.count(GameEvent.BOARD_RESET) == 1
    resets = [t.number for t in replay() for e in t.events if e is GameEvent.BOARD_RESET]
    assert resets == ["00643"], "the only reset is the window opening mid-session"
    unexplained = [t.number for t in replay() if t.accepted and t.kind is FrameKind.UNEXPLAINED]
    assert unexplained == [f"0064{n}" for n in range(4)]


def test_the_end_of_round_panel_is_refused_rather_than_read() -> None:
    """The window's last 14 frames are not a board and must not read as one."""
    for t in replay():
        if t.number in NOT_THE_GAME:
            assert not t.accepted, f"{t.number}: panel accepted at {t.confidence:.3f}"
            assert t.hint is not None, f"{t.number}: the held hint was dropped"


def test_the_preview_corner_is_never_read_as_board_content() -> None:
    # Load-bearing here as everywhere: (0,8) scores 0.178 — inside the
    # band — because it is the NEXT box's pixels, not the board's.
    for t in replay():
        for cell in COVERED:
            assert not t.committed(cell)
