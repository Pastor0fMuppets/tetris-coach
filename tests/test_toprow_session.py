"""The live deadlock: a NEXT preview parked on the board's own top row.

Real captured frames from a failing ROAS Stacker session (10 wide x 12
rows, light theme). The session's selected rectangles were
``board_rect = Rect(228, 314, 479, 578)`` and
``next_rect = Rect(613, 317, 92, 92)``, which put the preview over board
cells (0,8), (0,9), (1,8) and (1,9). NOTE this is a DIFFERENT geometry
from ``live_board_*.png``, which come from an earlier session with a
different board rect (see test_overlap_mask.py) — the two sets are not
interchangeable.

``live2_board_00001..00004`` are consecutive ticks from the session start
(the game's start screen); ``live2_board_00300..00800`` are every
hundredth tick of the game that followed. Captures are RGB PNGs and the
capture pipeline hands the engine BGR, so every frame is flipped on load.

What these frames pin: the top row of this game is essentially NEVER
all-empty — the preview owns two of its cells permanently and pieces
spawn visibly in row 0 — so a cap keyed on "any occupied top-row cell"
fired on every frame. Measured on the whole 36-frame capture before the
fix: confidence 0.00 everywhere, no frame ever accepted, and therefore
GridClassifier's background memory (which anchors only from ACCEPTED
frames) never anchored — a deadlock, not a degradation. The live symptoms
were 414 UNEXPLAINED frames, 145 spurious BOARD_RESETs, 7 tracked spawns
and 366 of 555 frames rejected at the gate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from tetris_coach.app import CoachConfig, CoachEngine, compute_overlap_mask
from tetris_coach.capture.screen import Rect
from tetris_coach.vision.grid import GridClassifier, classify_grid

FIXTURES = Path(__file__).parent / "fixtures" / "roas_stacker"
GATE = CoachConfig().min_confidence
ROWS = 12

SESSION_BOARD = Rect(left=228, top=314, width=479, height=578)
SESSION_NEXT = Rect(left=613, top=317, width=92, height=92)
COVERED = frozenset({(0, 8), (0, 9), (1, 8), (1, 9)})

# Board content of the six mid-game frames, read off the images cell by
# cell. '?' marks a cell the preview covers: whatever the capture reads
# there is the NEXT piece, so it is not asserted. The measured confidence
# is the value the cap used to zero.
MID_GAME: dict[str, tuple[float, tuple[str, ...]]] = {
    "00300": (
        0.47,
        (
            "..##....??",
            "..##....??",
            "..........",
            "..........",
            "..........",
            "####......",
            "########..",
            "########..",
            "#########.",
            "#########.",
            "#########.",
            "#########.",
        ),
    ),
    "00400": (
        0.68,
        (
            "........??",
            "........??",
            "..........",
            "####......",
            "####......",
            "########..",
            "########..",
            "########..",
            "#########.",
            "#########.",
            "#########.",
            "#########.",
        ),
    ),
    "00500": (
        0.26,
        (
            "...#....??",
            "...###..??",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "####.#....",
            "#######.##",
        ),
    ),
    "00600": (
        0.32,
        (
            ".....#..??",
            ".....#..??",
            ".....#....",
            "..........",
            "..........",
            "..........",
            "..........",
            "..#.......",
            "#.##.#....",
            "######..##",
            "######..##",
            "#######.##",
        ),
    ),
    "00700": (
        0.18,
        (
            "........??",
            "........??",
            ".........#",
            ".........#",
            "..........",
            "..........",
            "..........",
            "........#.",
            "..#.....#.",
            "#.##.#.###",
            "######.###",
            "######.###",
        ),
    ),
    "00800": (
        0.40,
        (
            "..#.....??",
            "..##....??",
            "..#.......",
            "..........",
            "..........",
            "..........",
            "..........",
            "........##",
            ".....#..##",
            "..#.######",
            "#.########",
            "######.###",
        ),
    ),
}

STARTUP = ("00001", "00002", "00003", "00004")


def load(name: str) -> np.ndarray:
    """One captured frame as the capture pipeline would hand it over (BGR)."""
    return np.asarray(Image.open(FIXTURES / f"live2_board_{name}.png"))[:, :, ::-1]


def expected_grid(lines: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "#" for ch in line] for line in lines], dtype=bool)


def observable_mask(lines: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch != "?" for ch in line] for line in lines], dtype=bool)


def expected_rows(lines: tuple[str, ...]) -> tuple[int, ...]:
    """The board as one bitmask per row; covered cells count as empty."""
    return tuple(sum(1 << c for c, ch in enumerate(line) if ch == "#") for line in lines)


def test_session_rects_cover_the_top_right_corner() -> None:
    assert compute_overlap_mask(SESSION_BOARD, SESSION_NEXT, rows=ROWS) == COVERED


def test_every_mid_game_frame_has_an_occupied_top_row() -> None:
    # The premise of the deadlock, on the real pixels: there is no frame
    # here for the old cap to spare. Two cells are the preview's on every
    # frame; several frames also carry a piece spawning in row 0.
    spawn_contaminated = 0
    for name in MID_GAME:
        occupancy, _ = classify_grid(load(name), rows=ROWS, unobservable_cells=COVERED)
        assert occupancy[0].any(), f"{name}: top row unexpectedly clear"
        observable_occupied = [c for c in range(10) if (0, c) not in COVERED and occupancy[0, c]]
        assert len(observable_occupied) * 2 <= 8, f"{name}: {observable_occupied}"
        if observable_occupied:
            spawn_contaminated += 1
    assert spawn_contaminated >= 3, "fixtures no longer show pieces in row 0"


def test_mid_game_frames_clear_the_gate_and_read_the_board() -> None:
    # The measured payoff: all six frames above the gate, with the board
    # read exactly outside the covered corner.
    for name, (confidence, lines) in MID_GAME.items():
        occupancy, measured = classify_grid(load(name), rows=ROWS, unobservable_cells=COVERED)
        assert measured >= GATE, f"{name}: rejected at {measured:.2f}"
        assert measured == round(confidence, 2) or abs(measured - confidence) < 0.005, (
            f"{name}: confidence drifted to {measured:.3f} from the measured {confidence}"
        )
        seen = observable_mask(lines)
        np.testing.assert_array_equal(
            occupancy[seen], expected_grid(lines)[seen], err_msg=f"{name}: wrong occupancy"
        )


def test_startup_frames_are_the_start_screen_and_anchor_nothing() -> None:
    # Why the memory had no other way in: the session opens on the game's
    # start screen (text and confetti over a white board), which is
    # correctly unreadable. Every later frame was then capped, so nothing
    # ever anchored.
    classifier = GridClassifier(rows=ROWS, unobservable_cells=COVERED)
    for name in STARTUP:
        _occupancy, confidence = classifier.classify(load(name))
        assert confidence < GATE, f"{name}: start screen read at {confidence:.2f}"
    assert classifier.background is None
    assert not classifier.confirmed


def test_classifier_anchors_on_the_real_stream() -> None:
    # The whole point of the change: fed the real session in order, the
    # memory anchors and confirms on the board's own background — a
    # near-white — instead of never forming.
    classifier = GridClassifier(rows=ROWS, unobservable_cells=COVERED)
    for name in STARTUP:
        classifier.classify(load(name))
    accepted = 0
    for name in MID_GAME:
        _occupancy, confidence = classifier.classify(load(name))
        accepted += confidence >= GATE
    assert accepted == len(MID_GAME)
    assert classifier.confirmed
    background = classifier.background
    assert background is not None
    assert float(background.min()) > 230.0, f"anchored off the board: {background}"


class TestEngineOverTheRealSession:
    """CoachEngine, wired exactly as app.run wires it for these rects."""

    @staticmethod
    def _engine_with_spy():  # type: ignore[no-untyped-def]
        covered = compute_overlap_mask(SESSION_BOARD, SESSION_NEXT, rows=ROWS)
        engine = CoachEngine(CoachConfig(rows=ROWS), unobservable_cells=covered)
        accepted: list[np.ndarray] = []
        original = engine.tracker.update

        def spy(occupancy, nxt):  # type: ignore[no-untyped-def]
            accepted.append(occupancy)
            return original(occupancy, nxt)

        engine.tracker.update = spy  # type: ignore[method-assign]
        return engine, accepted

    def test_the_stream_is_accepted_instead_of_rejected(self) -> None:
        # Before: 0 of these frames reached the tracker. The engine gate is
        # the same 0.15; only the vision layer changed.
        engine, accepted = self._engine_with_spy()
        for name in STARTUP:
            engine.process_frame(load(name), None)
        assert accepted == [], "the start screen must not reach the tracker"
        for name in MID_GAME:
            engine.process_frame(load(name), None)
        assert len(accepted) == len(MID_GAME)
        assert engine.classifier.confirmed

    def test_the_preview_never_reaches_the_tracker(self) -> None:
        # The corner reading is still discarded downstream: accepted or
        # not, the NEXT piece is never board content.
        engine, accepted = self._engine_with_spy()
        for name in MID_GAME:
            engine.process_frame(load(name), None)
        for occupancy in accepted:
            for r, c in COVERED:
                assert not occupancy[r, c]

    def test_a_settled_real_board_commits_exactly(self) -> None:
        # End to end on real pixels: a mid-game attach. Four identical
        # frames of the settled board pass the tracker's reset debounce and
        # the committed stack is the board as read off the image.
        engine, _accepted = self._engine_with_spy()
        name = "00400"
        frame = load(name)
        for _ in range(4):
            engine.process_frame(frame, None)
        assert engine.tracker.committed.stack_rows == expected_rows(MID_GAME[name][1])
