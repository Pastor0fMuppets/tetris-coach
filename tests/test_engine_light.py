"""End-to-end engine sequences on a LIGHT theme (white board, colored pieces).

The dark-theme engine scenarios live in test_headless.py; these drive the
same CoachEngine through rendered paper-white frames — the theme family of
the real game behind tests/fixtures/roas_stacker — plus the two real
fixtures as a permanent non-gameplay stream. Before the background-distance
classifier, every one of these frames read as (nearly) all-occupied at
confidence ~0 and the whole game was invisible to the engine.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from tetris_coach.app import CoachConfig, CoachEngine
from tetris_coach.vision.state import GameEvent

from .boards import EMPTY, grid_of, merge, piece_cells, rows_of
from .synthetic import STYLES, render_board

FIXTURES = Path(__file__).parent / "fixtures"
STYLE = next(s for s in STYLES if s.name == "paper-white")
CELL = 20


def _engine_with_spy():  # type: ignore[no-untyped-def]
    engine = CoachEngine(CoachConfig(tracker="shape"))
    events_log: list[GameEvent] = []
    original = engine.tracker.update

    def spy(occupancy, nxt):  # type: ignore[no-untyped-def]
        events = original(occupancy, nxt)
        events_log.extend(events)
        return events

    engine.tracker.update = spy  # type: ignore[method-assign]
    return engine, events_log


def _frame(rows: tuple[int, ...]) -> np.ndarray:
    return render_board(grid_of(rows), STYLE, cell_size=CELL)


def test_light_theme_game_sequence() -> None:
    engine, events_log = _engine_with_spy()

    # Two empty white frames: an empty LIGHT board classifies empty with
    # usable confidence (the generalized uniform rule) — nothing commits,
    # no hint, no reset.
    for _ in range(2):
        assert engine.process_frame(_frame(EMPTY), None) is None
    assert engine.tracker.committed.stack_rows == EMPTY
    assert events_log == []

    # An S spawns and is committed on the second frame; hint appears.
    spawn = rows_of(piece_cells("S", 0, 0, 3))
    engine.process_frame(_frame(spawn), None)
    hint = engine.process_frame(_frame(spawn), None)
    assert hint is not None
    assert hint.piece == "S"
    assert events_log == [GameEvent.PIECE_SPAWNED]

    # Descent and lock-delay rest: identity is unchanged, hint held.
    for rows in (
        rows_of(piece_cells("S", 0, 8, 3)),
        rows_of(piece_cells("S", 0, 18, 3)),
        rows_of(piece_cells("S", 0, 18, 3)),
    ):
        assert engine.process_frame(_frame(rows), None) is hint

    # The next spawn reveals the lock: stack rows commit, T is hinted.
    locked = rows_of(piece_cells("S", 0, 18, 3))
    with_t = merge(locked, piece_cells("T", 0, 0, 3))
    engine.process_frame(_frame(with_t), None)
    hint2 = engine.process_frame(_frame(with_t), None)
    assert hint2 is not None
    assert hint2.piece == "T"
    assert engine.tracker.committed.stack_rows == locked
    assert events_log == [
        GameEvent.PIECE_SPAWNED,
        GameEvent.PIECE_LOCKED,
        GameEvent.PIECE_SPAWNED,
    ]

    # Three whole-board white-flash frames over the committed non-empty
    # stack: on a white theme a flash is pixel-identical to an empty
    # board, so it passes the gate as empty/0.5 — but against a non-empty
    # stack it is UNEXPLAINED, and below the tracker's reset debounce
    # (4 identical frames) it commits nothing and the hint is held.
    committed = engine.tracker.committed
    flash = np.full((20 * CELL, 10 * CELL, 3), 250, dtype=np.uint8)
    for _ in range(3):
        assert engine.process_frame(flash, None) is hint2
    assert engine.tracker.committed == committed
    assert GameEvent.BOARD_RESET not in events_log

    # Clean recovery: the next coherent frame resets the streak.
    assert engine.process_frame(_frame(with_t), None) is hint2
    assert engine.tracker.committed == committed

    # A real wipe: four identical empty frames pass the debounce and
    # reset the board; the stale hint is cleared.
    for _ in range(3):
        assert engine.process_frame(_frame(EMPTY), None) is hint2
    assert engine.process_frame(_frame(EMPTY), None) is None
    assert events_log.count(GameEvent.BOARD_RESET) == 1
    assert engine.tracker.committed.stack_rows == EMPTY
    assert engine.current_hint is None


def test_fixture_stream_never_commits() -> None:
    # The real failure mode this redesign fixes, end to end: repeated
    # frames of the game's start screen (text over a white board) with a
    # blank white preview box. Every board frame must be rejected at the
    # gate — no hint, no events, no commit, no crash. Before the fix the
    # start screen read 200/200 occupied and moments of contrast
    # committed garbage grids.
    engine, events_log = _engine_with_spy()
    board = np.asarray(Image.open(FIXTURES / "roas_stacker" / "start_screen_board.png"))
    preview = np.asarray(Image.open(FIXTURES / "roas_stacker" / "blank_light_preview.png"))
    initial = engine.tracker.committed
    for _ in range(8):
        assert engine.process_frame(board, preview) is None
    assert engine.tracker.committed == initial
    assert events_log == []
