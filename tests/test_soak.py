"""Self-play soak: the tracker (and engine) follow a full simulated game.

The solver plays itself; every placement is turned into a realistic frame
sequence (spawn, descent, lock delay, clear flash/fade, occasional torn
frames and blank previews, occasional instant hard drops). After every
lock the tracker's committed stack must equal the simulator's ground
truth, every placement must produce exactly one PIECE_LOCKED, and
BOARD_RESET must never fire.
"""

import random

import numpy as np
import pytest

from tetris_coach.core.board import FULL_ROW, HEIGHT, Board
from tetris_coach.core.pieces import PIECES
from tetris_coach.solver.search import best_move
from tetris_coach.vision.state import GameEvent, GameStateTracker

from .boards import Cell, grid_of, merge, piece_cells
from .synthetic import STYLES, render_board, render_next_preview


def _spawn_cells(piece: str) -> list[Cell]:
    return piece_cells(piece, 0, 0, 3)


def _disjoint(rows: tuple[int, ...], cells: list[Cell]) -> bool:
    return all(not rows[r] >> c & 1 for r, c in cells)


def _touching(a_cells: list[Cell], b_cells: list[Cell]) -> bool:
    b = set(b_cells)
    for r, c in a_cells:
        if (r, c) in b:
            return True
        if any(n in b for n in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))):
            return True
    return False


def _fade_frames(pre: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Morphing partial-row frames of a clear animation."""
    full = [r for r in range(len(pre)) if pre[r] == FULL_ROW]
    frames = []
    for mask in (0b1010101010, 0b0001111000):
        rows = list(pre)
        for r in full:
            rows[r] &= mask
        frames.append(tuple(rows))
    return frames


@pytest.mark.parametrize("rows", [HEIGHT, 12])
def test_tracker_soak_self_play(rows: int) -> None:
    rng = random.Random(20260913)
    board = Board([0] * rows)
    tracker = GameStateTracker(rows=rows)
    lock_count = 0
    reset_count = 0

    def feed(rows: tuple[int, ...], nxt: str | None) -> list[GameEvent]:
        nonlocal lock_count, reset_count
        events = tracker.update(grid_of(rows), nxt)
        lock_count += events.count(GameEvent.PIECE_LOCKED)
        reset_count += events.count(GameEvent.BOARD_RESET)
        return events

    current = rng.choice(PIECES)
    upcoming = rng.choice(PIECES)
    placed = 0
    for _ in range(200):
        move = best_move(board, current, upcoming)
        assert move is not None, "solver topped out mid-soak"
        spawn = _spawn_cells(current)
        assert _disjoint(board.rows, spawn), "stack reached the spawn zone"
        nxt = upcoming if rng.random() > 0.15 else None  # occasional blank preview
        observed = merge(board.rows, spawn)
        feed(observed, nxt)
        feed(observed, nxt)
        # Ground truth after every lock (revealed by this spawn):
        assert tracker.committed.stack_rows == board.rows
        assert tracker.committed.falling_piece == current
        assert reset_count == 0

        # Occasional torn frame: must change nothing.
        if rng.random() < 0.2:
            blob = [(7, 0), (7, 1), (7, 2), (8, 1), (8, 2)]
            if _disjoint(observed, blob):
                assert feed(merge(observed, blob), nxt) == []

        final_cells = list(move.cells)
        next_spawn = _spawn_cells(upcoming)
        hard_drop = rng.random() < 0.3
        # An instant lock 4-adjacent to the next spawn is only resolvable
        # by the position anchor, which needs the rest position observed.
        if hard_drop and _touching(final_cells, next_spawn):
            hard_drop = False
        if not hard_drop:
            for row in (4, 9):  # a couple of descent frames
                cells = piece_cells(current, 0, row, 3)
                if _disjoint(board.rows, cells):
                    feed(merge(board.rows, cells), nxt)
            for _ in range(rng.randint(1, 3)):  # lock-delay frames at rest
                feed(merge(board.rows, final_cells), nxt)
            if move.lines_cleared:
                pre = merge(board.rows, final_cells)
                feed(pre, nxt)  # flash frame: cleared rows still lit
                for fade in _fade_frames(pre)[: rng.randint(1, 2)]:
                    feed(fade, nxt)
        elif move.lines_cleared:
            # Zero-ARE clear flash: the completed rows are still fully lit
            # while the next piece is already visible. Committing such a
            # frame would anchor the tracker on a stack with a full row.
            flash = merge(board.rows, final_cells)
            if _disjoint(flash, next_spawn):
                before = tracker.committed
                for _ in range(2):
                    assert feed(merge(flash, next_spawn), nxt) == []
                assert tracker.committed == before

        board = move.board
        current, upcoming = upcoming, rng.choice(PIECES)
        placed += 1

    # One more spawn reveals the final lock.
    spawn = _spawn_cells(current)
    assert _disjoint(board.rows, spawn)
    observed = merge(board.rows, spawn)
    feed(observed, None)
    feed(observed, None)
    assert tracker.committed.stack_rows == board.rows

    assert placed == 200
    assert lock_count == placed
    assert reset_count == 0


def test_engine_soak_self_play() -> None:
    from tetris_coach.app import CoachEngine
    from tetris_coach.core.pieces import ROTATIONS

    style = STYLES[2]
    engine = CoachEngine()
    events_log: list[GameEvent] = []
    original = engine.tracker.update

    def spy(occupancy, nxt):  # type: ignore[no-untyped-def]
        events = original(occupancy, nxt)
        events_log.extend(events)
        return events

    engine.tracker.update = spy  # type: ignore[method-assign]

    def process(rows: tuple[int, ...], nxt: str | None):  # type: ignore[no-untyped-def]
        board_image = render_board(grid_of(rows), style, cell_size=12)
        if nxt is None:
            next_image = np.full((64, 96, 3), 10, dtype=np.uint8)
        else:
            next_image = render_next_preview(ROTATIONS[nxt][0].cells, style, cell_size=16)
        return engine.process_frame(board_image, next_image)

    rng = random.Random(7)
    board = Board()
    # The pre-game empty board, as every real session sees it before the
    # first spawn. It anchors the classifier's background memory, so the
    # first spawn — which touches row 0, where a memoryless reading is
    # never vouched for — is read at full confidence like every later one.
    process(board.rows, None)

    current = rng.choice(PIECES)
    upcoming = rng.choice(PIECES)
    placed = 0
    for _ in range(30):
        move = best_move(board, current, upcoming)
        assert move is not None
        spawn = _spawn_cells(current)
        assert _disjoint(board.rows, spawn)
        nxt = upcoming if rng.random() > 0.2 else None
        observed = merge(board.rows, spawn)
        process(observed, nxt)
        hint = process(observed, nxt)
        # Whenever a hint is shown it names the true current piece.
        assert hint is not None
        assert hint.piece == current
        assert engine.tracker.committed.stack_rows == board.rows

        final_cells = list(move.cells)
        resting = merge(board.rows, final_cells)
        hint = process(resting, nxt)  # lock delay: hint held
        assert hint is not None
        assert hint.piece == current
        if move.lines_cleared:
            for fade in _fade_frames(resting)[:1]:
                hint = process(fade, nxt)
                assert hint is not None
                assert hint.piece == current

        board = move.board
        current, upcoming = upcoming, rng.choice(PIECES)
        placed += 1

    spawn = _spawn_cells(current)
    assert _disjoint(board.rows, spawn)
    observed = merge(board.rows, spawn)
    process(observed, upcoming)
    hint = process(observed, upcoming)
    assert hint is not None
    assert hint.piece == current
    assert engine.tracker.committed.stack_rows == board.rows

    assert events_log.count(GameEvent.PIECE_LOCKED) == placed
    assert GameEvent.BOARD_RESET not in events_log
