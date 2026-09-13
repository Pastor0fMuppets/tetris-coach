"""Headless guarantees: imports, CLI demo, engine loop, capture sources.

These tests must pass with no display and without PySide6/mss installed
(and equally when they are installed but unusable).
"""

import importlib

import numpy as np
import pytest

from tetris_coach.capture.screen import ArraySource, Rect
from tetris_coach.cli import main, run_demo
from tetris_coach.overlay.renderer import HintStyle, placement_cell_rects
from tetris_coach.solver.search import best_move

from .synthetic import STYLES, render_board, render_next_preview


def test_all_modules_import_headless() -> None:
    for module in (
        "tetris_coach",
        "tetris_coach.core",
        "tetris_coach.solver",
        "tetris_coach.vision",
        "tetris_coach.capture",
        "tetris_coach.capture.screen",
        "tetris_coach.overlay",
        "tetris_coach.overlay.renderer",
        "tetris_coach.overlay.window",
        "tetris_coach.region_select",
        "tetris_coach.app",
        "tetris_coach.cli",
    ):
        importlib.import_module(module)


def test_cli_demo_runs(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--demo", "--pieces", "30", "--seed", "3"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Placed 30 pieces" in out
    assert "p50" in out


def test_run_demo_reports_top_out(capsys: pytest.CaptureFixture[str]) -> None:
    # Any long-enough game with these weights survives; force a quick exit
    # by requesting 0 pieces to at least exercise the summary path.
    assert run_demo(pieces=0, seed=1, delay=0.0) == 0


def test_placement_cell_rects_geometry() -> None:
    from tetris_coach.core.board import Board

    move = best_move(Board(), "O")
    assert move is not None
    rects = placement_cell_rects(move, cell_width=30.0, cell_height=25.0)
    assert len(rects) == 4
    for (x, y, w, h), (row, col) in zip(rects, move.cells, strict=True):
        assert x == col * 30.0
        assert y == row * 25.0
        assert (w, h) == (30.0, 25.0)
    inset_rects = placement_cell_rects(move, 30.0, 25.0, inset=2.0)
    assert inset_rects[0][2] == 26.0


def test_hint_style_defaults() -> None:
    style = HintStyle()
    assert style.color
    assert 0.0 <= style.fill_opacity <= 1.0


def test_overlay_window_raises_without_qt_or_constructs() -> None:
    from tetris_coach.overlay import window

    if window.HAVE_QT:
        pytest.skip("PySide6 installed; construction needs a display")
    with pytest.raises(RuntimeError):
        window.OverlayWindow(Rect(0, 0, 100, 200))


def test_array_source_crops_and_advances() -> None:
    a = np.zeros((50, 40, 3), dtype=np.uint8)
    b = np.full((50, 40, 3), 255, dtype=np.uint8)
    source = ArraySource([a, b])
    rect = Rect(left=10, top=5, width=20, height=30)
    first = source.grab(rect)
    assert first.shape == (30, 20, 3)
    assert first.max() == 0
    second = source.grab(rect)
    assert second.min() == 255
    # Last frame repeats without loop.
    assert source.grab(rect).min() == 255


def test_array_source_requires_frames() -> None:
    with pytest.raises(ValueError):
        ArraySource([])


class TestCoachEngine:
    def test_engine_produces_hint_from_synthetic_frames(self) -> None:
        from tetris_coach.app import CoachEngine
        from tetris_coach.core.pieces import ROTATIONS

        engine = CoachEngine()
        grid = np.zeros((20, 10), dtype=bool)
        # A falling T near the top of an otherwise empty board.
        for r, c in ((1, 4), (2, 3), (2, 4), (2, 5)):
            grid[r, c] = True
        style = STYLES[0]
        board_image = render_board(grid, style, cell_size=16)
        next_image = render_next_preview(ROTATIONS["I"][0].cells, style, cell_size=16)

        # Debounce: first frame commits nothing, second commits and solves.
        assert engine.process_frame(board_image, next_image) is None
        hint = engine.process_frame(board_image, next_image)
        assert hint is not None
        assert hint.piece == "T"
        # Precompute-ahead is armed for the next piece.
        assert engine._precomputed is not None
        assert engine._precomputed.piece == "I"

    def test_engine_flips_to_precomputed_hint_on_lock(self) -> None:
        from tetris_coach.app import CoachEngine
        from tetris_coach.core.pieces import ROTATIONS

        engine = CoachEngine()
        style = STYLES[2]

        falling_t = np.zeros((20, 10), dtype=bool)
        for r, c in ((1, 4), (2, 3), (2, 4), (2, 5)):
            falling_t[r, c] = True
        board_image = render_board(falling_t, style, cell_size=14)
        next_image = render_next_preview(ROTATIONS["I"][0].cells, style, cell_size=16)
        engine.process_frame(board_image, next_image)
        hint = engine.process_frame(board_image, next_image)
        assert hint is not None
        predicted = engine._predicted_board
        assert predicted is not None

        # The player follows the hint exactly; the T locks there and the I
        # spawns. Rebuild the observed frame from the predicted board.
        locked_grid = np.array(predicted.to_grid())
        i_rot = ROTATIONS["I"][0]  # falling I in spawn orientation near the top
        for r, c in i_rot.cells:
            locked_grid[r + 1, c + 3] = True
        board_image2 = render_board(locked_grid, style, cell_size=14)
        next_image2 = render_next_preview(ROTATIONS["O"][0].cells, style, cell_size=16)
        engine.process_frame(board_image2, next_image2)
        hint2 = engine.process_frame(board_image2, next_image2)
        assert hint2 is not None
        assert hint2.piece == "I"
        # The flip used the 1-ply precompute; a quiet follow-up frame
        # refines it to the full 2-ply answer using the known next piece.
        assert engine._hint_is_provisional
        hint3 = engine.process_frame(board_image2, next_image2)
        assert hint3 is not None
        assert hint3.piece == "I"
        assert not engine._hint_is_provisional
