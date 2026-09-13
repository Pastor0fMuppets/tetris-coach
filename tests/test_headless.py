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
from tetris_coach.vision.state import GameEvent

from .boards import bottom_lines, grid_of, merge, piece_cells, rows_of
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

    def test_preview_vision_skipped_on_identical_frames(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        # F9: the preview image is byte-identical on most frames; the full
        # identify_next pass must run only when the pixels change.
        import tetris_coach.app as app_module
        from tetris_coach.app import CoachEngine
        from tetris_coach.core.pieces import ROTATIONS

        calls = 0
        real = app_module.identify_next

        def counting(image):  # type: ignore[no-untyped-def]
            nonlocal calls
            calls += 1
            return real(image)

        monkeypatch.setattr(app_module, "identify_next", counting)
        engine = CoachEngine()
        grid = np.zeros((20, 10), dtype=bool)
        for r, c in ((1, 4), (2, 3), (2, 4), (2, 5)):
            grid[r, c] = True
        style = STYLES[0]
        board_image = render_board(grid, style, cell_size=16)
        next_image = render_next_preview(ROTATIONS["I"][0].cells, style, cell_size=16)

        for _ in range(4):
            # Fresh copies: the cache must compare content, not identity.
            hint = engine.process_frame(board_image, next_image.copy())
        assert calls == 1
        assert hint is not None and hint.piece == "T"
        assert engine._precomputed is not None and engine._precomputed.piece == "I"

        # The preview changes: one more real pass.
        other = render_next_preview(ROTATIONS["O"][0].cells, style, cell_size=16)
        engine.process_frame(board_image, other)
        assert calls == 2
        # No preview region at all: no pass, and no stale cached answer.
        engine.process_frame(board_image, None)
        assert calls == 2

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


class TestCoachEngineScenarios:
    """End-to-end regressions for the diff-anchored vision redesign."""

    STYLE = STYLES[2]  # jstris-like: solid colors, no noise
    CELL = 14

    def _engine_with_spy(self):  # type: ignore[no-untyped-def]
        from tetris_coach.app import CoachEngine

        engine = CoachEngine()
        events_log: list[GameEvent] = []
        original = engine.tracker.update

        def spy(occupancy, nxt):  # type: ignore[no-untyped-def]
            events = original(occupancy, nxt)
            events_log.extend(events)
            return events

        engine.tracker.update = spy  # type: ignore[method-assign]
        return engine, events_log

    def _board_image(self, rows: tuple[int, ...]):  # type: ignore[no-untyped-def]
        return render_board(grid_of(rows), self.STYLE, cell_size=self.CELL)

    def _preview(self, piece: str | None):  # type: ignore[no-untyped-def]
        from tetris_coach.core.pieces import ROTATIONS

        if piece is None:
            return np.full((64, 96, 3), 10, dtype=np.uint8)  # blank box
        return render_next_preview(ROTATIONS[piece][0].cells, self.STYLE, cell_size=16)

    def _process(self, engine, rows, next_piece, times=1):  # type: ignore[no-untyped-def]
        board_image = self._board_image(rows)
        next_image = self._preview(next_piece)
        hint = None
        for _ in range(times):
            hint = engine.process_frame(board_image, next_image)
        return hint

    def _attach(self, engine, rows, next_piece=None):  # type: ignore[no-untyped-def]
        """Force-commit a stack via the reset rule (test setup)."""
        self._process(engine, rows, next_piece, times=4)
        assert engine.tracker.committed.stack_rows == rows

    def test_hint_persists_through_lock_delay(self) -> None:
        # F1: the piece rests on the floor for several frames before the
        # game locks it; the hint must not disappear.
        engine, events_log = self._engine_with_spy()
        hint = self._process(engine, rows_of(piece_cells("T", 0, 1, 3)), "I", times=2)
        assert hint is not None
        assert hint.piece == "T"
        resting = rows_of(piece_cells("T", 0, 18, 3))
        for _ in range(4):
            assert self._process(engine, resting, "I") is hint
        assert GameEvent.PIECE_LOCKED not in events_log

    def test_no_wipe_on_clear_lock(self) -> None:
        # F2: a line-clearing lock must never be misread as BOARD_RESET
        # (which would wipe the caches and the hint).
        engine, events_log = self._engine_with_spy()
        stack = rows_of(bottom_lines("#########."))
        self._attach(engine, stack, "I")
        events_log.clear()  # the setup resync is expected; the lock is not

        hint = self._process(engine, merge(stack, piece_cells("I", 1, 4, 9)), "T", times=2)
        assert hint is not None
        assert hint.piece == "I"
        self._process(engine, merge(stack, piece_cells("I", 1, 16, 9)), "T")
        # Morphing fade frames of the clearing row: no commit, hint held.
        lit = merge(stack, piece_cells("I", 1, 16, 9))
        fade1 = tuple(row & ~0b0000111100 if r == 19 else row for r, row in enumerate(lit))
        fade2 = tuple(row & ~0b0111111110 if r == 19 else row for r, row in enumerate(lit))
        assert self._process(engine, fade1, "T") is not None
        assert self._process(engine, fade2, "T") is not None
        # Settled collapsed board + the next spawn: hint present on the
        # commit frame (flip or solve), and never a reset.
        s2 = rows_of([(17, 9), (18, 9), (19, 9)])
        hint2 = self._process(engine, merge(s2, piece_cells("T", 0, 0, 3)), "S", times=2)
        assert hint2 is not None
        assert hint2.piece == "T"
        assert GameEvent.PIECE_LOCKED in events_log
        assert GameEvent.BOARD_RESET not in events_log
        assert engine.tracker.committed.stack_rows == s2

    def test_debris_never_hinted(self) -> None:
        # F3: a floating island in the committed stack is never mistaken
        # for the falling piece, even when the real piece is below it.
        engine, events_log = self._engine_with_spy()
        # An O-shaped island left at rows 10-11 after the rows below cleared,
        # above a normal bottom stack.
        island = rows_of(piece_cells("O", 0, 10, 7), bottom_lines("#####....."))
        self._attach(engine, island, "I")
        events_log.clear()

        hint = self._process(engine, merge(island, piece_cells("T", 0, 1, 3)), "I", times=2)
        assert hint is not None
        assert hint.piece == "T"
        for rows in (
            merge(island, piece_cells("T", 0, 8, 5)),
            merge(island, piece_cells("T", 0, 15, 6)),  # below the island
            merge(island, piece_cells("T", 0, 17, 3)),
        ):
            for _ in range(2):
                hint = self._process(engine, rows, "I")
                assert hint is not None
                assert hint.piece == "T"
        assert GameEvent.BOARD_RESET not in events_log
        assert GameEvent.PIECE_LOCKED not in events_log

    def test_late_next_read_upgrades_hint(self) -> None:
        # F5: a hint solved while the preview was blank is provisional and
        # upgrades to 2-ply the moment the preview becomes readable.
        engine, _ = self._engine_with_spy()
        spawn = rows_of(piece_cells("T", 0, 1, 3))
        hint = self._process(engine, spawn, None, times=2)
        assert hint is not None
        assert hint.piece == "T"
        assert engine._hint_is_provisional
        # The preview becomes readable: quiet commit, then refinement.
        hint2 = self._process(engine, spawn, "I", times=2)
        assert hint2 is not None
        assert hint2.piece == "T"
        assert not engine._hint_is_provisional
        assert engine._precomputed is not None
        assert engine._precomputed.piece == "I"

    def test_flip_with_blank_preview_stays_provisional(self) -> None:
        from tetris_coach.core.board import Board

        engine, _ = self._engine_with_spy()
        spawn = rows_of(piece_cells("T", 0, 1, 3))
        hint = self._process(engine, spawn, "I", times=2)
        assert hint is not None
        predicted = engine._predicted_board
        assert predicted is not None
        # The T locks at the predicted target and the I spawns, but the
        # preview blanks during the piece-shift animation.
        locked = merge(
            tuple(predicted.rows), piece_cells("I", 0, 1, 3)
        )
        hint2 = self._process(engine, locked, None, times=2)
        assert hint2 is not None
        assert hint2.piece == "I"
        assert engine._hint_is_provisional
        # Still blank: stays provisional (no next piece to refine with).
        assert self._process(engine, locked, None) is hint2
        assert engine._hint_is_provisional
        # Preview reads: refined to 2-ply and precompute re-armed.
        hint3 = self._process(engine, locked, "O", times=2)
        assert hint3 is not None
        assert hint3.piece == "I"
        assert not engine._hint_is_provisional
        assert engine._precomputed is not None
        assert engine._precomputed.piece == "O"
        assert engine._predicted_board == Board(hint3.board.rows)

    def test_lock_gap_precompute_flips_without_solve(self) -> None:
        from tetris_coach.core.board import Board

        engine, events_log = self._engine_with_spy()
        stack = rows_of(bottom_lines("#########."))
        self._attach(engine, stack, "I")
        events_log.clear()

        self._process(engine, merge(stack, piece_cells("I", 1, 4, 9)), "T", times=2)
        self._process(engine, merge(stack, piece_cells("I", 1, 16, 9)), "T")
        # The clear settles with no new spawn visible yet (lock gap), but
        # the preview is readable: the upcoming piece is pre-solved.
        s2 = rows_of([(17, 9), (18, 9), (19, 9)])
        hint = self._process(engine, s2, "T", times=2)
        assert hint is None
        assert GameEvent.PIECE_LOCKED in events_log
        precomputed = engine._precomputed
        assert precomputed is not None
        assert precomputed.piece == "T"
        assert engine._predicted_board == Board(s2)
        # The spawn flips instantly to the precomputed hint (no re-solve).
        hint2 = self._process(engine, merge(s2, piece_cells("T", 0, 0, 3)), "S", times=2)
        assert hint2 is precomputed
        assert engine._hint_is_provisional

    def test_confidence_drop_mid_sequence(self) -> None:
        engine, _events_log = self._engine_with_spy()
        spawn = rows_of(piece_cells("T", 0, 1, 3))
        hint = self._process(engine, spawn, "I", times=2)
        assert hint is not None
        committed = engine.tracker.committed
        # A washed-out frame (smooth gradient: classes barely separate).
        gradient = np.tile(
            np.linspace(0, 255, 10 * self.CELL, dtype=np.uint8), (20 * self.CELL, 1)
        )
        washed = np.stack([gradient] * 3, axis=2)
        assert engine.process_frame(washed, self._preview("I")) is hint
        assert engine.tracker.committed == committed
        # The next good frame carries on as if nothing happened.
        assert self._process(engine, spawn, "I") is hint
