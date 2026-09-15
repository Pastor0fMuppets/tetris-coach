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

from .boards import EMPTY, bottom_lines, grid_of, merge, piece_cells, rows_of
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


def test_cli_demo_runs_on_12_rows(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--demo", "--rows", "12", "--pieces", "30", "--seed", "3"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Placed 30 pieces" in out
    # The rendered final board is 12 rows tall (plus the floor line).
    board_lines = [line for line in out.splitlines() if line.startswith("|")]
    assert len(board_lines) == 12


def test_cli_rejects_too_few_rows(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--demo", "--rows", "4"]) == 2
    assert "--rows must be at least 5" in capsys.readouterr().err
    assert main(["--demo", "--rows", "0"]) == 2


def test_region_size_validation_messages() -> None:
    # F6: a click-without-drag (or any too-small selection) must be
    # rejected with an explanation, not fed to the capture loop.
    from tetris_coach.cli import _rect_error
    from tetris_coach.region_select import MIN_BOARD_SIZE, MIN_PREVIEW_SIZE

    assert _rect_error(Rect(0, 0, 120, 240), MIN_BOARD_SIZE, "Board") is None
    message = _rect_error(Rect(10, 10, 1, 1), MIN_BOARD_SIZE, "Board")
    assert message is not None
    assert "1x1" in message and "40x80" in message
    assert _rect_error(Rect(0, 0, 15, 40), MIN_PREVIEW_SIZE, "Next-piece") is not None


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
    def test_a_refuted_preview_name_takes_its_hint_off_the_screen(self) -> None:
        # End to end, on rendered frames: the preview names a piece the
        # top edge has cut in half, the coach hints it, and the next frame
        # of the piece's own descent rules that name out. The hint has to
        # come DOWN — it is a placement for a piece the player does not
        # have. Before the retraction it stayed up for the rest of the
        # piece's tenure at the top edge, because the frame that refutes a
        # hint names no replacement and so holds the committed state.
        from tetris_coach.app import CoachEngine
        from tetris_coach.core.pieces import ROTATIONS

        style = STYLES[0]
        engine = CoachEngine()

        def frame(cells: tuple[tuple[int, int], ...], nxt: str):  # type: ignore[no-untyped-def]
            return engine.process_frame(
                render_board(grid_of(rows_of(cells)), style, cell_size=16),
                render_next_preview(ROTATIONS[nxt][0].cells, style, cell_size=16),
            )

        for _ in range(2):
            assert frame((), "T") is None  # the box holds the T
        one = ((0, 0),)
        frame(one, "I")  # the T is dealt: the box flips to the I
        frame(one, "I")  # ...believed on the flip's second capture
        hint = frame(one, "I")
        assert hint is not None and hint.piece == "T"  # hinted on one cell
        # The piece descends: a second cell in the same column, which no T
        # fits. Nothing on screen until it names itself.
        assert frame(((0, 0), (1, 0)), "I") is None
        assert frame(((0, 0), (1, 0)), "I") is None

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

        def counting(image, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal calls
            calls += 1
            return real(image, **kwargs)

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

    def test_debug_view_prints_per_committed_frame(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # S2: --debug prints the full "what vision sees" block once per
        # committed frame (never per raw frame), plus a throttled one-line
        # [vision] status on every frame whose gate/occupancy state changes
        # so a never-committing stream is still diagnosable.
        from tetris_coach.app import CoachConfig, CoachEngine, render_debug_frame
        from tetris_coach.core.pieces import ROTATIONS

        engine = CoachEngine(CoachConfig(debug=True))
        grid = np.zeros((20, 10), dtype=bool)
        for r, c in ((1, 4), (2, 3), (2, 4), (2, 5)):
            grid[r, c] = True
        style = STYLES[0]
        board_image = render_board(grid, style, cell_size=16)
        next_image = render_next_preview(ROTATIONS["I"][0].cells, style, cell_size=16)

        engine.process_frame(board_image, next_image)  # debounce: no commit
        out = capsys.readouterr().out
        assert "events:" not in out  # no committed-frame block yet
        assert "[vision]" in out  # but the status line always reports
        assert "gate ok" in out
        engine.process_frame(board_image, next_image)  # commit: one block
        out = capsys.readouterr().out
        assert out.count("events:") == 1
        assert "PIECE_SPAWNED" in out
        assert "falling: T rot0 @ (row 1, col 3)" in out
        assert "next: I" in out
        assert "confidence:" in out
        assert "....#....." in out  # observed grid, row 1
        assert "...###...." in out  # observed grid, row 2
        # An identical quiet frame commits nothing, and its unchanged
        # status is throttled away: no output at all.
        engine.process_frame(board_image, next_image)
        assert capsys.readouterr().out == ""
        # Absent falling piece / events render as placeholders.
        text = render_debug_frame(
            np.zeros((20, 10), dtype=bool), 0.42, engine.tracker.committed, None, []
        )
        assert "falling: -" in text
        assert "events: -" in text
        assert "confidence: 0.42" in text

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


class TestFrameWorker:
    """F10: the worker-thread tick logic, driven headless (no Qt).

    FrameWorker is exactly what the live app runs on the QThreadPool
    thread; these tests exercise its grab->engine->result contract and
    the consecutive-failure accounting without any GUI.
    """

    BOARD_RECT = Rect(0, 0, 160, 320)
    NEXT_RECT = Rect(200, 0, 96, 64)

    def _worker(self):  # type: ignore[no-untyped-def]
        from tetris_coach.app import CoachEngine, FrameWorker
        from tetris_coach.core.pieces import ROTATIONS

        style = STYLES[0]
        grid = np.zeros((20, 10), dtype=bool)
        for r, c in ((1, 4), (2, 3), (2, 4), (2, 5)):
            grid[r, c] = True
        board_image = render_board(grid, style, cell_size=16)
        next_image = render_next_preview(ROTATIONS["I"][0].cells, style, cell_size=16)
        board_rect, next_rect = self.BOARD_RECT, self.NEXT_RECT

        class Source:
            def grab(self, rect: Rect) -> np.ndarray:
                return board_image if rect == board_rect else next_image

        return FrameWorker(CoachEngine(), Source(), board_rect, next_rect)

    def test_tick_processes_frames_and_reports_hint(self) -> None:
        worker = self._worker()
        first = worker.run_tick()
        # Debounce: the first committed-nothing frame is still a success.
        assert first.ok and not first.stop and first.hint is None
        second = worker.run_tick()
        assert second.ok and not second.stop
        assert second.hint is not None
        assert second.hint.piece == "T"
        assert worker.consecutive_failures == 0

    def test_failure_streak_logs_once_and_stops_at_cap(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from tetris_coach.app import (
            MAX_CONSECUTIVE_TICK_FAILURES,
            CoachEngine,
            FrameWorker,
        )

        class Broken:
            def grab(self, rect: Rect) -> np.ndarray:
                raise OSError("display gone")

        worker = FrameWorker(CoachEngine(), Broken(), self.BOARD_RECT, None)
        results = [worker.run_tick() for _ in range(MAX_CONSECUTIVE_TICK_FAILURES)]
        assert all(not r.ok and r.hint is None for r in results)
        # Every failed tick below the cap asks the loop to carry on ...
        assert not any(r.stop for r in results[:-1])
        # ... and the tick that reaches the cap asks it to shut down.
        assert results[-1].stop
        err = capsys.readouterr().err
        # Logged once per failure streak (plus the final give-up notice),
        # never once per frame.
        assert err.count("skipping frames") == 1
        assert err.count("consecutive frames") == 1

    def test_success_resets_failure_streak(self, capsys: pytest.CaptureFixture[str]) -> None:
        from tetris_coach.app import MAX_CONSECUTIVE_TICK_FAILURES

        worker = self._worker()
        good_source = worker.frame_source

        class Broken:
            def grab(self, rect: Rect) -> np.ndarray:
                raise OSError("display gone")

        for _ in range(MAX_CONSECUTIVE_TICK_FAILURES - 1):
            worker.frame_source = Broken()
            assert not worker.run_tick().ok
            worker.frame_source = good_source
            result = worker.run_tick()
            assert result.ok and not result.stop
            assert worker.consecutive_failures == 0
        # Interleaved successes: the cap is never reached, and each new
        # streak logs afresh.
        assert capsys.readouterr().err.count("skipping frames") == (
            MAX_CONSECUTIVE_TICK_FAILURES - 1
        )


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
        # TWO O-shaped islands left at rows 10-11 after the rows below
        # cleared, above a normal bottom stack. Two of them, because a
        # resync holds back a piece in flight and one floating tetromino is
        # exactly that; a second one puts the frame past that budget, so
        # the whole board is adopted as the stack (see
        # ``strip_piece_in_flight`` and TestResyncAndThePieceInFlight).
        island = rows_of(
            piece_cells("O", 0, 10, 7),
            piece_cells("O", 0, 10, 2),
            bottom_lines("#####....."),
        )
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
        locked = merge(tuple(predicted.rows), piece_cells("I", 0, 1, 3))
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

    def test_flash_and_transient_occlusion_never_reset(self) -> None:
        # F4, background-memory era: by the time a mid-game flash can
        # happen, the classifier's memory is anchored on this dark
        # theme's background, so the two uniform frames part ways. The
        # bright flash is uniform-FAR from the anchor: gated outright
        # (all-occupied @0.0) on every frame of ANY duration — the
        # strong original F4 guarantee, see
        # test_bright_overlay_on_dark_theme_never_resets. The dark
        # occlusion sits AT the anchor: pixel-indistinguishable from a
        # real wipe (see test_board_wipe_resets_and_clears_hint), it
        # reads empty with usable confidence and is UNEXPLAINED against
        # the committed stack — held harmless while its identical-frame
        # streak stays below the reset debounce (4 frames); a coherent
        # frame in between resets the streak.
        engine, events_log = self._engine_with_spy()
        stack = rows_of(bottom_lines("#########."))
        self._attach(engine, stack, "I")
        events_log.clear()
        falling = merge(stack, piece_cells("I", 1, 4, 9))
        hint = self._process(engine, falling, "T", times=2)
        assert hint is not None
        committed = engine.tracker.committed

        shape = (20 * self.CELL, 10 * self.CELL, 3)
        flash = np.full(shape, 245, dtype=np.uint8)
        occluded = np.full(shape, 15, dtype=np.uint8)
        for image in (flash, flash, occluded):
            assert engine.process_frame(image, self._preview("T")) is hint
        assert self._process(engine, falling, "T") is hint  # streak resets
        for image in (occluded, occluded, flash):
            assert engine.process_frame(image, self._preview("T")) is hint
        assert engine.tracker.committed == committed
        assert GameEvent.BOARD_RESET not in events_log
        # The next good frame carries on as if nothing happened.
        assert self._process(engine, falling, "T") is hint

    def test_bright_overlay_on_dark_theme_never_resets(self) -> None:
        # The restored strong F4: a bright uniform frame CANNOT be this
        # board's empty state once the background memory is anchored
        # dark, so a solid pause panel / game-over splash / whole-board
        # flash of ANY duration — far beyond the reset debounce — never
        # commits anything, and when it lifts the same committed stack
        # still explains the next frame (no second re-sync needed).
        engine, events_log = self._engine_with_spy()
        stack = rows_of(bottom_lines("####..####", "#########."))
        self._attach(engine, stack, "I")
        falling = merge(stack, piece_cells("T", 0, 1, 3))
        hint = self._process(engine, falling, "I", times=2)
        assert hint is not None
        committed = engine.tracker.committed
        events_log.clear()  # setup: the attach resync and the T spawn

        overlay = np.full((20 * self.CELL, 10 * self.CELL, 3), 245, dtype=np.uint8)
        for _ in range(12):  # 3x the 4-frame reset debounce
            assert engine.process_frame(overlay, self._preview("I")) is hint
        assert engine.tracker.committed == committed
        assert events_log == []
        # Overlay lifts: play continues on the same committed state.
        assert self._process(engine, falling, "I") is hint
        assert engine.tracker.committed == committed
        assert events_log == []

    def test_board_wipe_resets_and_clears_hint(self) -> None:
        # A mid-game wipe (game over / new game) renders a uniform DARK
        # empty board. Vision classifies it as empty with usable
        # confidence — not gated out — so the reset debounce runs: the
        # stale hint is cleared and the stack re-anchors on empty instead
        # of staying painted over the empty board indefinitely.
        engine, events_log = self._engine_with_spy()
        stack = rows_of(bottom_lines("####..####", "#########."))
        self._attach(engine, stack, "I")
        events_log.clear()
        hint = self._process(engine, merge(stack, piece_cells("T", 0, 1, 3)), "I", times=2)
        assert hint is not None
        for _ in range(3):
            assert self._process(engine, EMPTY, None) is hint  # debouncing
        assert self._process(engine, EMPTY, None) is None  # 4th frame: reset
        assert GameEvent.BOARD_RESET in events_log
        assert engine.tracker.committed.stack_rows == EMPTY
        assert engine.current_hint is None
        # The next game's first spawn is tracked and hinted normally.
        hint2 = self._process(engine, rows_of(piece_cells("J", 0, 0, 3)), "S", times=2)
        assert hint2 is not None
        assert hint2.piece == "J"

    def test_confidence_drop_mid_sequence(self) -> None:
        engine, _events_log = self._engine_with_spy()
        spawn = rows_of(piece_cells("T", 0, 1, 3))
        hint = self._process(engine, spawn, "I", times=2)
        assert hint is not None
        committed = engine.tracker.committed
        # A washed-out frame (smooth gradient: classes barely separate).
        gradient = np.tile(np.linspace(0, 255, 10 * self.CELL, dtype=np.uint8), (20 * self.CELL, 1))
        washed = np.stack([gradient] * 3, axis=2)
        assert engine.process_frame(washed, self._preview("I")) is hint
        assert engine.tracker.committed == committed
        # The next good frame carries on as if nothing happened.
        assert self._process(engine, spawn, "I") is hint


class TestBackgroundMemoryScenarios:
    """Engine-level pins for the classifier's cross-frame background memory.

    The states here are legal, reachable Tetris that the memoryless
    top-row estimator misread at committable confidence: a stack growing
    into visible row 0 and versus-mode garbage pushed to the top row both
    leave the top row majority piece-colored, inverting every cell of a
    self-estimated reading (at 0.95 on monochrome themes). Four identical
    inverted frames then passed the reset debounce and BOARD_RESET
    committed the inverted board. With the memory anchored during normal
    play, the same frames read exactly.
    """

    CELL = 20

    @staticmethod
    def _rows(grid: np.ndarray) -> tuple[int, ...]:
        return tuple(int(sum(1 << c for c in range(10) if grid[r, c])) for r in range(20))

    @staticmethod
    def _engine_with_spy():  # type: ignore[no-untyped-def]
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

    def _feed(self, engine, grid: np.ndarray, style, times: int = 1):  # type: ignore[no-untyped-def]
        image = render_board(grid, style, cell_size=self.CELL)
        hint = None
        for _ in range(times):
            hint = engine.process_frame(image, None)
        return hint

    def test_growth_into_top_row_commits_exactly(self) -> None:
        # gray-flat: monochrome pieces, the worst case — self-estimated
        # readings of the final stages inverted at confidence ~0.95, and
        # polarity flipped between ADJACENT frames as the stack grew
        # (4 filled top-row cells: exact; 5: rejected; 6: inverted).
        # Anchored during the empty pre-game frame, every stage commits
        # exactly, through the last one with row 0 filled.
        style = STYLES[1]
        engine, _events_log = self._engine_with_spy()
        self._feed(engine, np.zeros((20, 10), dtype=bool), style)
        for top in range(6, -1, -1):  # stack top row: 6, 5, ..., 0
            grid = np.zeros((20, 10), dtype=bool)
            grid[top:, 0:3] = True
            grid[top:, 7:10] = True
            # Each jump is a mid-game attach: unexplained, committed via
            # the 4-frame reset debounce — to the TRUE board every time.
            self._feed(engine, grid, style, times=5)
            assert engine.tracker.committed.stack_rows == self._rows(grid), (
                f"wrong commit with stack top at row {top}"
            )

    @pytest.mark.parametrize(
        "style",
        [STYLES[1], STYLES[4]],
        ids=lambda s: s.name,  # type: ignore[no-untyped-def]
    )
    def test_garbage_push_resyncs_to_true_board(self, style) -> None:  # type: ignore[no-untyped-def]
        # Versus garbage (9/10 filled, one well) pushed up to the top
        # row: 190/200 non-background cells, top row included. gray-flat
        # pins the monochrome inversion; paper-white pins the MIN_SPREAD
        # re-split (near-background lavender pieces) through the engine.
        engine, _events_log = self._engine_with_spy()
        self._feed(engine, np.zeros((20, 10), dtype=bool), style)
        stack = np.zeros((20, 10), dtype=bool)
        stack[18:, 0:6] = True
        self._feed(engine, stack, style, times=5)
        assert engine.tracker.committed.stack_rows == self._rows(stack)
        garbage = np.ones((20, 10), dtype=bool)
        garbage[:, 6] = False
        self._feed(engine, garbage, style, times=5)
        assert engine.tracker.committed.stack_rows == self._rows(garbage)

    def test_pause_panel_at_startup_recovers(self) -> None:
        # Session started while a solid bright panel covers a dark-theme
        # board: with no memory the panel reads as an empty board (a
        # lone frame cannot know better) and anchors only provisionally.
        # Real frames then re-bootstrap and confirm the true background;
        # from that point the same panel is gated on every frame, so it
        # can never wipe the committed stack.
        style = STYLES[0]
        engine, events_log = self._engine_with_spy()
        panel = np.full((20 * self.CELL, 10 * self.CELL, 3), 235, dtype=np.uint8)
        for _ in range(6):
            assert engine.process_frame(panel, None) is None
        assert engine.tracker.committed.stack_rows == (0,) * 20
        stack = np.zeros((20, 10), dtype=bool)
        stack[17:, 0:5] = True
        self._feed(engine, stack, style, times=5)
        assert engine.tracker.committed.stack_rows == self._rows(stack)
        events_log.clear()  # the mid-game attach above is expected
        for _ in range(10):
            engine.process_frame(panel, None)
        assert engine.tracker.committed.stack_rows == self._rows(stack)
        assert events_log == []


class TestTwelveRowEngine:
    """CoachEngine end to end on a 12-row board (CoachConfig(rows=12)):
    spawn -> hint -> lock -> clear, driven by synthetic frames on a dark
    and a light theme. config.rows is the one fan-out point; everything
    downstream must derive the height from the frames."""

    ROWS = 12
    CELL = 20
    EMPTY12: tuple[int, ...] = (0,) * 12

    def _engine(self, **overrides):  # type: ignore[no-untyped-def]
        from tetris_coach.app import CoachConfig, CoachEngine

        return CoachEngine(CoachConfig(rows=self.ROWS, **overrides))

    def _process(self, engine, style, rows, next_piece, times=1):  # type: ignore[no-untyped-def]
        from tetris_coach.core.pieces import ROTATIONS

        board_image = render_board(grid_of(rows), style, cell_size=self.CELL)
        if next_piece is None:
            next_image = None
        else:
            next_image = render_next_preview(ROTATIONS[next_piece][0].cells, style, cell_size=16)
        hint = None
        for _ in range(times):
            hint = engine.process_frame(board_image, next_image)
        return hint

    @pytest.mark.parametrize(
        "style",
        [STYLES[2], STYLES[4]],
        ids=lambda s: s.name,  # type: ignore[no-untyped-def]
    )
    def test_spawn_hint_lock_clear(self, style) -> None:  # type: ignore[no-untyped-def]
        engine = self._engine()
        # Anchor the background memory on the empty pre-game board.
        assert self._process(engine, style, self.EMPTY12, None) is None
        # Attach a 9/10 bottom row via the reset debounce.
        stack = rows_of(bottom_lines("#########.", height=self.ROWS), height=self.ROWS)
        self._process(engine, style, stack, "I", times=4)
        assert engine.tracker.committed.stack_rows == stack
        # The I spawns: a hint appears within the 2-frame debounce.
        spawn = merge(stack, piece_cells("I", 1, 0, 9))
        hint = self._process(engine, style, spawn, "J", times=2)
        assert hint is not None
        assert hint.piece == "I"
        for r, c in hint.cells:
            assert 0 <= r < self.ROWS
        # The I comes to rest on the 12-row floor, completing the row.
        resting = merge(stack, piece_cells("I", 1, self.ROWS - 4, 9))
        assert self._process(engine, style, resting, "J") is hint
        # Settled post-clear board plus the J spawn: lock + spawn commit.
        s2 = rows_of(
            [(self.ROWS - 3, 9), (self.ROWS - 2, 9), (self.ROWS - 1, 9)],
            height=self.ROWS,
        )
        settled = merge(s2, piece_cells("J", 0, 0, 4))
        hint2 = self._process(engine, style, settled, "O", times=2)
        assert hint2 is not None
        assert hint2.piece == "J"
        assert engine.tracker.committed.stack_rows == s2
        assert engine.tracker.committed.falling_piece == "J"

    def test_debug_denominator_is_data_derived(self, capsys: pytest.CaptureFixture[str]) -> None:
        engine = self._engine(debug=True)
        self._process(engine, STYLES[2], self.EMPTY12, None)
        out = capsys.readouterr().out
        assert f"/{self.ROWS * 10}," in out
        assert "/200," not in out


def test_no_stray_board_size_literals_in_src() -> None:
    """Guard the height generalization: no bare 20/200 board sizes in src.

    Allowed: DEFAULT_HEIGHT's definition, prose naming the default in
    docstrings/comments, and the demo's --pieces count (200 pieces, not
    200 cells).
    """
    import re
    from pathlib import Path

    import tetris_coach

    src = Path(tetris_coach.__file__).parent
    pattern = re.compile(r"\b(20|200)\b")
    allowed = (
        "DEFAULT_HEIGHT = 20",  # the one definition
        "default 20",  # docstring/comment prose about the default
        "default=200",  # cli --pieces: demo piece count, not a board size
        "default {DEFAULT_HEIGHT}",  # cli --rows help text
    )
    offenders: list[str] = []
    for path in sorted(src.rglob("*.py")):
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if pattern.search(line) and not any(mark in line for mark in allowed):
                offenders.append(f"{path.relative_to(src)}:{lineno}: {line.strip()}")
    assert not offenders, "bare board-size literals in src:\n" + "\n".join(offenders)
