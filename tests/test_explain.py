"""Unit tests for explain_grid: diff-anchored frame classification."""

import pytest

from tetris_coach.core.board import Board
from tetris_coach.core.pieces import PIECES, ROTATIONS
from tetris_coach.vision.pieces_vision import (
    FrameKind,
    clear_full_rows,
    explain_grid,
)

from .boards import EMPTY, bottom_lines, fp, merge, piece_cells, rows_of


class TestQuiet:
    def test_quiet_idle(self) -> None:
        stack = rows_of(bottom_lines("####..####"))
        exp = explain_grid(stack, stack, None)
        assert exp.kind is FrameKind.QUIET
        assert exp.stack_rows == stack
        assert exp.falling is None

    def test_quiet_empty_board(self) -> None:
        exp = explain_grid(EMPTY, EMPTY, None)
        assert exp.kind is FrameKind.QUIET
        assert exp.stack_rows == EMPTY


class TestFlightAnywhere:
    """F1: the falling piece is FALLING wherever it is — no support rules."""

    def test_piece_resting_on_stack(self) -> None:
        stack = rows_of(bottom_lines("#########."))
        t = piece_cells("T", 0, 17, 3)  # resting directly on the stack
        exp = explain_grid(merge(stack, t), stack, fp("T", 0, 16, 3))
        assert exp.kind is FrameKind.FALLING
        assert exp.stack_rows == stack
        assert exp.falling is not None
        assert exp.falling.piece == "T"
        assert (exp.falling.row, exp.falling.col) == (17, 3)

    def test_piece_on_floor_of_empty_column(self) -> None:
        # The old floor-touch rule merged this piece into the stack.
        stack = rows_of([(18, 0), (19, 0)])
        o = piece_cells("O", 0, 18, 4)
        exp = explain_grid(merge(stack, o), stack, None)
        assert exp.kind is FrameKind.FALLING
        assert exp.falling is not None
        assert exp.falling.piece == "O"
        assert exp.stack_rows == stack

    def test_piece_leaning_against_tall_column(self) -> None:
        stack = rows_of([(r, 0) for r in range(10, 20)])
        i = piece_cells("I", 1, 12, 1)  # side-adjacent, mid-air
        exp = explain_grid(merge(stack, i), stack, None)
        assert exp.kind is FrameKind.FALLING
        assert exp.falling is not None
        assert exp.falling.piece == "I"
        assert exp.stack_rows == stack

    def test_sliding_during_lock_delay(self) -> None:
        stack = rows_of(bottom_lines("#########."))
        for col in (3, 4, 5):
            t = piece_cells("T", 0, 17, col)
            exp = explain_grid(merge(stack, t), stack, fp("T", 0, 17, col - 1))
            assert exp.kind is FrameKind.FALLING
            assert exp.stack_rows == stack
            assert exp.falling is not None
            assert exp.falling.col == col


class TestDebrisStaysStack:
    """F3: floating debris lives inside the committed stack, never falls."""

    def test_real_piece_above_island(self) -> None:
        island = rows_of(piece_cells("O", 0, 10, 7))
        t = piece_cells("T", 0, 2, 3)
        exp = explain_grid(merge(island, t), island, None)
        assert exp.kind is FrameKind.FALLING
        assert exp.falling is not None
        assert exp.falling.piece == "T"
        assert exp.stack_rows == island

    def test_real_piece_below_island(self) -> None:
        # The old topmost-component rule fails exactly here: the island is
        # above the real piece and would have been picked as falling.
        island = rows_of(piece_cells("O", 0, 10, 7))
        t = piece_cells("T", 0, 15, 6)
        exp = explain_grid(merge(island, t), island, None)
        assert exp.kind is FrameKind.FALLING
        assert exp.falling is not None
        assert exp.falling.piece == "T"
        assert exp.falling.row == 15
        assert exp.stack_rows == island


class TestLockRevealL1:
    def test_lock_plus_adjacent_spawn_one_blob(self) -> None:
        # Tall stack: the locked piece is adjacent to the spawn (L2 cannot
        # split the single 8-cell blob; L1 must anchor on the last position).
        stack = rows_of([(r, 3) for r in range(6, 20)])
        last = fp("O", 0, 4, 3)  # at rest on the tower
        spawn = piece_cells("I", 0, 3, 3)  # 4-adjacent to the O
        observed = merge(stack, last.cells, spawn)
        exp = explain_grid(observed, stack, last)
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == merge(stack, last.cells)
        assert exp.falling is not None
        assert exp.falling.piece == "I"
        assert (exp.falling.row, exp.falling.col) == (3, 3)


class TestLockRevealL2:
    STACK = rows_of(bottom_lines("###....###"))

    def test_hard_drop_lock_with_spawn(self) -> None:
        # Zero-ARE: last observed mid-air, locked cells are NOT those cells.
        last = fp("T", 0, 1, 3)
        lock = piece_cells("T", 0, 18, 3)
        spawn = piece_cells("J", 0, 0, 4)
        exp = explain_grid(merge(self.STACK, lock, spawn), self.STACK, last)
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == merge(self.STACK, lock)
        assert exp.falling is not None
        assert exp.falling.piece == "J"

    def test_no_prior_observation_name_unconstrained(self) -> None:
        lock = piece_cells("T", 0, 18, 3)
        spawn = piece_cells("J", 0, 0, 4)
        exp = explain_grid(merge(self.STACK, lock, spawn), self.STACK, None)
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == merge(self.STACK, lock)
        assert exp.falling is not None
        assert exp.falling.piece == "J"

    def _towers(self) -> tuple[int, ...]:
        left = [(r, c) for r in range(4, 20) for c in (0, 1, 2, 3)]
        right = [(r, c) for r in range(4, 20) for c in (6, 7, 8, 9)]
        return rows_of(left, right)

    def test_both_in_spawn_zone_name_tiebreak(self) -> None:
        # Near top-out: both components supported and in the spawn zone;
        # the piece-name constraint disambiguates.
        stack = self._towers()
        lock_o = piece_cells("O", 0, 2, 0)  # rests on the left tower
        spawn_s = piece_cells("S", 0, 2, 6)  # rests on the right tower
        last = fp("O", 0, 2, 3)  # mid-gap: not a subset of added (L1 fails)
        exp = explain_grid(merge(stack, lock_o, spawn_s), stack, last)
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == merge(stack, lock_o)
        assert exp.falling is not None
        assert exp.falling.piece == "S"

    def test_same_name_intersection_tiebreak(self) -> None:
        stack = self._towers()
        left_o = piece_cells("O", 0, 2, 0)
        right_o = piece_cells("O", 0, 2, 6)
        # Overlaps the left component only (and partly the stack, so L1's
        # subset test fails and L2 must break the tie by intersection).
        last = fp("O", 0, 3, 1)
        exp = explain_grid(merge(stack, left_o, right_o), stack, last)
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == merge(stack, left_o)
        assert exp.falling is not None
        assert exp.falling.piece == "O"
        assert exp.falling.col == 6

    def test_same_name_no_overlap_is_unexplained(self) -> None:
        stack = self._towers()
        left_o = piece_cells("O", 0, 2, 0)
        right_o = piece_cells("O", 0, 2, 6)
        last = fp("O", 0, 2, 4)  # mid-gap: intersects neither component
        exp = explain_grid(merge(stack, left_o, right_o), stack, last)
        assert exp.kind is FrameKind.UNEXPLAINED


class TestLockRevealRejectsFullRows:
    """A zero-ARE clear-flash frame — the completed row still fully lit
    while the next piece is already visible — must stay UNEXPLAINED:
    committing it would anchor the tracker on a stack containing a row
    that is about to vanish."""

    STACK = rows_of(bottom_lines("#########."))
    SPAWN = piece_cells("J", 0, 0, 4)

    def test_l1_flash_frame_with_spawn_unexplained(self) -> None:
        last = fp("I", 1, 16, 9)  # observed at rest, completing row 19
        observed = merge(self.STACK, last.cells, self.SPAWN)
        assert explain_grid(observed, self.STACK, last).kind is FrameKind.UNEXPLAINED

    def test_l2_flash_frame_with_spawn_unexplained(self) -> None:
        last = fp("I", 1, 4, 9)  # hard drop: last observed mid-air
        lock = piece_cells("I", 1, 16, 9)
        observed = merge(self.STACK, lock, self.SPAWN)
        assert explain_grid(observed, self.STACK, last).kind is FrameKind.UNEXPLAINED

    def test_settled_post_clear_frame_still_locks(self) -> None:
        # Once the animation settles, the collapsed board plus the
        # (descended) spawn is explained by the clear tiers as usual.
        last = fp("I", 1, 16, 9)
        s2 = rows_of([(17, 9), (18, 9), (19, 9)])
        observed = merge(s2, piece_cells("J", 0, 1, 4))
        exp = explain_grid(observed, self.STACK, last)
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == s2
        assert exp.falling is not None
        assert exp.falling.piece == "J"


class TestClearTierC1:
    def test_last_position_completes_row(self) -> None:
        stack = rows_of(bottom_lines("#########."))
        last = fp("I", 1, 16, 9)  # vertical I resting, completing row 19
        s2 = rows_of([(17, 9), (18, 9), (19, 9)])
        exp = explain_grid(s2, stack, last)
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == s2
        assert exp.falling is None

    def test_with_residual_spawn(self) -> None:
        stack = rows_of(bottom_lines("#########."))
        last = fp("I", 1, 16, 9)
        s2 = rows_of([(17, 9), (18, 9), (19, 9)])
        spawn = piece_cells("J", 0, 0, 4)
        exp = explain_grid(merge(s2, spawn), stack, last)
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == s2
        assert exp.falling is not None
        assert exp.falling.piece == "J"

    def test_t_spin_double_not_gravity_reachable(self) -> None:
        stack = rows_of(
            [(17, 2)],
            [(18, c) for c in (0, 4, 5, 6, 7, 8, 9)],
            [(19, c) for c in (0, 1, 3, 4, 5, 6, 7, 8, 9)],
        )
        last = fp("T", 2, 18, 1)  # point-down, tucked under the roof at (17, 2)
        # Sanity: gravity cannot reach the observed position (the roof).
        dropped = Board(stack).drop(ROTATIONS["T"][2], 1)
        assert dropped is not None and dropped.landing_row != 18
        s2 = rows_of([(19, 2)])  # double clear collapses everything else
        exp = explain_grid(s2, stack, last)
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == s2
        assert exp.falling is None


class TestClearTierC2:
    def test_gravity_drop_from_last_alignment(self) -> None:
        stack = rows_of(bottom_lines("#########."))
        last = fp("I", 1, 5, 9)  # mid-air; the drop lands and clears row 19
        s2 = rows_of([(17, 9), (18, 9), (19, 9)])
        exp = explain_grid(s2, stack, last)
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == s2
        assert exp.falling is None

    def test_last_instant_shift_found_by_c3(self) -> None:
        # The piece shifted a column after the last observation: C2's drop
        # from the stale alignment misses; C3 retries the same name.
        stack = rows_of(bottom_lines("#########."))
        last = fp("I", 1, 5, 8)
        s2 = rows_of([(17, 9), (18, 9), (19, 9)])
        exp = explain_grid(s2, stack, last)
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == s2
        assert exp.falling is None


class TestClearTierC3Unobserved:
    """F2: a line-clearing lock by a never-observed piece is LOCKED, never
    anything reset-shaped."""

    def test_single_clear(self) -> None:
        stack = rows_of(bottom_lines("#########."))
        s2 = rows_of([(17, 9), (18, 9), (19, 9)])
        exp = explain_grid(s2, stack, None)
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == s2
        assert exp.falling is None

    def test_double_clear_to_empty_board(self) -> None:
        stack = rows_of(bottom_lines("########..", "########.."))
        exp = explain_grid(EMPTY, stack, None)
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == EMPTY
        assert exp.falling is None

    def test_quad_clear_to_empty_board(self) -> None:
        stack = rows_of(bottom_lines("#########.", "#########.", "#########.", "#########."))
        exp = explain_grid(EMPTY, stack, None)
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == EMPTY
        assert exp.falling is None

    def test_clear_with_simultaneous_spawn(self) -> None:
        stack = rows_of(bottom_lines("#########."))
        s2 = rows_of([(17, 9), (18, 9), (19, 9)])
        spawn = piece_cells("S", 0, 1, 4)
        exp = explain_grid(merge(s2, spawn), stack, None)
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == s2
        assert exp.falling is not None
        assert exp.falling.piece == "S"


class TestUnexplained:
    STACK = rows_of(bottom_lines("###....###"))

    @pytest.mark.parametrize("n_cells", [1, 2, 3])
    def test_piece_entering_from_above(self, n_cells: int) -> None:
        entering = [(0, 4 + i) for i in range(n_cells)]
        exp = explain_grid(merge(self.STACK, entering), self.STACK, None)
        assert exp.kind is FrameKind.UNEXPLAINED

    def test_five_cell_blob(self) -> None:
        blob = [(5, 2), (5, 3), (5, 4), (6, 3), (6, 4)]
        exp = explain_grid(merge(self.STACK, blob), self.STACK, None)
        assert exp.kind is FrameKind.UNEXPLAINED

    def test_three_components(self) -> None:
        observed = merge(
            self.STACK,
            piece_cells("O", 0, 2, 1),
            piece_cells("O", 0, 2, 5),
            piece_cells("O", 0, 10, 7),
        )
        assert explain_grid(observed, self.STACK, None).kind is FrameKind.UNEXPLAINED

    def test_eight_cells_failing_l1_and_l2(self) -> None:
        # Two mid-air tetrominoes: neither supported, neither in the zone.
        observed = merge(self.STACK, piece_cells("I", 0, 10, 2), piece_cells("I", 0, 12, 5))
        assert explain_grid(observed, self.STACK, None).kind is FrameKind.UNEXPLAINED

    def test_missing_cells_without_clear_explanation(self) -> None:
        stack = rows_of(bottom_lines("#########."))
        observed = tuple(
            row & ~sum(1 << c for c in (2, 3, 4)) if r == 19 else row for r, row in enumerate(stack)
        )
        assert explain_grid(observed, stack, None).kind is FrameKind.UNEXPLAINED

    def test_piece_plus_same_named_ghost(self) -> None:
        # A bright ghost piece at the bottom with the real piece mid-air
        # must hold as UNEXPLAINED, not commit a false lock.
        stack = rows_of(bottom_lines("###...####"))
        real = piece_cells("T", 0, 5, 3)
        ghost = piece_cells("T", 0, 18, 3)
        last = fp("T", 0, 4, 3)
        exp = explain_grid(merge(stack, real, ghost), stack, last)
        assert exp.kind is FrameKind.UNEXPLAINED


class TestOcclusionTolerance:
    STACK = rows_of(bottom_lines("#########."))

    def _occlude(self, rows: tuple[int, ...], cells: list[tuple[int, int]]) -> tuple[int, ...]:
        out = list(rows)
        for r, c in cells:
            out[r] &= ~(1 << c)
        return tuple(out)

    def test_falling_with_two_missing_keeps_stack_memory(self) -> None:
        observed = self._occlude(merge(self.STACK, piece_cells("T", 0, 5, 3)), [(19, 0), (19, 1)])
        exp = explain_grid(observed, self.STACK, None)
        assert exp.kind is FrameKind.FALLING
        assert exp.stack_rows == self.STACK  # memory wins over vision
        assert exp.falling is not None
        assert exp.falling.piece == "T"

    def test_quiet_with_one_missing(self) -> None:
        observed = self._occlude(self.STACK, [(19, 4)])
        exp = explain_grid(observed, self.STACK, None)
        assert exp.kind is FrameKind.QUIET
        assert exp.stack_rows == self.STACK

    def test_three_missing_is_unexplained(self) -> None:
        observed = self._occlude(self.STACK, [(19, 0), (19, 1), (19, 2)])
        assert explain_grid(observed, self.STACK, None).kind is FrameKind.UNEXPLAINED

    def test_tolerance_never_applies_to_lock_reveal(self) -> None:
        stack = rows_of(bottom_lines("###....###"))
        lock = piece_cells("T", 0, 18, 3)
        spawn = piece_cells("J", 0, 0, 4)
        observed = self._occlude(merge(stack, lock, spawn), [(19, 0), (19, 1)])
        exp = explain_grid(observed, stack, fp("T", 0, 18, 3))
        assert exp.kind is FrameKind.UNEXPLAINED


class TestClearFullRowsMatchesCore:
    @pytest.mark.parametrize("piece", PIECES)
    def test_parity_with_board_drop(self, piece: str) -> None:
        stacks = (
            rows_of(bottom_lines("#########.")),
            rows_of(bottom_lines("########..", "########..")),
            rows_of(bottom_lines("#####.....")),
        )
        for stack in stacks:
            board = Board(stack)
            for rotation in ROTATIONS[piece]:
                for col in range(10 - rotation.width + 1):
                    dropped = board.drop(rotation, col)
                    if dropped is None:
                        continue
                    merged = merge(
                        stack,
                        [(dropped.landing_row + r, col + c) for r, c in rotation.cells],
                    )
                    assert clear_full_rows(merged) == dropped.board.rows
