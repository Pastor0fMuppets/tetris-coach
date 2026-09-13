import pytest

from tetris_coach.core.board import DEFAULT_HEIGHT, FULL_ROW, HEIGHT, WIDTH, Board
from tetris_coach.core.pieces import ROTATIONS


def board_from_strings(*lines: str) -> Board:
    """Build a board from drawing strings; missing top rows are empty.

    '#' (or any non-'.'/' ') marks a filled cell.
    """
    assert len(lines) <= HEIGHT
    grid = [[False] * WIDTH for _ in range(HEIGHT - len(lines))]
    for line in lines:
        assert len(line) == WIDTH
        grid.append([ch not in ". " for ch in line])
    return Board.from_grid(grid)


def rot(piece: str, index: int):
    return ROTATIONS[piece][index]


def horizontal_i():
    return next(r for r in ROTATIONS["I"] if r.width == 4)


def vertical_i():
    return next(r for r in ROTATIONS["I"] if r.width == 1)


class TestConstruction:
    def test_empty_board(self) -> None:
        b = Board()
        assert b.rows == (0,) * HEIGHT
        assert b.heights == (0,) * WIDTH
        assert b.cell_count() == 0

    def test_from_grid_round_trip(self) -> None:
        grid = [[False] * WIDTH for _ in range(HEIGHT)]
        grid[19][0] = True
        grid[19][9] = True
        grid[10][5] = True
        b = Board.from_grid(grid)
        assert b.to_grid() == grid
        assert b.cell_count() == 3

    def test_from_grid_validates_shape(self) -> None:
        with pytest.raises(ValueError):
            Board.from_grid([])  # no rows at all
        with pytest.raises(ValueError):
            Board.from_grid([[False] * 9] * HEIGHT)  # width is fixed at 10

    def test_immutability(self) -> None:
        b = Board()
        with pytest.raises(AttributeError):
            b.rows = (0,) * HEIGHT  # type: ignore[misc]

    def test_equality_and_hash(self) -> None:
        a = board_from_strings("#.........")
        b = board_from_strings("#.........")
        assert a == b
        assert hash(a) == hash(b)
        assert a != Board()


class TestHeights:
    def test_heights_from_rows(self) -> None:
        b = board_from_strings(
            "#.........",
            "#...#.....",
            "##..#....#",
        )
        assert b.heights == (3, 1, 0, 0, 2, 0, 0, 0, 0, 1)

    def test_heights_ignore_holes(self) -> None:
        b = board_from_strings(
            "#.........",
            "..........",
            "#.........",
        )
        assert b.heights[0] == 3


class TestDrop:
    def test_drop_on_empty_board(self) -> None:
        res = Board().drop(horizontal_i(), 0)
        assert res is not None
        assert res.lines_cleared == 0
        assert res.landing_row == HEIGHT - 1
        assert res.board.rows[HEIGHT - 1] == 0b1111
        assert res.board.heights == (1, 1, 1, 1, 0, 0, 0, 0, 0, 0)

    def test_drop_lands_on_stack(self) -> None:
        b = board_from_strings(
            "..##......",
            "..##......",
        )
        res = b.drop(rot("O", 0), 2)
        assert res is not None
        assert res.landing_row == HEIGHT - 4
        assert res.board.heights[2] == 4
        assert res.board.heights[3] == 4

    def test_drop_respects_highest_column(self) -> None:
        b = board_from_strings(
            ".....#....",
            ".....#....",
            ".....#....",
            "..........",
        )
        # Horizontal I over columns 3-6: column 5 has height 4 (rows 16-18
        # filled above an empty bottom row), so the bar rests on top of it.
        res = b.drop(horizontal_i(), 3)
        assert res is not None
        assert res.landing_row == HEIGHT - 5
        assert res.board.heights[3] == 5
        assert res.board.heights[4] == 5
        assert res.board.heights[5] == 5
        assert res.board.heights[6] == 5
        # Columns 3, 4, 6 now have holes under the bar.
        assert res.board.hole_count() == 3 * 4 + 1

    def test_drop_creates_overhang_hole(self) -> None:
        b = board_from_strings(
            "#.........",
            "#.........",
        )
        # S piece (bottom row cols 0-1, top row cols 1-2) rests on the col-0
        # tower: col 1 traps 2 empty cells below it, col 2 traps 3.
        res = b.drop(rot("S", 0), 0)
        assert res is not None
        assert res.landing_row == HEIGHT - 4
        assert res.board.hole_count() == 5

    def test_drop_out_of_range(self) -> None:
        assert Board().drop(horizontal_i(), 6) is not None
        assert Board().drop(horizontal_i(), 7) is None
        assert Board().drop(horizontal_i(), -1) is None

    def test_drop_top_out(self) -> None:
        rows = [FULL_ROW ^ 1] * HEIGHT  # column 0 empty, everything else full
        b = Board(rows)
        assert b.heights[1] == HEIGHT
        assert b.drop(rot("O", 0), 0) is None
        # A vertical I in the empty column still fits.
        res = b.drop(vertical_i(), 0)
        assert res is not None
        assert res.lines_cleared == 4

    def test_single_line_clear(self) -> None:
        b = board_from_strings("######....")
        res = b.drop(horizontal_i(), 6)
        assert res is not None
        assert res.lines_cleared == 1
        assert res.eroded_cells == 4
        assert res.board == Board()

    def test_multi_line_clear_shifts_rows(self) -> None:
        b = board_from_strings(
            "#.........",
            "#########.",
            "#########.",
        )
        # Vertical I in column 9 clears the bottom two rows; leaves two cells.
        res = b.drop(vertical_i(), 9)
        assert res is not None
        assert res.lines_cleared == 2
        assert res.eroded_cells == 2
        # The I fills rows 16-19 in column 9; rows 18-19 clear. The surviving
        # rows keep their order: row 16 (I cell only) above row 17 (old col-0
        # cell plus an I cell).
        expected = board_from_strings(
            ".........#",
            "#........#",
        )
        assert res.board == expected

    def test_cleared_board_heights_recomputed(self) -> None:
        b = board_from_strings(
            "#.........",
            "######....",
        )
        res = b.drop(horizontal_i(), 6)
        assert res is not None
        assert res.lines_cleared == 1
        assert res.board.heights == (1, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    def test_drop_does_not_mutate_original(self) -> None:
        b = Board()
        b.drop(rot("O", 0), 0)
        assert b == Board()


class TestFeatures:
    def test_empty_board_features(self) -> None:
        b = Board()
        assert b.hole_count() == 0
        assert b.row_transitions() == 2 * HEIGHT  # every empty row counts 2
        assert b.column_transitions() == WIDTH  # floor transition per column
        assert b.cumulative_wells() == 0

    def test_full_bottom_row_features(self) -> None:
        b = board_from_strings("##########")
        assert b.row_transitions() == 2 * (HEIGHT - 1)
        assert b.column_transitions() == WIDTH

    def test_holes(self) -> None:
        b = board_from_strings(
            "###.......",
            "..........",
            "#.#.......",
        )
        # col 0: 1 hole (middle row); col 1: 2 holes; col 2: 1 hole.
        assert b.hole_count() == 4

    def test_row_transitions_simple(self) -> None:
        b = board_from_strings(".#.#.#.#.#")
        # Bottom row alternates: transitions at every boundary incl. walls.
        assert b.row_transitions() == 10 + 2 * (HEIGHT - 1)

    def test_column_transitions_simple(self) -> None:
        b = board_from_strings(
            "#.........",
            "..........",
            "#.........",
        )
        # Column 0 top-down: empty..., #, ., #, floor(filled) -> 3 transitions.
        # Other 9 columns: 1 floor transition each.
        assert b.column_transitions() == 3 + 9

    def test_cumulative_wells_single_deep_well(self) -> None:
        b = board_from_strings(
            "#.#.......",
            "#.#.......",
            "#.#.......",
        )
        # Column 1 is a well of depth 3: 3 + 2 + 1.
        assert b.cumulative_wells() == 6

    def test_cumulative_wells_wall_counts_as_filled(self) -> None:
        b = board_from_strings(
            ".#........",
            ".#........",
        )
        # Column 0 against the left wall: depth-2 well -> 2 + 1.
        assert b.cumulative_wells() == 3

    def test_well_open_below(self) -> None:
        b = board_from_strings(
            "#.#.......",
            "#.........",
            "#.........",
        )
        # One well cell at the top of column 1 with two empty cells below:
        # contributes 1 + 2 = 3 (Fahey counting).
        assert b.cumulative_wells() == 3

    def test_features_after_i_drop_match_known_values(self) -> None:
        res = Board().drop(vertical_i(), 0)
        assert res is not None
        b = res.board
        assert b.heights == (4, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        assert b.hole_count() == 0
        # Rows 16-19 each have one filled cell at col 0 -> 2 transitions/row,
        # empty rows contribute 2 each: total unchanged from empty board.
        assert b.row_transitions() == 2 * HEIGHT
        # Column 0: one empty->filled transition, then filled to the floor;
        # the other 9 columns keep their single floor transition.
        assert b.column_transitions() == WIDTH
        # Column 1 cells have the stack on the left but nothing on the right,
        # so they are not well cells.
        assert b.cumulative_wells() == 0


class TestNonDefaultHeight:
    """12-row boards: height flows from len(rows), never from a constant."""

    ROWS = 12

    def test_default_height_is_the_alias(self) -> None:
        assert HEIGHT == DEFAULT_HEIGHT == 20
        assert len(Board().rows) == DEFAULT_HEIGHT

    def test_construction_and_heights(self) -> None:
        b = Board([0] * self.ROWS)
        assert len(b.rows) == self.ROWS
        assert b.heights == (0,) * WIDTH
        stacked = Board([1] + [0] * (self.ROWS - 1))  # col 0 filled at row 0
        assert stacked.heights[0] == self.ROWS

    def test_from_grid_derives_height(self) -> None:
        grid = [[False] * WIDTH for _ in range(self.ROWS)]
        grid[self.ROWS - 1][3] = True
        b = Board.from_grid(grid)
        assert len(b.rows) == self.ROWS
        assert b.to_grid() == grid
        assert b.heights[3] == 1

    def test_drop_to_floor(self) -> None:
        res = Board([0] * self.ROWS).drop(horizontal_i(), 0)
        assert res is not None
        assert res.landing_row == self.ROWS - 1
        assert res.board.rows[self.ROWS - 1] == 0b1111
        assert res.board.heights == (1, 1, 1, 1, 0, 0, 0, 0, 0, 0)

    def test_drop_onto_stack_every_height(self) -> None:
        # Stack a column to every height including landing at row 0; the
        # cached heights must stay exact after each no-clear fast path.
        b = Board([0] * self.ROWS)
        for i in range(self.ROWS // 4):
            res = b.drop(vertical_i(), 0)
            assert res is not None
            assert res.landing_row == self.ROWS - 4 * (i + 1)
            b = res.board
            assert b.heights[0] == 4 * (i + 1)
            assert b.heights == Board(b.rows).heights  # cache == recompute
        assert b.heights[0] == self.ROWS
        assert b.drop(vertical_i(), 0) is None  # column is full: top-out

    def test_top_out(self) -> None:
        rows = [FULL_ROW ^ 1] * self.ROWS  # column 0 empty, rest full
        b = Board(rows)
        assert b.heights[1] == self.ROWS
        assert b.drop(rot("O", 0), 0) is None
        res = b.drop(vertical_i(), 0)
        assert res is not None
        assert res.lines_cleared == 4

    def test_clear_preserves_row_count(self) -> None:
        b = Board([0] * (self.ROWS - 1) + [FULL_ROW ^ 0b1111])
        res = b.drop(horizontal_i(), 0)
        assert res is not None
        assert res.lines_cleared == 1
        assert len(res.board.rows) == self.ROWS
        assert res.board == Board([0] * self.ROWS)

    def test_features_on_12_row_board(self) -> None:
        b = Board([0] * self.ROWS)
        assert b.hole_count() == 0
        assert b.row_transitions() == 2 * self.ROWS
        assert b.column_transitions() == WIDTH
        assert b.cumulative_wells() == 0
        # A depth-3 well between two towers, anchored to the 12-row floor.
        tower = 0b101
        wells = Board([0] * (self.ROWS - 3) + [tower] * 3)
        assert wells.cumulative_wells() == 6  # 3 + 2 + 1
        holes = Board([0] * (self.ROWS - 2) + [1, 0])  # covered empty cell
        assert holes.hole_count() == 1
        assert holes.column_transitions() == 2 + WIDTH  # col 0: in, out, floor

    def test_rejects_empty_rows(self) -> None:
        with pytest.raises(ValueError):
            Board([])
