import numpy as np
import pytest

from tetris_coach.core.pieces import PIECES, ROTATIONS
from tetris_coach.vision.grid import classify_grid
from tetris_coach.vision.pieces_vision import (
    identify_falling,
    identify_next,
    match_cells,
    split_grid,
)

from .synthetic import STYLES, render_next_preview


def grid_with(cells: list[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros((20, 10), dtype=bool)
    for r, c in cells:
        grid[r, c] = True
    return grid


def stack_rows(*lines: str) -> list[tuple[int, int]]:
    cells = []
    for i, line in enumerate(lines):
        r = 20 - len(lines) + i
        for c, ch in enumerate(line):
            if ch == "#":
                cells.append((r, c))
    return cells


class TestMatchCells:
    def test_all_rotations_match(self) -> None:
        for piece, rots in ROTATIONS.items():
            for rot in rots:
                # Shifted anywhere on the board, the shape still matches.
                cells = {(r + 5, c + 3) for r, c in rot.cells}
                assert match_cells(cells) == (piece, rot.index)

    def test_non_tetromino_shapes_do_not_match(self) -> None:
        assert match_cells({(0, 0), (0, 1), (0, 2)}) is None  # 3 cells
        assert match_cells({(0, 0), (0, 1), (1, 0), (2, 2)}) is None
        assert match_cells({(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)}) is None


class TestSplitGrid:
    def test_falling_piece_separated_from_stack(self) -> None:
        t_cells = [(2, 4), (3, 3), (3, 4), (3, 5)]  # T pointing up
        stack = stack_rows(
            "###...####",
            "#####.####",
        )
        grid = grid_with(t_cells + stack)
        stack_grid, falling = split_grid(grid)
        assert falling is not None
        assert falling.piece == "T"
        assert (falling.row, falling.col) == (2, 3)
        np.testing.assert_array_equal(stack_grid, grid_with(stack))

    def test_no_falling_piece(self) -> None:
        grid = grid_with(stack_rows("####......"))
        stack_grid, falling = split_grid(grid)
        assert falling is None
        np.testing.assert_array_equal(stack_grid, grid)

    def test_piece_touching_floor_is_stack(self) -> None:
        # An O sitting on the floor is part of the stack, not falling.
        grid = grid_with([(18, 4), (18, 5), (19, 4), (19, 5)])
        _, falling = split_grid(grid)
        assert falling is None

    def test_floating_non_tetromino_debris_is_stack(self) -> None:
        # 5-cell floating blob (e.g. mid-clear animation artifact).
        blob = [(5, 2), (5, 3), (5, 4), (6, 3), (6, 4)]
        grid = grid_with(blob)
        stack_grid, falling = split_grid(grid)
        assert falling is None
        np.testing.assert_array_equal(stack_grid, grid)

    def test_topmost_candidate_wins(self) -> None:
        high_i = [(1, 0), (1, 1), (1, 2), (1, 3)]
        floating_o = [(8, 7), (8, 8), (9, 7), (9, 8)]  # floating overhang debris
        grid = grid_with(high_i + floating_o)
        stack_grid, falling = split_grid(grid)
        assert falling is not None
        assert falling.piece == "I"
        # The O stays in the stack grid.
        np.testing.assert_array_equal(stack_grid, grid_with(floating_o))

    def test_empty_grid(self) -> None:
        grid = np.zeros((20, 10), dtype=bool)
        stack_grid, falling = split_grid(grid)
        assert falling is None
        assert not stack_grid.any()


class TestIdentifyFalling:
    @pytest.mark.parametrize("piece", PIECES)
    def test_every_piece_recognized(self, piece: str) -> None:
        for rot in ROTATIONS[piece]:
            cells = [(r + 1, c + 3) for r, c in rot.cells]
            grid = grid_with(cells + stack_rows("####.#####"))
            falling = identify_falling(grid)
            assert falling is not None
            assert falling.piece == piece
            assert falling.rotation_index == rot.index
            assert (falling.row, falling.col) == (1, 3)


class TestIdentifyNext:
    @pytest.mark.parametrize("style", STYLES, ids=lambda s: s.name)
    @pytest.mark.parametrize("piece", PIECES)
    def test_spawn_orientation_recognized(self, piece: str, style) -> None:  # type: ignore[no-untyped-def]
        image = render_next_preview(ROTATIONS[piece][0].cells, style, cell_size=20)
        assert identify_next(image) == piece

    @pytest.mark.parametrize("piece", PIECES)
    def test_any_rotation_recognized(self, piece: str) -> None:
        for rot in ROTATIONS[piece]:
            image = render_next_preview(rot.cells, STYLES[0], cell_size=16)
            assert identify_next(image) == piece

    @pytest.mark.parametrize("cell_size", [10, 18, 27])
    def test_various_cell_sizes(self, cell_size: int) -> None:
        image = render_next_preview(ROTATIONS["L"][0].cells, STYLES[2], cell_size=cell_size)
        assert identify_next(image) == "L"

    def test_empty_preview_returns_none(self) -> None:
        image = np.full((80, 120, 3), 12, dtype=np.uint8)
        assert identify_next(image) is None

    def test_preview_pipeline_matches_grid_pipeline(self) -> None:
        # Sanity: classify_grid and identify_next agree on what is foreground.
        image = render_next_preview(ROTATIONS["O"][0].cells, STYLES[1], cell_size=22)
        occupancy, _ = classify_grid(image, rows=4, cols=6)
        assert occupancy.sum() == 4
        assert identify_next(image) == "O"
