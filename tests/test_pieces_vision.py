import numpy as np
import pytest

from tetris_coach.core.pieces import PIECES, ROTATIONS
from tetris_coach.vision.grid import classify_grid
from tetris_coach.vision.pieces_vision import identify_next, match_cells

from .synthetic import STYLES, render_next_preview


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
