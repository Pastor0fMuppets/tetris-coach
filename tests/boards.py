"""Shared board-construction helpers for vision tests (row-bitmask based)."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from tetris_coach.core.board import HEIGHT, WIDTH
from tetris_coach.core.pieces import ROTATIONS
from tetris_coach.vision.pieces_vision import FallingPiece

Cell = tuple[int, int]

EMPTY: tuple[int, ...] = (0,) * HEIGHT


def rows_of(*cell_groups: Iterable[Cell]) -> tuple[int, ...]:
    """Row bitmasks with every cell of every group set."""
    rows = [0] * HEIGHT
    for cells in cell_groups:
        for r, c in cells:
            rows[r] |= 1 << c
    return tuple(rows)


def merge(rows: tuple[int, ...], *cell_groups: Iterable[Cell]) -> tuple[int, ...]:
    """``rows`` with extra cells merged in."""
    out = list(rows)
    for cells in cell_groups:
        for r, c in cells:
            out[r] |= 1 << c
    return tuple(out)


def bottom_lines(*lines: str) -> list[Cell]:
    """Cells of an ASCII stack anchored to the bottom of the board."""
    cells: list[Cell] = []
    for i, line in enumerate(lines):
        r = HEIGHT - len(lines) + i
        for c, ch in enumerate(line):
            if ch == "#":
                cells.append((r, c))
    return cells


def piece_cells(piece: str, rotation_index: int, row: int, col: int) -> list[Cell]:
    """Absolute cells of a piece rotation placed at (row, col)."""
    return [(row + r, col + c) for r, c in ROTATIONS[piece][rotation_index].cells]


def fp(piece: str, rotation_index: int, row: int, col: int) -> FallingPiece:
    return FallingPiece(piece=piece, rotation_index=rotation_index, row=row, col=col)


def grid_of(rows: tuple[int, ...]) -> np.ndarray:
    """Row bitmasks -> (HEIGHT, WIDTH) bool occupancy grid."""
    grid = np.zeros((HEIGHT, WIDTH), dtype=bool)
    for r, row in enumerate(rows):
        for c in range(WIDTH):
            if row >> c & 1:
                grid[r, c] = True
    return grid
