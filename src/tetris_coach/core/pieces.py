"""Tetromino definitions.

Each of the 7 pieces is expanded into its distinct rotations. Every rotation
carries precomputed data for O(1) drop-height computation on a bitboard:

- ``row_masks``: for each row of the rotation's bounding box, a bitmask of the
  occupied columns (bit ``c`` set means relative column ``c`` is occupied).
- ``bottom``: for each occupied column, the row offset of the lowest cell in
  that column (used to rest the piece on the stack surface).
- ``top``: for each occupied column, the row offset of the highest cell in
  that column (used to update column heights without rescanning).

Rows are indexed top-down (row 0 is the top of the bounding box), matching the
board convention.
"""

from __future__ import annotations

from dataclasses import dataclass

Cell = tuple[int, int]  # (row, col), row 0 = top


@dataclass(frozen=True)
class Rotation:
    """One distinct orientation of a tetromino."""

    piece: str
    index: int
    cells: tuple[Cell, ...]
    width: int
    height: int
    row_masks: tuple[int, ...]
    bottom: tuple[int, ...]
    top: tuple[int, ...]


# Spawn orientations; cells as (row, col) with row 0 on top.
_BASE_CELLS: dict[str, tuple[Cell, ...]] = {
    "I": ((0, 0), (0, 1), (0, 2), (0, 3)),
    "O": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "T": ((0, 1), (1, 0), (1, 1), (1, 2)),
    "S": ((0, 1), (0, 2), (1, 0), (1, 1)),
    "Z": ((0, 0), (0, 1), (1, 1), (1, 2)),
    "J": ((0, 0), (1, 0), (1, 1), (1, 2)),
    "L": ((0, 2), (1, 0), (1, 1), (1, 2)),
}


def _normalize(cells: tuple[Cell, ...]) -> tuple[Cell, ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _rotate_cw(cells: tuple[Cell, ...]) -> tuple[Cell, ...]:
    height = max(r for r, _ in cells) + 1
    return _normalize(tuple((c, height - 1 - r) for r, c in cells))


def _make_rotation(piece: str, index: int, cells: tuple[Cell, ...]) -> Rotation:
    width = max(c for _, c in cells) + 1
    height = max(r for r, _ in cells) + 1
    row_masks = tuple(
        sum(1 << c for r, c in cells if r == row) for row in range(height)
    )
    bottom = tuple(max(r for r, c in cells if c == col) for col in range(width))
    top = tuple(min(r for r, c in cells if c == col) for col in range(width))
    return Rotation(
        piece=piece,
        index=index,
        cells=cells,
        width=width,
        height=height,
        row_masks=row_masks,
        bottom=bottom,
        top=top,
    )


def _build() -> dict[str, tuple[Rotation, ...]]:
    table: dict[str, tuple[Rotation, ...]] = {}
    for piece, base in _BASE_CELLS.items():
        seen: list[tuple[Cell, ...]] = []
        cells = _normalize(base)
        for _ in range(4):
            if cells not in seen:
                seen.append(cells)
            cells = _rotate_cw(cells)
        table[piece] = tuple(
            _make_rotation(piece, i, c) for i, c in enumerate(seen)
        )
    return table


ROTATIONS: dict[str, tuple[Rotation, ...]] = _build()
PIECES: tuple[str, ...] = tuple(ROTATIONS)
