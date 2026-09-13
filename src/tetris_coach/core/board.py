"""Bitboard: 20 rows x 10 columns, one int per row.

Row 0 is the *top* of the board; bit ``c`` of a row represents column ``c``
(column 0 on the left). The board is immutable: :meth:`Board.drop` returns a
new board.

Feature helpers (holes, transitions, wells) follow Pierre Dellacherie's
definitions as popularised by Colin Fahey:

- Row transitions: filled/empty changes scanning each row left to right, with
  both side walls counted as filled (an empty row contributes 2).
- Column transitions: filled/empty changes scanning each column top to bottom,
  with the area above the board counted as empty and the floor as filled.
- Holes: empty cells with at least one filled cell anywhere above them in the
  same column.
- Cumulative wells: for every empty cell whose left and right neighbours are
  both filled (walls count as filled), 1 plus the number of consecutive empty
  cells directly below it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .pieces import Rotation

WIDTH = 10
HEIGHT = 20
FULL_ROW = (1 << WIDTH) - 1

_RIGHT_WALL = 1 << (WIDTH - 1)


def _build_row_transition_table() -> tuple[int, ...]:
    table = []
    for row in range(1 << WIDTH):
        prev = 1  # left wall is filled
        transitions = 0
        for c in range(WIDTH):
            cur = (row >> c) & 1
            if cur != prev:
                transitions += 1
            prev = cur
        if prev != 1:  # right wall is filled
            transitions += 1
        table.append(transitions)
    return tuple(table)


_ROW_TRANSITIONS: tuple[int, ...] = _build_row_transition_table()


def _compute_heights(rows: tuple[int, ...]) -> tuple[int, ...]:
    heights = [0] * WIDTH
    remaining = FULL_ROW
    for r, row in enumerate(rows):
        newly_topped = row & remaining
        while newly_topped:
            bit = newly_topped & -newly_topped
            newly_topped ^= bit
            heights[bit.bit_length() - 1] = HEIGHT - r
        remaining &= ~row
        if not remaining:
            break
    return tuple(heights)


@dataclass(frozen=True)
class DropResult:
    """Outcome of dropping a piece."""

    board: Board
    lines_cleared: int
    landing_row: int  # top row index of the piece's bounding box, pre-clear
    eroded_cells: int  # number of the piece's own cells removed by the clears


class Board:
    """Immutable 20x10 bitboard."""

    __slots__ = ("heights", "rows")

    rows: tuple[int, ...]
    heights: tuple[int, ...]

    def __init__(
        self,
        rows: Sequence[int] | None = None,
        *,
        _heights: tuple[int, ...] | None = None,
    ) -> None:
        """Build a board from row bitmasks; heights are always derived.

        ``_heights`` is private to :meth:`drop`'s no-clear fast path, which
        already knows the exact new heights; it must equal
        ``_compute_heights(rows)``. External callers must not pass it — a
        board whose heights disagree with its rows breaks every drop.
        """
        row_tuple = (0,) * HEIGHT if rows is None else tuple(rows)
        if len(row_tuple) != HEIGHT:
            raise ValueError(f"expected {HEIGHT} rows, got {len(row_tuple)}")
        object.__setattr__(self, "rows", row_tuple)
        object.__setattr__(
            self, "heights", _compute_heights(row_tuple) if _heights is None else _heights
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Board is immutable")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Board) and self.rows == other.rows

    def __hash__(self) -> int:
        return hash(self.rows)

    def __str__(self) -> str:
        lines = []
        for row in self.rows:
            lines.append("".join("#" if row >> c & 1 else "." for c in range(WIDTH)))
        return "\n".join(lines)

    @classmethod
    def from_grid(cls, grid: Sequence[Sequence[object]]) -> Board:
        """Build from a 20x10 array of truthy values (row 0 = top)."""
        if len(grid) != HEIGHT:
            raise ValueError(f"expected {HEIGHT} rows, got {len(grid)}")
        rows = []
        for grid_row in grid:
            if len(grid_row) != WIDTH:
                raise ValueError(f"expected {WIDTH} columns, got {len(grid_row)}")
            rows.append(sum(1 << c for c, filled in enumerate(grid_row) if filled))
        return cls(rows)

    def to_grid(self) -> list[list[bool]]:
        return [[bool(row >> c & 1) for c in range(WIDTH)] for row in self.rows]

    def cell_count(self) -> int:
        return sum(row.bit_count() for row in self.rows)

    # ------------------------------------------------------------------
    # Piece placement
    # ------------------------------------------------------------------

    def drop(self, rotation: Rotation, col: int) -> DropResult | None:
        """Drop ``rotation`` with its left edge at ``col``.

        Returns ``None`` if the column is out of range or the piece cannot
        fit on the board (top-out).
        """
        if col < 0 or col + rotation.width > WIDTH:
            return None
        heights = self.heights
        bottom = rotation.bottom
        # Row index of the highest filled cell of column c is HEIGHT - heights[c];
        # the piece's lowest cell in each column rests directly above it.
        landing = HEIGHT  # will be reduced by the binding column
        for j in range(rotation.width):
            candidate = HEIGHT - heights[col + j] - bottom[j] - 1
            landing = min(landing, candidate)
        if landing < 0:
            return None  # piece would stick out the top

        rows = list(self.rows)
        cleared_rows: list[int] = []
        eroded = 0
        for i, mask in enumerate(rotation.row_masks):
            r = landing + i
            merged = rows[r] | (mask << col)
            rows[r] = merged
            if merged == FULL_ROW:
                cleared_rows.append(r)
                eroded += mask.bit_count()

        if cleared_rows:
            cleared_set = set(cleared_rows)
            kept = [row for r, row in enumerate(rows) if r not in cleared_set]
            new_rows = tuple([0] * len(cleared_rows) + kept)
            board = Board(new_rows)
        else:
            new_heights = list(heights)
            for j in range(rotation.width):
                # The piece's top cell in every occupied column ends up above
                # the previous stack top, so the new height is exact.
                new_heights[col + j] = HEIGHT - (landing + rotation.top[j])
            board = Board(tuple(rows), _heights=tuple(new_heights))

        return DropResult(
            board=board,
            lines_cleared=len(cleared_rows),
            landing_row=landing,
            eroded_cells=eroded,
        )

    # ------------------------------------------------------------------
    # Dellacherie features
    # ------------------------------------------------------------------

    def hole_count(self) -> int:
        covered = 0
        holes = 0
        for row in self.rows:
            holes += (covered & ~row & FULL_ROW).bit_count()
            covered |= row
        return holes

    def row_transitions(self) -> int:
        table = _ROW_TRANSITIONS
        return sum(table[row] for row in self.rows)

    def column_transitions(self) -> int:
        rows = self.rows
        transitions = rows[0].bit_count()  # area above the board is empty
        prev = rows[0]
        for i in range(1, HEIGHT):
            row = rows[i]
            transitions += (prev ^ row).bit_count()
            prev = row
        transitions += (prev ^ FULL_ROW).bit_count()  # floor is filled
        return transitions

    def cumulative_wells(self) -> int:
        rows = self.rows
        wells = 0
        for r in range(HEIGHT):
            row = rows[r]
            well_mask = ~row & ((row << 1) | 1) & ((row >> 1) | _RIGHT_WALL) & FULL_ROW
            while well_mask:
                bit = well_mask & -well_mask
                well_mask ^= bit
                depth = 1
                rr = r + 1
                while rr < HEIGHT and not rows[rr] & bit:
                    depth += 1
                    rr += 1
                wells += depth
        return wells
