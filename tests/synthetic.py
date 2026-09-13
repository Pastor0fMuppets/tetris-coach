"""Synthetic Tetris screenshots rendered with Pillow, for vision tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class Style:
    """Rendering style approximating a family of Tetris skins."""

    name: str
    background: tuple[int, int, int]
    cell_colors: tuple[tuple[int, int, int], ...]
    gridline: tuple[int, int, int] | None = None  # None = no gridlines
    gridline_width: int = 1
    cell_inset: int = 0  # gap between cell fill and cell border, in pixels
    noise: int = 0  # +/- amplitude of uniform per-pixel noise


STYLES = (
    Style(
        name="classic-dark",
        background=(8, 8, 10),
        cell_colors=(
            (0, 240, 240),  # cyan
            (240, 240, 0),  # yellow
            (160, 0, 240),  # purple
            (0, 240, 0),  # green
            (240, 0, 0),  # red
            (0, 0, 240),  # blue
            (240, 160, 0),  # orange
        ),
        gridline=(40, 40, 46),
        gridline_width=1,
        noise=6,
    ),
    Style(
        name="gray-flat",
        background=(52, 54, 60),
        cell_colors=((200, 205, 215),),  # monochrome pieces
        gridline=None,
        cell_inset=1,
        noise=3,
    ),
    Style(
        name="jstris-like",
        background=(0, 0, 0),
        cell_colors=(
            (15, 155, 215),
            (227, 91, 2),
            (33, 65, 198),
            (89, 177, 1),
            (215, 15, 55),
            (175, 41, 138),
            (227, 159, 2),
        ),
        gridline=(25, 25, 25),
        gridline_width=2,
        noise=0,
    ),
    Style(
        name="pastel-navy",
        background=(28, 32, 58),
        cell_colors=(
            (150, 220, 220),
            (235, 230, 160),
            (200, 160, 230),
            (170, 225, 170),
        ),
        gridline=(45, 50, 80),
        gridline_width=1,
        cell_inset=2,
        noise=4,
    ),
)


def render_board(
    grid: np.ndarray,
    style: Style,
    cell_size: int = 24,
    seed: int = 0,
) -> np.ndarray:
    """Render an occupancy grid into an RGB uint8 image (H, W, 3)."""
    rows, cols = grid.shape
    width, height = cols * cell_size, rows * cell_size
    img = Image.new("RGB", (width, height), style.background)
    draw = ImageDraw.Draw(img)
    rng = np.random.default_rng(seed)

    if style.gridline is not None:
        for r in range(rows + 1):
            y = min(r * cell_size, height - 1)
            draw.rectangle(
                [0, y, width - 1, min(y + style.gridline_width - 1, height - 1)],
                fill=style.gridline,
            )
        for c in range(cols + 1):
            x = min(c * cell_size, width - 1)
            draw.rectangle(
                [x, 0, min(x + style.gridline_width - 1, width - 1), height - 1],
                fill=style.gridline,
            )

    colors = style.cell_colors
    for r in range(rows):
        for c in range(cols):
            if not grid[r, c]:
                continue
            color = colors[(r * cols + c) % len(colors)]
            inset = style.cell_inset
            x0 = c * cell_size + inset
            y0 = r * cell_size + inset
            x1 = (c + 1) * cell_size - 1 - inset
            y1 = (r + 1) * cell_size - 1 - inset
            draw.rectangle([x0, y0, x1, y1], fill=color)

    out = np.asarray(img, dtype=np.int16)
    if style.noise:
        out = out + rng.integers(-style.noise, style.noise + 1, size=out.shape)
    return out.clip(0, 255).astype(np.uint8)


def render_next_preview(
    cells: tuple[tuple[int, int], ...],
    style: Style,
    cell_size: int = 20,
    box_cells: tuple[int, int] = (4, 6),
    seed: int = 0,
) -> np.ndarray:
    """Render a next-piece preview box with the piece roughly centered."""
    box_rows, box_cols = box_cells
    piece_h = max(r for r, _ in cells) + 1
    piece_w = max(c for _, c in cells) + 1
    off_r = (box_rows - piece_h) // 2
    off_c = (box_cols - piece_w) // 2
    grid = np.zeros((box_rows, box_cols), dtype=bool)
    for r, c in cells:
        grid[r + off_r, c + off_c] = True
    return render_board(grid, style, cell_size=cell_size, seed=seed)
