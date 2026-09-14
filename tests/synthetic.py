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


# WARNING: append new styles only — never insert or reorder. Tests index
# STYLES[0..2] by position (classic-dark, gray-flat, jstris-like).
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
    # Light themes: colored pieces on a white/near-white ground (the real
    # target game family behind tests/fixtures/roas_stacker). The lavender
    # + deep-blue pair is the multi-modality stressor that makes the sqrt
    # compression in the distance score load-bearing: lavender is the
    # near-background color, deep blue the far one.
    Style(
        name="paper-white",
        background=(248, 248, 246),
        cell_colors=(
            (37, 37, 229),  # deep blue (far from background)
            (210, 160, 235),  # lavender (near background)
            (227, 91, 2),
            (89, 177, 1),
            (215, 15, 55),
            (15, 155, 215),
            (227, 159, 2),
        ),
        gridline=(228, 228, 232),
        gridline_width=1,
        noise=4,
    ),
    Style(
        name="cream-mono",
        background=(245, 240, 228),
        cell_colors=((150, 140, 120),),  # monochrome pieces on light ground
        gridline=None,
        cell_inset=1,
        noise=3,
    ),
    Style(
        name="mid-gray-both",
        background=(128, 128, 128),
        # Pieces on BOTH sides of the background's brightness: any absolute
        # level rule fails here; distance-from-background does not care.
        cell_colors=((225, 225, 230), (38, 38, 44)),
        gridline=(118, 118, 118),
        gridline_width=1,
        noise=3,
    ),
)


def ghost_color(style: Style, score: float, color_index: int = 0) -> tuple[int, int, int]:
    """A style color faded toward the ground until it scores ``score``.

    A landing preview is the falling piece drawn translucent, so its cells
    sit on the segment between a piece color and the board's background.
    Which point of that segment is what the rule in
    :func:`~tetris_coach.vision.grid._ghost_layer` turns on, so the tests
    address it directly: ``score`` is the cell score the faded color is
    meant to produce (``grid._distance_scores``' sqrt-compressed normalized
    distance), not an alpha, so one number means the same thing on a dark
    theme and a light one.
    """
    background = np.asarray(style.background, dtype=np.float64)
    piece = np.asarray(style.cell_colors[color_index % len(style.cell_colors)], dtype=np.float64)
    # score = sqrt(distance / (255 * sqrt(C))) -> distance = score^2 * 255 * sqrt(C)
    target = (score**2) * 255.0 * np.sqrt(background.size)
    full = float(np.linalg.norm(piece - background))
    fraction = 0.0 if full == 0.0 else min(1.0, target / full)
    faded = background + (piece - background) * fraction
    return tuple(round(float(v)) for v in np.clip(faded, 0, 255))  # type: ignore[return-value]


def render_board(
    grid: np.ndarray,
    style: Style,
    cell_size: int = 24,
    seed: int = 0,
    ghost: np.ndarray | None = None,
    ghost_score: float = 0.30,
) -> np.ndarray:
    """Render an occupancy grid into an RGB uint8 image (H, W, 3).

    ``ghost`` is an optional second grid drawn the way almost every modern
    Tetris draws its landing preview: the same cell geometry as a piece,
    filled with a faded piece color (see :func:`ghost_color`) rather than
    a solid one. It is painted before the real cells, so a piece sitting
    on top of its own ghost hides it exactly as the game would.
    """
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

    def paint(mask: np.ndarray, fill: tuple[int, int, int] | None) -> None:
        for r in range(rows):
            for c in range(cols):
                if not mask[r, c]:
                    continue
                color = fill if fill is not None else colors[(r * cols + c) % len(colors)]
                inset = style.cell_inset
                x0 = c * cell_size + inset
                y0 = r * cell_size + inset
                x1 = (c + 1) * cell_size - 1 - inset
                y1 = (r + 1) * cell_size - 1 - inset
                draw.rectangle([x0, y0, x1, y1], fill=color)

    if ghost is not None:
        paint(ghost, ghost_color(style, ghost_score))
    paint(grid, None)

    out = np.asarray(img, dtype=np.int16)
    if style.noise:
        out = out + rng.integers(-style.noise, style.noise + 1, size=out.shape)
    return out.clip(0, 255).astype(np.uint8)


def label_color(style: Style) -> tuple[int, int, int]:
    """A faint caption color for ``style``: a third of the way off the ground.

    Matched to the real capture (tests/fixtures/live_session): a grey
    ~170 "NEXT" on a ~252 white ground, which scores 0.57 against that
    background — as far from it as the piece colors themselves (0.53-0.75),
    so no threshold can separate caption from piece.
    """
    return tuple(  # type: ignore[return-value]
        round(channel + 0.35 * ((0 if channel > 127 else 255) - channel))
        for channel in style.background
    )


def render_next_preview(
    cells: tuple[tuple[int, int], ...],
    style: Style,
    cell_size: int = 20,
    box_cells: tuple[int, int] = (4, 6),
    seed: int = 0,
    label: str | None = None,
    offset: tuple[int, int] | None = None,
) -> np.ndarray:
    """Render a next-piece preview box.

    The piece is roughly centered unless ``offset`` places its bounding
    box explicitly. ``label`` draws a faint caption in the top-left corner
    (games title their preview box "NEXT"), which is what a real preview
    crop holds besides the piece.
    """
    box_rows, box_cols = box_cells
    piece_h = max(r for r, _ in cells) + 1
    piece_w = max(c for _, c in cells) + 1
    off_r, off_c = (
        offset
        if offset is not None
        else (
            (box_rows - piece_h) // 2,
            (box_cols - piece_w) // 2,
        )
    )
    grid = np.zeros((box_rows, box_cols), dtype=bool)
    for r, c in cells:
        grid[r + off_r, c + off_c] = True
    image = render_board(grid, style, cell_size=cell_size, seed=seed)
    return image if label is None else with_label(image, style, label)


def with_label(image: np.ndarray, style: Style, text: str = "NEXT") -> np.ndarray:
    """Draw a faint caption into the top-left corner of a preview image."""
    pil = Image.fromarray(image)
    ImageDraw.Draw(pil).text((2, 1), text, fill=label_color(style))
    return np.asarray(pil, dtype=np.uint8)
