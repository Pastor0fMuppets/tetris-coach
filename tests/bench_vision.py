"""Vision-stage wall-clock timings for BENCH.md.

Not part of the test suite (no ``test_`` prefix, so pytest never collects
it): wall-clock numbers are flaky under load, so they are reported in
BENCH.md rather than asserted. Run with:

    python -m tests.bench_vision
"""

from __future__ import annotations

import time

import numpy as np

from tetris_coach.core.pieces import ROTATIONS
from tetris_coach.vision.grid import classify_grid
from tetris_coach.vision.pieces_vision import identify_next

from .synthetic import STYLES, render_board
from .test_vision_grid import sample_grid


def render_preview_200(piece: str, style, seed: int = 0) -> np.ndarray:  # type: ignore[no-untyped-def]
    """A 200x200 preview box (5x5 cells of 40 px) with the piece centered."""
    cells = ROTATIONS[piece][0].cells
    piece_h = max(r for r, _ in cells) + 1
    piece_w = max(c for _, c in cells) + 1
    grid = np.zeros((5, 5), dtype=bool)
    off_r = (5 - piece_h) // 2
    off_c = (5 - piece_w) // 2
    for r, c in cells:
        grid[r + off_r, c + off_c] = True
    return render_board(grid, style, cell_size=40, seed=seed)


def timeit(fn, n: int = 100) -> tuple[float, float, float]:  # type: ignore[no-untyped-def]
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    return float(np.percentile(arr, 50)), float(np.percentile(arr, 95)), float(arr.max())


def main() -> None:
    style = STYLES[0]  # classic-dark: gridlines + per-pixel noise
    board = render_board(sample_grid(), style, cell_size=60)  # 600x1200, Retina-scale
    assert board.shape == (1200, 600, 3), board.shape
    preview = render_preview_200("T", style)
    assert preview.shape == (200, 200, 3), preview.shape
    preview_same = preview.copy()

    classify_grid(board)  # warm-up
    identify_next(preview)

    for label, fn, n in (
        ("grid stage: classify_grid 600x1200", lambda: classify_grid(board), 100),
        ("preview stage: identify_next 200x200", lambda: identify_next(preview), 100),
        (
            "preview cache gate: np.array_equal 200x200",
            lambda: np.array_equal(preview, preview_same),
            200,
        ),
    ):
        p50, p95, mx = timeit(fn, n)
        print(f"{label:45s} p50 {p50:7.3f} ms  p95 {p95:7.3f} ms  max {mx:7.3f} ms")


if __name__ == "__main__":
    main()
