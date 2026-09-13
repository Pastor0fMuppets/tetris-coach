import numpy as np
import pytest

from tetris_coach.vision.grid import cell_scores, classify_grid, otsu_threshold

from .synthetic import STYLES, render_board


def sample_grid() -> np.ndarray:
    """A mid-game board: uneven stack, holes, and a falling piece up top."""
    grid = np.zeros((20, 10), dtype=bool)
    # Falling S piece near the top.
    for r, c in ((2, 4), (2, 5), (3, 3), (3, 4)):
        grid[r, c] = True
    # Stack.
    rows = [
        "..........",
        "#.........",
        "##...#....",
        "###..##..#",
        "#######.##",
        "##.#######",
        "######.###",
    ]
    for i, line in enumerate(rows):
        r = 20 - len(rows) + i
        for c, ch in enumerate(line):
            if ch == "#":
                grid[r, c] = True
    return grid


@pytest.mark.parametrize("style", STYLES, ids=lambda s: s.name)
@pytest.mark.parametrize("cell_size", [12, 24, 37])
def test_classify_grid_recovers_occupancy(style, cell_size) -> None:  # type: ignore[no-untyped-def]
    grid = sample_grid()
    image = render_board(grid, style, cell_size=cell_size)
    occupancy, confidence = classify_grid(image)
    assert occupancy.shape == (20, 10)
    np.testing.assert_array_equal(occupancy, grid)
    assert confidence > 0.2


@pytest.mark.parametrize("style", STYLES, ids=lambda s: s.name)
def test_empty_board_classifies_empty(style) -> None:  # type: ignore[no-untyped-def]
    grid = np.zeros((20, 10), dtype=bool)
    image = render_board(grid, style, cell_size=20)
    occupancy, _ = classify_grid(image)
    assert not occupancy.any()


@pytest.mark.parametrize("style", STYLES, ids=lambda s: s.name)
def test_single_cell_detected(style) -> None:  # type: ignore[no-untyped-def]
    grid = np.zeros((20, 10), dtype=bool)
    grid[19, 0] = True
    image = render_board(grid, style, cell_size=18)
    occupancy, _ = classify_grid(image)
    np.testing.assert_array_equal(occupancy, grid)


def test_channel_order_is_irrelevant() -> None:
    grid = sample_grid()
    image = render_board(grid, STYLES[0], cell_size=20)
    occupancy_rgb, _ = classify_grid(image)
    occupancy_bgr, _ = classify_grid(image[:, :, ::-1])
    np.testing.assert_array_equal(occupancy_rgb, occupancy_bgr)
    np.testing.assert_array_equal(occupancy_rgb, grid)


def test_grayscale_image_supported() -> None:
    grid = sample_grid()
    image = render_board(grid, STYLES[1], cell_size=20)
    gray = image.mean(axis=2).astype(np.uint8)
    occupancy, _ = classify_grid(gray)
    np.testing.assert_array_equal(occupancy, grid)


def test_non_multiple_image_size() -> None:
    # Region selection is by hand: the image is rarely an exact multiple of
    # the cell grid. Render at 25px then crop a few pixels off two edges.
    grid = sample_grid()
    image = render_board(grid, STYLES[0], cell_size=25)
    cropped = image[3:-2, 2:-3]
    occupancy, _ = classify_grid(cropped)
    np.testing.assert_array_equal(occupancy, grid)


def test_cell_scores_shape_and_range() -> None:
    image = render_board(sample_grid(), STYLES[0], cell_size=16)
    scores = cell_scores(image)
    assert scores.shape == (20, 10)
    assert float(scores.min()) >= 0.0
    assert float(scores.max()) <= 1.0


def test_image_too_small_raises() -> None:
    with pytest.raises(ValueError):
        classify_grid(np.zeros((10, 5, 3), dtype=np.uint8))


def test_otsu_threshold_bimodal() -> None:
    values = np.array([0.1, 0.12, 0.11, 0.9, 0.88, 0.91], dtype=np.float32)
    t = otsu_threshold(values)
    assert 0.12 < t < 0.88


def test_otsu_threshold_degenerate() -> None:
    assert otsu_threshold(np.array([], dtype=np.float32)) == 0.5
    assert otsu_threshold(np.array([0.3], dtype=np.float32)) == pytest.approx(0.3)
    # All-identical values must not crash.
    t = otsu_threshold(np.full(10, 0.4, dtype=np.float32))
    assert t == pytest.approx(0.4)


def test_confidence_drops_with_ambiguity() -> None:
    grid = sample_grid()
    clean = render_board(grid, STYLES[2], cell_size=24)  # noise-free
    _, clean_conf = classify_grid(clean)
    # Replace one occupied cell with a mid-gray patch: neither bright nor
    # colored, so its score falls between the two classes.
    ambiguous = clean.copy()
    r, c = 19, 0
    assert grid[r, c]
    ambiguous[r * 24 : (r + 1) * 24, c * 24 : (c + 1) * 24] = 102
    _, ambiguous_conf = classify_grid(ambiguous)
    assert ambiguous_conf < clean_conf
