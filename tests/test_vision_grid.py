from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tetris_coach.app import CoachConfig
from tetris_coach.vision.grid import (
    cell_scores,
    classify_grid,
    otsu_threshold,
    otsu_threshold_hist,
)

from .synthetic import STYLES, render_board

FIXTURES = Path(__file__).parent / "fixtures"


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


def test_otsu_threshold_hist_matches_exact_within_bin_width() -> None:
    # Pixel-scale bimodal data: the histogram variant lands within one bin
    # width of the exact per-sample split.
    rng = np.random.default_rng(7)
    values = np.concatenate(
        [
            rng.normal(0.08, 0.02, 4000).clip(0.0, 1.0),
            rng.normal(0.85, 0.05, 1000).clip(0.0, 1.0),
        ]
    ).astype(np.float32)
    exact = otsu_threshold(values)
    hist = otsu_threshold_hist(values)
    bin_width = (float(values.max()) - float(values.min())) / 256
    assert abs(hist - exact) <= bin_width
    assert 0.2 < hist < 0.8


def test_otsu_threshold_hist_degenerate() -> None:
    assert otsu_threshold_hist(np.array([], dtype=np.float32)) == 0.5
    assert otsu_threshold_hist(np.array([0.3], dtype=np.float32)) == pytest.approx(0.3)
    # All-identical values must not crash.
    t = otsu_threshold_hist(np.full(500, 0.4, dtype=np.float32))
    assert t == pytest.approx(0.4)


def test_uniform_dark_region_is_empty_with_usable_confidence() -> None:
    # A uniform DARK region is exactly the empty board the darker-
    # background requirement guarantees: classified empty with enough
    # confidence to pass the engine's frame gate, so the tracker actually
    # observes a board wipe (game over, new game).
    dark = np.full((200, 100, 3), 20, dtype=np.uint8)
    occupancy, confidence = classify_grid(dark)
    assert not occupancy.any()
    assert confidence >= CoachConfig().min_confidence


def test_uniform_region_reads_empty_at_any_level() -> None:
    # A uniform region sits at its own background estimate whatever its
    # absolute level: a solid dark region is a dark theme's empty board
    # and a solid near-white region is a light theme's empty board. Both
    # classify empty with gate-passing confidence — no pure per-frame
    # classifier can tell a real white empty board from a pixel-identical
    # white flash; the flash safety this replaces (the old uniform-BRIGHT
    # all-filled rule) lives where the cross-frame information lives, the
    # tracker's reset debounce: see test_state.py's
    # test_three_unexplained_frames_then_recovery,
    # test_stable_empty_board_resets_on_fourth_frame, and
    # test_clear_lock_with_fade_frames_never_reset.
    for level in (20, 245):
        image = np.full((200, 100, 3), level, dtype=np.uint8)
        occupancy, confidence = classify_grid(image)
        assert not occupancy.any()
        assert confidence >= CoachConfig().min_confidence


def test_uniform_empty_board_render_passes_gate() -> None:
    # A rendered empty board is a uniform dark region: classified empty
    # with usable confidence, so a mid-game wipe reaches the tracker
    # instead of being dropped at the gate (which would leave the stale
    # stack and hint painted over an empty board indefinitely).
    image = render_board(np.zeros((20, 10), dtype=bool), STYLES[0], cell_size=20)
    occupancy, confidence = classify_grid(image)
    assert not occupancy.any()
    assert confidence >= CoachConfig().min_confidence


def test_start_screen_fixture_rejected() -> None:
    # The real light-board failure this redesign fixes: a white board
    # covered in start-screen text/graphics. The old absolute score read
    # it as 200/200 occupied at confidence 0.00 and moments of contrast
    # committed garbage grids. Now it must simply fail the engine's gate
    # (the exact occupancy is irrelevant for a non-gameplay frame).
    image = np.asarray(Image.open(FIXTURES / "roas_stacker" / "start_screen_board.png"))
    _, confidence = classify_grid(image)
    assert confidence < CoachConfig().min_confidence


def near_top_out_grid() -> np.ndarray:
    """A 160/200-filled board with buried holes; rows 0-2 empty."""
    grid = np.zeros((20, 10), dtype=bool)
    grid[3:, :] = True  # 170 cells
    for r in range(5, 15):  # 10 buried holes
        grid[r, (3 * r + 1) % 10] = False
    assert int(grid.sum()) == 160
    return grid


@pytest.mark.parametrize("style", STYLES, ids=lambda s: s.name)
def test_near_top_out_recovered(style) -> None:  # type: ignore[no-untyped-def]
    # Filled cells are the 80% MAJORITY here: any dominant-cluster
    # background estimate locks onto the pieces and inverts the reading
    # (at high confidence, on monochrome styles especially). The top-row
    # median estimator recovers the board exactly — a stack in row 0 is
    # game over, so the top row is background whatever the fill level.
    grid = near_top_out_grid()
    image = render_board(grid, style, cell_size=20)
    occupancy, confidence = classify_grid(image)
    np.testing.assert_array_equal(occupancy, grid)
    assert confidence > 0.2


@pytest.mark.parametrize("style", [STYLES[0], STYLES[4]], ids=lambda s: s.name)
def test_spawn_in_top_row_recovered(style) -> None:  # type: ignore[no-untyped-def]
    # An I-piece lying across row 0 contaminates 4 of the 10 top-row
    # cells the background estimator samples; the per-channel median
    # still lands on the 6 background cells.
    grid = np.zeros((20, 10), dtype=bool)
    grid[0, 3:7] = True  # I across the top row
    grid[19, :] = [True] * 6 + [False] * 4  # some stack far below
    image = render_board(grid, style, cell_size=18)
    occupancy, _ = classify_grid(image)
    np.testing.assert_array_equal(occupancy, grid)


def test_estimator_disagreement_reads_filled_at_zero_confidence() -> None:
    # The uniform-far branch: a heterogeneous top row (alternating black/
    # white cells) whose median matches nothing, over a solid field. Every
    # cell is far from the estimate yet mutually similar — the estimator
    # is inconsistent with the field. Must be the all-filled/0.0 sentinel
    # (rejected at the gate), never read as an empty board.
    image = np.zeros((400, 200, 3), dtype=np.uint8)
    for c in range(0, 10, 2):
        image[0:20, c * 20 : (c + 1) * 20] = 255
    occupancy, confidence = classify_grid(image)
    assert occupancy.all()
    assert confidence == 0.0


@pytest.mark.parametrize("style", [STYLES[0], STYLES[4]], ids=lambda s: s.name)
def test_gradient_match_or_reject(style) -> None:  # type: ignore[no-untyped-def]
    # Vertical lighting gradients: a mild ramp must classify exactly or
    # be rejected at the gate; a steep ramp swamps the class gap and must
    # be rejected. Either way the invariant is the same: never commit a
    # wrong reading (availability may degrade, correctness may not).
    grid = sample_grid()
    base = render_board(grid, style, cell_size=24).astype(np.int16)
    gate = CoachConfig().min_confidence
    for amp, must_reject in ((60, False), (140, True)):
        ramp = np.linspace(-amp, amp, base.shape[0]).astype(np.int16)
        image = (base + ramp[:, None, None]).clip(0, 255).astype(np.uint8)
        occupancy, confidence = classify_grid(image)
        if must_reject:
            assert confidence < gate
        else:
            assert bool(np.array_equal(occupancy, grid)) or confidence < gate


def test_random_boards_match_or_reject() -> None:
    # Broad-net sweep: ~50 seeded random legal boards (bottom-heavy
    # column stacks with buried holes) across all styles. Every frame
    # must either be read exactly or be rejected at the gate — a wrong
    # reading above the gate is a committable-garbage violation.
    rng = np.random.default_rng(20260913)
    gate = CoachConfig().min_confidence
    for i in range(50):
        style = STYLES[i % len(STYLES)]
        heights = rng.integers(0, 15, size=10)
        grid = np.zeros((20, 10), dtype=bool)
        for c in range(10):
            for r in range(20 - int(heights[c]), 20):
                grid[r, c] = rng.random() > 0.1  # ~10% buried holes
        image = render_board(grid, style, cell_size=16, seed=i)
        occupancy, confidence = classify_grid(image)
        assert bool(np.array_equal(occupancy, grid)) or confidence < gate, (
            f"garbage above the gate: style={style.name} seed={i} conf={confidence:.3f}"
        )


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
