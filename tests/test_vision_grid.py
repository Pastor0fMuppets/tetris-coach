from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tetris_coach.app import CoachConfig
from tetris_coach.vision.grid import (
    GridClassifier,
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
    # median estimator recovers the board exactly, because rows 0-2 are
    # empty: the median is taken over 10 true background cells, whatever
    # the fill level below. (A stack CAN legally reach row 0 — that case
    # is capped, see test_top_row_stack_never_wrong_above_gate.)
    grid = near_top_out_grid()
    image = render_board(grid, style, cell_size=20)
    occupancy, confidence = classify_grid(image)
    np.testing.assert_array_equal(occupancy, grid)
    assert confidence > 0.2


@pytest.mark.parametrize("style", STYLES, ids=lambda s: s.name)
def test_spawn_in_top_row_recovered(style) -> None:  # type: ignore[no-untyped-def]
    # An I-piece lying across row 0 contaminates 4 of the 10 top-row
    # cells the background estimator samples; the per-channel median
    # still lands on the 6 background cells and the occupancy is exact.
    # It is also VOUCHED FOR: the piece is airborne (empty board under
    # every one of its cells), which is what separates it from the stack
    # grounded at row 0 that inverts the estimate. Capping this case —
    # and pieces are visible in row 0 in most games — is what left a
    # whole real game unreadable. With a background hint, same reading.
    grid = np.zeros((20, 10), dtype=bool)
    grid[0, 3:7] = True  # I across the top row
    grid[19, :] = [True] * 6 + [False] * 4  # some stack far below
    image = render_board(grid, style, cell_size=18)
    occupancy, confidence = classify_grid(image)
    np.testing.assert_array_equal(occupancy, grid)
    assert confidence >= CoachConfig().min_confidence
    occupancy, confidence = classify_grid(image, background=style.background)
    np.testing.assert_array_equal(occupancy, grid)
    assert confidence >= CoachConfig().min_confidence


@pytest.mark.parametrize("style", STYLES, ids=lambda s: s.name)
def test_falling_piece_through_the_top_rows_stays_vouched(style) -> None:  # type: ignore[no-untyped-def]
    # The same piece on its way down, one row at a time, over a stack:
    # every frame in which it still touches row 0 must be exact AND
    # above the gate. This is the frame-to-frame availability the strict
    # cap destroyed — a T spawning at row 0 blanked the coach until it
    # had fallen clear of the top row.
    gate = CoachConfig().min_confidence
    for row in range(3):
        grid = np.zeros((20, 10), dtype=bool)
        grid[row, 3:6] = True  # T's bar
        grid[row + 1, 4] = True  # T's nub
        grid[18:, 0:6] = True  # a stack, well below
        image = render_board(grid, style, cell_size=18)
        occupancy, confidence = classify_grid(image)
        np.testing.assert_array_equal(occupancy, grid)
        assert confidence >= gate, f"{style.name} row {row}: conf={confidence:.3f}"


@pytest.mark.parametrize("style", STYLES, ids=lambda s: s.name)
def test_majority_occupied_top_row_still_caps(style) -> None:  # type: ignore[no-untyped-def]
    # The other half of the relaxation: six of the ten top-row cells
    # reading occupied contradicts the estimator's own majority-
    # background premise (and no single tetromino can put six cells in
    # one row), so the reading is not vouched for whatever it says.
    grid = np.zeros((20, 10), dtype=bool)
    grid[0:2, 0:6] = True
    image = render_board(grid, style, cell_size=18)
    _occupancy, confidence = classify_grid(image)
    assert confidence == 0.0


@pytest.mark.parametrize("style", STYLES, ids=lambda s: s.name)
def test_grounded_top_row_cell_still_caps(style) -> None:  # type: ignore[no-untyped-def]
    # A single column stacked from row 0 to the floor puts ONE cell in
    # the top row — a minority, and indistinguishable by count from a
    # spawned piece. It is grounded, not airborne, so it is capped: this
    # is the configuration whose median inverts once enough columns join
    # it, and the count alone can never tell the two apart.
    grid = np.zeros((20, 10), dtype=bool)
    grid[:, 2] = True
    image = render_board(grid, style, cell_size=18)
    _occupancy, confidence = classify_grid(image)
    assert confidence == 0.0


def side_stack_grid() -> np.ndarray:
    """Side columns stacked to visible row 0; spawn columns clear: legal."""
    grid = np.zeros((20, 10), dtype=bool)
    grid[:, 0:3] = True
    grid[:, 7:10] = True
    return grid


def garbage_grid(well: int = 6) -> np.ndarray:
    """Versus-mode garbage (9/10 filled) pushed up to the top row: legal."""
    grid = np.ones((20, 10), dtype=bool)
    grid[:, well] = False
    return grid


@pytest.mark.parametrize("style", STYLES, ids=lambda s: s.name)
def test_top_row_stack_never_wrong_above_gate(style) -> None:  # type: ignore[no-untyped-def]
    # A stack legally reaching visible row 0 leaves the top row majority
    # non-background, so the top-row median locks onto the piece colors
    # and the WHOLE board inverts — before the strict cap, at confidence
    # 0.95 on monochrome styles: committable garbage. The invariant:
    # never a wrong reading above the gate (availability may degrade,
    # correctness may not).
    gate = CoachConfig().min_confidence
    for grid in (side_stack_grid(), garbage_grid()):
        image = render_board(grid, style, cell_size=20)
        occupancy, confidence = classify_grid(image)
        assert bool(np.array_equal(occupancy, grid)) or confidence < gate, (
            f"wrong reading above the gate: {style.name} conf={confidence:.3f}"
        )


@pytest.mark.parametrize("style", STYLES, ids=lambda s: s.name)
def test_growth_into_top_row_never_flips_polarity(style) -> None:  # type: ignore[no-untyped-def]
    # The adjacent-frame failure the cap exists for (gray-flat, worst
    # case: monochrome pieces): as side columns grow into row 0, the
    # pre-cap pipeline read 4 filled top-row cells exactly @0.95, then 5
    # rejected @0.0, then 6 INVERTED @0.95 — a polarity flip between
    # adjacent frames that sails through any confidence gate. Every
    # contaminated frame must now be exact or below the gate.
    #
    # Swept over every style because a COUNT-based cap is not enough here:
    # once the median inverts, the reading shows 10-k occupied top-row
    # cells, a MINORITY, so a majority rule waves the inverted board
    # through at 0.95 on both monochrome styles (measured at k=6..9 on
    # gray-flat and cream-mono). What refuses it is that the cells it
    # calls occupied run all the way down — grounded, not airborne.
    gate = CoachConfig().min_confidence
    for k in range(10):
        grid = np.zeros((20, 10), dtype=bool)
        grid[:, 0:k] = True
        grid[19, :] = True
        image = render_board(grid, style, cell_size=20)
        occupancy, confidence = classify_grid(image)
        assert bool(np.array_equal(occupancy, grid)) or confidence < gate, (
            f"wrong reading above the gate at k={k}: conf={confidence:.3f}"
        )


class TestUnobservableCells:
    """Cells a game UI panel covers: out of the background sample, out of
    the top-row cap.

    A floating NEXT preview parks opaque pixels on fixed board cells for a
    whole session. Fed to the top-row median they bias the board's
    background estimate on every frame, and fed to the top-row cap they
    make the top row permanently "occupied" — which, before this, rejected
    every frame and so kept :class:`GridClassifier`'s memory (which anchors
    only from accepted frames) from ever forming.
    """

    ROWS = 12
    STYLE = STYLES[2]  # jstris-like: solid colors, no noise
    CELL = 16
    PANEL = (9, 200, 40)  # a color on no style's palette

    def _board(self, grid: np.ndarray) -> np.ndarray:
        return render_board(grid, self.STYLE, cell_size=self.CELL)

    def _paint(self, image: np.ndarray, cells) -> np.ndarray:  # type: ignore[no-untyped-def]
        """Stamp the panel color over whole cells, as an opaque box does."""
        out = image.copy()
        for r, c in cells:
            out[
                r * self.CELL : (r + 1) * self.CELL,
                c * self.CELL : (c + 1) * self.CELL,
            ] = self.PANEL
        return out

    @staticmethod
    def _mask(rows: range, cols: range) -> frozenset[tuple[int, int]]:
        return frozenset((r, c) for r in rows for c in cols)

    def test_panel_cells_are_out_of_the_background_median(self) -> None:
        # A five-cell-wide panel over the top row is HALF the estimator's
        # sample: the median lands between the panel color and the board,
        # and every score in the frame is measured from a color that is on
        # neither. Declared unobservable, the estimate is bit-identical to
        # the same board with no panel at all.
        grid = twelve_row_grid()
        clean = self._board(grid)
        mask = self._mask(range(2), range(5, 10))
        painted = self._paint(clean, mask)
        observable = np.ones((self.ROWS, 10), dtype=bool)
        for r, c in mask:
            observable[r, c] = False

        clean_scores = cell_scores(clean, rows=self.ROWS)
        masked_scores = cell_scores(painted, rows=self.ROWS, unobservable_cells=mask)
        np.testing.assert_allclose(masked_scores[observable], clean_scores[observable], atol=1e-6)
        # Undeclared, the same frame is measured from a color the board
        # never shows: the estimate moves off the background.
        naive_scores = cell_scores(painted, rows=self.ROWS)
        assert float(np.abs(naive_scores[observable] - clean_scores[observable]).max()) > 0.1

        # ... and the occupancy that follows from it is the true board.
        occupancy, confidence = classify_grid(painted, rows=self.ROWS, unobservable_cells=mask)
        np.testing.assert_array_equal(occupancy[observable], grid[observable])
        assert confidence >= CoachConfig().min_confidence

    def test_panel_over_the_top_row_no_longer_caps_every_frame(self) -> None:
        # Six of the ten top-row cells covered: undeclared, a majority of
        # the top row reads occupied on EVERY frame and the cap fires on
        # every frame — the permanent rejection that deadlocked the live
        # session. Declared, the remaining four cells are the sample and
        # the frame is read and vouched for.
        grid = twelve_row_grid()
        mask = self._mask(range(2), range(4, 10))
        painted = self._paint(self._board(grid), mask)
        _naive_occupancy, naive_confidence = classify_grid(painted, rows=self.ROWS)
        assert naive_confidence == 0.0
        occupancy, confidence = classify_grid(painted, rows=self.ROWS, unobservable_cells=mask)
        assert confidence >= CoachConfig().min_confidence
        observable = np.ones((self.ROWS, 10), dtype=bool)
        for r, c in mask:
            observable[r, c] = False
        np.testing.assert_array_equal(occupancy[observable], grid[observable])

    def test_memory_anchors_and_skips_the_panel(self) -> None:
        # The deadlock in one stream: a classifier told about the panel
        # accepts frames, anchors its background memory, and confirms it —
        # on the exact background of the board, not a blend with the panel
        # color. Undeclared, nothing is ever accepted and nothing anchors.
        mask = self._mask(range(2), range(4, 10))
        frames = [self._paint(self._board(twelve_row_grid()), mask) for _ in range(3)]

        deaf = GridClassifier(rows=self.ROWS)
        for frame in frames:
            assert deaf.classify(frame)[1] < CoachConfig().min_confidence
        assert deaf.background is None and not deaf.confirmed

        told = GridClassifier(rows=self.ROWS, unobservable_cells=mask)
        for frame in frames:
            assert told.classify(frame)[1] >= CoachConfig().min_confidence
        assert told.confirmed
        np.testing.assert_allclose(told.background, self.STYLE.background, atol=2.0)


@pytest.mark.parametrize("style", STYLES, ids=lambda s: s.name)
def test_background_hint_reads_top_row_stacks_exactly(style) -> None:  # type: ignore[no-untyped-def]
    # With the true background supplied (a cross-frame memory), polarity
    # is anchored by the hint instead of the contaminated top row: the
    # same row-0-stacked boards read EXACTLY at solid confidence, on
    # every style. The paper-white garbage case additionally pins the
    # MIN_SPREAD re-split: with only 20 background cells, plain Otsu
    # splits lavender (the near-background piece color) from the far
    # piece colors and misreads 26 cells at gate-passing confidence.
    gate = CoachConfig().min_confidence
    for grid in (side_stack_grid(), garbage_grid(), near_top_out_grid()):
        image = render_board(grid, style, cell_size=20)
        occupancy, confidence = classify_grid(image, background=style.background)
        np.testing.assert_array_equal(occupancy, grid)
        assert confidence >= gate


def test_uniform_frame_against_remembered_background() -> None:
    # Grid-level half of the dark-theme overlay fix: with a remembered
    # DARK background, a solid BRIGHT frame cannot be the empty board —
    # it is uniform-far (all-occupied @0.0, gated), not empty @0.5. A
    # solid frame AT the remembered background is a true wipe and stays
    # readable as empty.
    dark = (10, 10, 14)
    flash = np.full((200, 100, 3), 245, dtype=np.uint8)
    occupancy, confidence = classify_grid(flash, background=dark)
    assert occupancy.all()
    assert confidence == 0.0
    wipe = np.full((200, 100, 3), 12, dtype=np.uint8)
    occupancy, confidence = classify_grid(wipe, background=dark)
    assert not occupancy.any()
    assert confidence >= CoachConfig().min_confidence


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


class TestGridClassifier:
    """The cross-frame background memory and its anchoring protocol."""

    GATE = CoachConfig().min_confidence

    @pytest.mark.parametrize("style", STYLES, ids=lambda s: s.name)
    def test_memory_reads_top_row_stacks_exactly(self, style) -> None:  # type: ignore[no-untyped-def]
        # The stream every real session produces: the empty board, a few
        # normal frames, then the stack legally reaching row 0. Where the
        # memoryless path must refuse (or, pre-cap, INVERTED the board at
        # confidence 0.95), the anchored classifier reads every frame
        # exactly at gate-passing confidence.
        classifier = GridClassifier()
        empty = np.zeros((20, 10), dtype=bool)
        occupancy, confidence = classifier.classify(render_board(empty, style, cell_size=20))
        assert not occupancy.any() and confidence >= self.GATE
        low = np.zeros((20, 10), dtype=bool)
        low[17:, 0:5] = True
        occupancy, confidence = classifier.classify(render_board(low, style, cell_size=20))
        np.testing.assert_array_equal(occupancy, low)
        assert confidence >= self.GATE
        assert classifier.confirmed
        for grid in (side_stack_grid(), garbage_grid(), near_top_out_grid()):
            occupancy, confidence = classifier.classify(render_board(grid, style, cell_size=20))
            np.testing.assert_array_equal(occupancy, grid)
            assert confidence >= self.GATE

    def test_bootstrap_keeps_the_strict_cap(self) -> None:
        # With no memory the classifier is exactly classify_grid: a
        # row-0-contaminated first frame is never vouched for, and a
        # rejected frame must not anchor anything.
        classifier = GridClassifier()
        image = render_board(side_stack_grid(), STYLES[1], cell_size=20)
        _, confidence = classifier.classify(image)
        assert confidence < self.GATE
        assert classifier.background is None

    def test_confirmed_memory_gates_bright_overlay_forever(self) -> None:
        # Issue-2 core: after real dark-theme frames confirmed the
        # memory, a solid bright frame is uniform-far (all-occupied
        # @0.0) on every one of any number of frames — never the
        # empty-board @0.5 that a memoryless read would produce — and
        # the memory itself is untouched.
        classifier = GridClassifier()
        grid = np.zeros((20, 10), dtype=bool)
        grid[18:, 0:4] = True
        image = render_board(grid, STYLES[0], cell_size=20)
        classifier.classify(image)
        assert classifier.confirmed
        anchored = classifier.background
        flash = np.full((400, 200, 3), 245, dtype=np.uint8)
        for _ in range(10):
            occupancy, confidence = classifier.classify(flash)
            assert occupancy.all()
            assert confidence == 0.0
        np.testing.assert_array_equal(classifier.background, anchored)
        # The board frame after the overlay reads as before: no re-sync.
        occupancy, confidence = classifier.classify(image)
        np.testing.assert_array_equal(occupancy, grid)
        assert confidence >= self.GATE

    def test_true_wipe_still_reads_empty_under_memory(self) -> None:
        # The reason uniform-near survives: a wipe to the board's own
        # background color must keep reaching the tracker.
        classifier = GridClassifier()
        grid = np.zeros((20, 10), dtype=bool)
        grid[18:, 0:4] = True
        classifier.classify(render_board(grid, STYLES[0], cell_size=20))
        empty = render_board(np.zeros((20, 10), dtype=bool), STYLES[0], cell_size=20)
        occupancy, confidence = classifier.classify(empty)
        assert not occupancy.any()
        assert confidence >= self.GATE

    def test_provisional_anchor_recovers_from_pause_panel(self) -> None:
        # Session started while a solid bright panel covers a dark-theme
        # board: the panel reads as a (light-theme) empty board and
        # anchors only PROVISIONALLY. The first real frame rejects under
        # that anchor, re-bootstraps, and confirms on the true
        # background — the wrong provisional anchor cannot wedge the
        # session.
        classifier = GridClassifier()
        panel = np.full((400, 200, 3), 235, dtype=np.uint8)
        occupancy, confidence = classifier.classify(panel)
        assert not occupancy.any() and confidence >= self.GATE
        assert classifier.background is not None and not classifier.confirmed
        grid = np.zeros((20, 10), dtype=bool)
        grid[18:, 0:4] = True
        occupancy, confidence = classifier.classify(render_board(grid, STYLES[0], cell_size=20))
        np.testing.assert_array_equal(occupancy, grid)
        assert confidence >= self.GATE
        assert classifier.confirmed


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


def twelve_row_grid() -> np.ndarray:
    """A mid-game 12-row board: stack at the bottom, piece near the top."""
    grid = np.zeros((12, 10), dtype=bool)
    for r, c in ((1, 4), (1, 5), (2, 3), (2, 4)):  # falling S
        grid[r, c] = True
    rows = [
        "#.........",
        "###..##..#",
        "##.#######",
    ]
    for i, line in enumerate(rows):
        r = 12 - len(rows) + i
        for c, ch in enumerate(line):
            if ch == "#":
                grid[r, c] = True
    return grid


@pytest.mark.parametrize("style", STYLES, ids=lambda s: s.name)
def test_classify_grid_rows_12(style) -> None:  # type: ignore[no-untyped-def]
    grid = twelve_row_grid()
    image = render_board(grid, style, cell_size=24)
    occupancy, confidence = classify_grid(image, rows=12)
    assert occupancy.shape == (12, 10)
    np.testing.assert_array_equal(occupancy, grid)
    assert confidence > 0.2


@pytest.mark.parametrize("style", STYLES, ids=lambda s: s.name)
def test_grid_classifier_rows_12(style) -> None:  # type: ignore[no-untyped-def]
    # The classifier slices the image by its constructed row count; its
    # background memory is a color vector, shape-free across heights.
    classifier = GridClassifier(rows=12)
    gate = CoachConfig().min_confidence
    empty = np.zeros((12, 10), dtype=bool)
    occupancy, confidence = classifier.classify(render_board(empty, style, cell_size=24))
    assert occupancy.shape == (12, 10)
    assert not occupancy.any() and confidence >= gate
    grid = twelve_row_grid()
    occupancy, confidence = classifier.classify(render_board(grid, style, cell_size=24))
    np.testing.assert_array_equal(occupancy, grid)
    assert confidence >= gate
    assert classifier.confirmed
