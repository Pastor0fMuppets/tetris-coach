"""The landing-preview layer over the whole synthetic style matrix.

``tests/fixtures/ghost_session`` pins the rule on the real pixels it was
diagnosed from. This pins the shape of the rule everywhere else: seven
themes (dark, light, monochrome, pastel, pieces on both sides of the
background) x a preview drawn at a range of opacities x where the preview
stands on the board.

The preview is rendered the way a game draws one — the same cell geometry
as a piece, filled with a piece color faded toward the ground — and
addressed by the cell SCORE it produces rather than by an alpha, so one
number means the same thing on a near-white theme and a near-black one
(see :func:`tests.synthetic.ghost_color`).

Three regimes, and the rule is supposed to behave differently in each:

- Faint (0.10): below the band. The ordinary split already calls these
  cells empty; the layer rule is not needed and does not fire.
- In the band (0.28, 0.34): named. The cells read EMPTY and — the point
  of naming rather than thresholding — the frame's confidence comes back
  to what the same board reads WITHOUT a preview on it.
- Close to a real piece (0.45, 0.60): the rule HOLDS. Nothing is deleted;
  the cells stay content and the frame reads as it would have anyway.
  This is the deliberate direction of the error, because a preview left
  in costs held frames while real content taken out deletes stack.
"""

from __future__ import annotations

import numpy as np
import pytest

from tetris_coach.vision.grid import (
    _GHOST_SEPARATION,
    MIN_SPREAD,
    _cell_colors,
    _distance_scores,
    _ghost_layer,
    classify_grid,
)

from .synthetic import STYLES, Style, render_board, with_label

ROWS, COLS = 12, 10

# Opacities, as the cell score each produces against the theme's ground.
FAINT = 0.10  # under the band: indistinguishable from the background
NAMED = (0.28, 0.34)  # inside the band
SOLID_LOOKING = (0.45, 0.60)  # at or above a real piece color

# A piece hanging at the top of the board on every scene, so each frame has
# a falling piece as well as a stack — and so the solid class is never just
# the stack.
FALLING = [(2, 7), (2, 8), (3, 7), (3, 8)]

# (stack cells, preview cells). The preview rests on something in all four,
# which is what a landing preview does; what differs is what is BESIDE and
# ABOVE it, which is what the rule turns on.
SCENES: dict[str, tuple[list[tuple[int, int]], list[tuple[int, int]]]] = {
    # On the bare floor, clear of the stack.
    "floor": (
        [(11, c) for c in range(4)] + [(10, c) for c in range(3)] + FALLING,
        [(10, 6), (10, 7), (11, 6), (11, 7)],
    ),
    # On the flat top of the stack — the ordinary case in a real game.
    "surface": (
        [(11, c) for c in range(9)] + FALLING,
        [(9, 3), (9, 4), (10, 3), (10, 4)],
    ),
}
# Walled in on both sides by the stack. Here a preview and a piece that was
# played into the notch look exactly alike, so the rule must decline.
WELL = (
    [(11, c) for c in (0, 1, 2, 5, 6, 7, 8)] + [(10, c) for c in (0, 1, 2, 5, 6)] + FALLING,
    [(10, 3), (10, 4), (11, 3), (11, 4)],
)
# The stack runs straight into its left edge along the floor — the shape of
# the real pale-periwinkle piece in tests/fixtures/roas_stacker that must
# never be mistaken for a preview.
ABUTTING = (
    [(11, c) for c in range(4)] + [(10, c) for c in range(4)] + FALLING,
    [(10, 5), (11, 4), (11, 5), (11, 6)],
)


def grid(cells: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros((ROWS, COLS), dtype=bool)
    for r, c in cells:
        out[r, c] = True
    return out


def read(
    style: Style,
    solid: list[tuple[int, int]],
    ghost: list[tuple[int, int]] | None,
    score: float = 0.30,
    seed: int = 3,
) -> tuple[np.ndarray, float]:
    """Render one scene and classify it against the theme's true ground."""
    image = render_board(
        grid(solid),
        style,
        cell_size=24,
        seed=seed,
        ghost=None if ghost is None else grid(ghost),
        ghost_score=score,
    )[:, :, ::-1]
    background = np.asarray(style.background, dtype=np.float64)[::-1]
    return classify_grid(image, rows=ROWS, cols=COLS, background=background)


def named_layer(
    style: Style,
    solid: list[tuple[int, int]],
    ghost: list[tuple[int, int]],
    score: float,
) -> set[tuple[int, int]] | None:
    """The cells the rule calls a landing preview on this scene, or None.

    Asked of the rule directly, because "the cells read empty" is not the
    same claim: a faint piece color on a board whose other pieces are far
    from the ground lands in the empty class under the ORDINARY split too,
    with or without this rule. What must be pinned for the scenes below is
    which cells the rule itself takes responsibility for deleting.
    """
    image = render_board(
        grid(solid), style, cell_size=24, seed=3, ghost=grid(ghost), ghost_score=score
    )[:, :, ::-1]
    background = np.asarray(style.background, dtype=np.float64)[::-1]
    scores = _distance_scores(_cell_colors(image, ROWS, COLS, 0.25), background)
    layer = _ghost_layer(scores, frozenset())
    if layer is None:
        return None
    return {(int(r), int(c)) for r, c in zip(*np.nonzero(layer), strict=True)}


@pytest.mark.parametrize("style", STYLES, ids=lambda s: s.name)
@pytest.mark.parametrize("scene", sorted(SCENES), ids=str)
@pytest.mark.parametrize("score", NAMED, ids=lambda v: f"score{v}")
def test_a_preview_in_the_open_is_named_and_costs_no_confidence(
    style: Style, scene: str, score: float
) -> None:
    solid, ghost = SCENES[scene]
    assert named_layer(style, solid, ghost, score) == set(ghost)
    occupancy, confidence = read(style, solid, ghost, score)
    np.testing.assert_array_equal(
        occupancy, grid(solid), err_msg=f"{style.name}/{scene}: the board did not read exactly"
    )
    # Direction of the whole fix: a frame with a preview on it is not an
    # ambiguous frame. Measured without the preview, the same board reads
    # 0.52-0.96 depending on the theme; with it, the same, to within noise.
    _bare, baseline = read(style, solid, None)
    assert confidence >= baseline - 0.01, (
        f"{style.name}/{scene}: the preview cost confidence ({confidence:.2f} vs {baseline:.2f})"
    )


@pytest.mark.parametrize("style", STYLES, ids=lambda s: s.name)
@pytest.mark.parametrize("scene", sorted(SCENES), ids=str)
def test_a_preview_too_faint_to_see_needs_no_rule(style: Style, scene: str) -> None:
    # Under the band the preview is at the background by every measure the
    # module has, so the ordinary split already calls it empty. Pinned so
    # that a widened band shows up as a behavior change here rather than
    # silently.
    solid, ghost = SCENES[scene]
    assert named_layer(style, solid, ghost, FAINT) is None
    occupancy, confidence = read(style, solid, ghost, FAINT)
    np.testing.assert_array_equal(occupancy, grid(solid))
    assert confidence > 0.15


@pytest.mark.parametrize("style", STYLES, ids=lambda s: s.name)
@pytest.mark.parametrize("scene", sorted(SCENES), ids=str)
@pytest.mark.parametrize("score", SOLID_LOOKING, ids=lambda v: f"score{v}")
def test_a_preview_as_opaque_as_a_piece_is_held_not_guessed(
    style: Style, scene: str, score: float
) -> None:
    # The rule holds rather than guesses. A preview drawn this opaque is
    # indistinguishable from a played piece, so it stays CONTENT: every one
    # of its cells reads occupied, and the stack under and around it is
    # untouched. Downstream that frame is a tetromino the tracker cannot
    # explain, so it holds state and keeps the last hint — which is the
    # behavior this game had before the rule existed.
    solid, ghost = SCENES[scene]
    assert named_layer(style, solid, ghost, score) is None
    occupancy, _confidence = read(style, solid, ghost, score)
    assert occupancy[grid(ghost)].all(), f"{style.name}/{scene}: an opaque preview was deleted"
    assert occupancy[grid(solid)].all(), f"{style.name}/{scene}: stack cells were lost"


@pytest.mark.parametrize("style", STYLES, ids=lambda s: s.name)
@pytest.mark.parametrize("score", (FAINT, *NAMED, *SOLID_LOOKING), ids=lambda v: f"score{v}")
def test_the_stack_walls_a_preview_in_and_the_rule_declines(style: Style, score: float) -> None:
    # A preview sitting in a notch of the stack and a piece played into
    # that notch are the same picture, so nothing may be deleted there.
    # The frame reads as it does without the rule: the stack is whole
    # whatever the opacity.
    solid, ghost = WELL
    assert named_layer(style, solid, ghost, score) is None, f"{style.name}: named in a well"
    occupancy, _confidence = read(style, solid, ghost, score)
    assert occupancy[grid(solid)].all(), f"{style.name}: stack cells were lost in the well"


@pytest.mark.parametrize("style", STYLES, ids=lambda s: s.name)
@pytest.mark.parametrize("score", (FAINT, *NAMED, *SOLID_LOOKING), ids=lambda v: f"score{v}")
def test_a_pale_piece_the_stack_runs_into_is_kept(style: Style, score: float) -> None:
    # The inverse error, on the shape that produced it for real: a piece
    # whose color lands in the band, resting on the floor with the stack
    # continuing straight into its left edge (roas_stacker's pale
    # periwinkle T, 0.346 against the ghost's 0.320 — no threshold
    # separates those, only the neighbours do). It is content and must
    # survive at every opacity.
    solid, pale = ABUTTING
    assert named_layer(style, solid, pale, score) is None, f"{style.name}: a real piece was named"
    occupancy, _confidence = read(style, solid, pale, score)
    assert occupancy[grid(solid)].all(), f"{style.name}: stack cells were lost"


@pytest.mark.parametrize("style", STYLES, ids=lambda s: s.name)
def test_a_ghostless_theme_never_reaches_the_band_at_all(style: Style) -> None:
    # Why the existing style matrix is untouched, stated as a measurement
    # rather than as a hope: with no preview drawn, no cell of any of these
    # themes scores inside the band the rule looks at, at any cell size, so
    # the rule cannot fire and the two-class path is the only one that runs.
    # (Measured over 840 random boards across the matrix: 0 band cells.)
    rng = np.random.default_rng(7)
    background = np.asarray(style.background, dtype=np.float64)[::-1]
    for seed in range(8):
        board = rng.random((ROWS, COLS)) < 0.35
        for cell_size in (18, 24, 31):
            image = render_board(board, style, cell_size=cell_size, seed=seed)[:, :, ::-1]
            scores = _distance_scores(_cell_colors(image, ROWS, COLS, 0.25), background)
            in_band = (scores >= _GHOST_SEPARATION) & (scores < MIN_SPREAD)
            assert not in_band.any(), f"{style.name}: cell scores {np.unique(scores[in_band])}"
            occupancy, _confidence = classify_grid(
                image, rows=ROWS, cols=COLS, background=background
            )
            np.testing.assert_array_equal(occupancy, board)


def test_the_preview_renderer_draws_a_piece_the_game_would_hide() -> None:
    # The fixture's own arrangement: a piece drawn over its own preview
    # hides it, so the band holds nothing and the rule is a no-op.
    style = STYLES[0]
    solid, ghost = SCENES["floor"]
    assert named_layer(style, solid + ghost, ghost, NAMED[0]) is None
    occupancy, _confidence = read(style, solid + ghost, ghost, NAMED[0])
    np.testing.assert_array_equal(occupancy, grid(solid + ghost))
    # And the caption helper still renders (shared with the preview-box
    # tests; kept exercised here so the ghost argument did not break it).
    assert with_label(render_board(grid(solid), style), style).shape[2] == 3
