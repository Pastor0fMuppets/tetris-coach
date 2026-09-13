from tetris_coach.core.board import HEIGHT, Board
from tetris_coach.core.pieces import ROTATIONS
from tetris_coach.solver.evaluate import (
    DELLACHERIE,
    Weights,
    evaluate_drop,
    landing_height,
)

from .test_board import board_from_strings, horizontal_i


def test_classic_dellacherie_default_weights() -> None:
    assert DELLACHERIE == Weights(
        landing_height=-1.0,
        eroded_cells=1.0,
        row_transitions=-1.0,
        column_transitions=-1.0,
        holes=-4.0,
        cumulative_wells=-1.0,
    )


def test_weights_are_overridable() -> None:
    w = Weights(holes=-10.0)
    assert w.holes == -10.0
    assert w.eroded_cells == 1.0


def test_landing_height_flat_i_on_floor() -> None:
    res = Board().drop(horizontal_i(), 0)
    assert res is not None
    # A 1-row-tall piece resting on the floor: midpoint altitude 0.
    assert landing_height(res, horizontal_i()) == 0.0


def test_landing_height_o_on_floor() -> None:
    o = ROTATIONS["O"][0]
    res = Board().drop(o, 0)
    assert res is not None
    # 2-row-tall piece on the floor: rows 18-19, midpoint altitude 0.5.
    assert landing_height(res, o) == 0.5


def test_landing_height_measured_pre_clear() -> None:
    b = board_from_strings("######....")
    res = b.drop(horizontal_i(), 6)
    assert res is not None
    assert res.lines_cleared == 1
    # The bar landed on the floor row even though the row then cleared.
    assert landing_height(res, horizontal_i()) == 0.0


def test_evaluate_drop_matches_manual_sum() -> None:
    res = Board().drop(horizontal_i(), 3)
    assert res is not None
    b = res.board
    expected = (
        -1.0 * 0.0  # landing height
        + 1.0 * 0.0  # no erosion
        - b.row_transitions()
        - b.column_transitions()
        - 4.0 * b.hole_count()
        - b.cumulative_wells()
    )
    assert evaluate_drop(res, horizontal_i()) == expected


def test_line_clear_scores_better_than_hole() -> None:
    # Bottom row missing only column 9.
    b = board_from_strings(
        "#########.",
    )
    vertical = next(r for r in ROTATIONS["I"] if r.width == 1)
    clear = b.drop(vertical, 9)
    bury = b.drop(horizontal_i(), 0)  # lands on top, buries nothing but stacks
    assert clear is not None and bury is not None
    assert evaluate_drop(clear, vertical) > evaluate_drop(bury, horizontal_i())


def test_eroded_cells_metric_uses_piece_contribution() -> None:
    # Clearing with more of the piece's own cells in the cleared line
    # scores a bigger erosion term.
    b = board_from_strings("######....")
    res = b.drop(horizontal_i(), 6)
    assert res is not None
    assert res.lines_cleared * res.eroded_cells == 4


def test_holes_weight_dominates() -> None:
    # Same landing spot; a custom zero weight for holes must change ranking.
    b = board_from_strings(
        "#.........",
        "#.........",
    )
    s = ROTATIONS["S"][0]  # bottom row cols 0-1, top row cols 1-2
    res = b.drop(s, 0)
    assert res is not None
    with_holes = evaluate_drop(res, s)
    without_holes_weight = evaluate_drop(res, s, Weights(holes=0.0))
    assert without_holes_weight - with_holes == 4.0 * res.board.hole_count()
    assert res.board.hole_count() > 0


def test_landing_row_altitude_consistency() -> None:
    vertical = next(r for r in ROTATIONS["I"] if r.width == 1)
    res = Board().drop(vertical, 5)
    assert res is not None
    # Vertical I occupies rows 16-19; midpoint altitude (0+3)/2 = 1.5.
    assert res.landing_row == HEIGHT - 4
    assert landing_height(res, vertical) == 1.5
