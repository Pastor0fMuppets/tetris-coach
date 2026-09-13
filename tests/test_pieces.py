from tetris_coach.core.pieces import PIECES, ROTATIONS

EXPECTED_ROTATION_COUNTS = {"I": 2, "O": 1, "T": 4, "S": 2, "Z": 2, "J": 4, "L": 4}


def test_seven_pieces() -> None:
    assert sorted(PIECES) == sorted("IOTSZJL")


def test_distinct_rotation_counts() -> None:
    counts = {piece: len(rots) for piece, rots in ROTATIONS.items()}
    assert counts == EXPECTED_ROTATION_COUNTS


def test_every_rotation_has_four_cells() -> None:
    for rots in ROTATIONS.values():
        for rot in rots:
            assert len(rot.cells) == 4
            assert len(set(rot.cells)) == 4


def test_rotations_are_normalized() -> None:
    for rots in ROTATIONS.values():
        for rot in rots:
            assert min(r for r, _ in rot.cells) == 0
            assert min(c for _, c in rot.cells) == 0
            assert max(r for r, _ in rot.cells) == rot.height - 1
            assert max(c for _, c in rot.cells) == rot.width - 1


def test_row_masks_match_cells() -> None:
    for rots in ROTATIONS.values():
        for rot in rots:
            assert len(rot.row_masks) == rot.height
            for row, mask in enumerate(rot.row_masks):
                cells_in_row = {c for r, c in rot.cells if r == row}
                assert mask == sum(1 << c for c in cells_in_row)
                assert mask != 0  # no empty rows in a bounding box


def test_bottom_and_top_profiles() -> None:
    for rots in ROTATIONS.values():
        for rot in rots:
            assert len(rot.bottom) == rot.width
            assert len(rot.top) == rot.width
            for col in range(rot.width):
                rows_in_col = [r for r, c in rot.cells if c == col]
                assert rows_in_col, "every bounding-box column must be occupied"
                assert rot.bottom[col] == max(rows_in_col)
                assert rot.top[col] == min(rows_in_col)


def test_s_and_z_differ() -> None:
    s_shapes = {rot.cells for rot in ROTATIONS["S"]}
    z_shapes = {rot.cells for rot in ROTATIONS["Z"]}
    assert not s_shapes & z_shapes


def test_i_piece_orientations() -> None:
    widths = sorted(rot.width for rot in ROTATIONS["I"])
    assert widths == [1, 4]
