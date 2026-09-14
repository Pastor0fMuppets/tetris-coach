from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from tetris_coach.core.pieces import PIECES, ROTATIONS
from tetris_coach.vision.grid import (
    MIN_SPREAD,
    _distance_scores,
    classify_grid,
    otsu_threshold_hist,
)
from tetris_coach.vision.pieces_vision import _preview_blocks, identify_next, match_cells

from .synthetic import STYLES, render_board, render_next_preview, with_label

FIXTURES = Path(__file__).parent / "fixtures"
LIVE = FIXTURES / "live_session"


def preview_mask(image: np.ndarray) -> np.ndarray | None:
    """identify_next's raw threshold mask, before the blocks are picked out."""
    img = np.asarray(image)
    background = np.median(img.reshape(-1, img.shape[2]), axis=0)
    scores = _distance_scores(img, background)
    flat = scores.ravel()
    if float(flat.max()) - float(flat.min()) < MIN_SPREAD:
        return None
    return scores > max(otsu_threshold_hist(flat), MIN_SPREAD)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def live_preview(number: str) -> np.ndarray:
    """One captured preview crop as the capture pipeline hands it over (BGR)."""
    return np.asarray(Image.open(LIVE / f"next_{number}.png"))[:, :, ::-1]


class TestMatchCells:
    def test_all_rotations_match(self) -> None:
        for piece, rots in ROTATIONS.items():
            for rot in rots:
                # Shifted anywhere on the board, the shape still matches.
                cells = {(r + 5, c + 3) for r, c in rot.cells}
                assert match_cells(cells) == (piece, rot.index)

    def test_non_tetromino_shapes_do_not_match(self) -> None:
        assert match_cells({(0, 0), (0, 1), (0, 2)}) is None  # 3 cells
        assert match_cells({(0, 0), (0, 1), (1, 0), (2, 2)}) is None
        assert match_cells({(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)}) is None


class TestIdentifyNext:
    @pytest.mark.parametrize("style", STYLES, ids=lambda s: s.name)
    @pytest.mark.parametrize("piece", PIECES)
    def test_spawn_orientation_recognized(self, piece: str, style) -> None:  # type: ignore[no-untyped-def]
        image = render_next_preview(ROTATIONS[piece][0].cells, style, cell_size=20)
        assert identify_next(image) == piece

    @pytest.mark.parametrize("piece", PIECES)
    def test_any_rotation_recognized(self, piece: str) -> None:
        for rot in ROTATIONS[piece]:
            image = render_next_preview(rot.cells, STYLES[0], cell_size=16)
            assert identify_next(image) == piece

    @pytest.mark.parametrize("cell_size", [10, 18, 27])
    def test_various_cell_sizes(self, cell_size: int) -> None:
        image = render_next_preview(ROTATIONS["L"][0].cells, STYLES[2], cell_size=cell_size)
        assert identify_next(image) == "L"

    def test_empty_preview_returns_none(self) -> None:
        image = np.full((80, 120, 3), 12, dtype=np.uint8)
        assert identify_next(image) is None

    @pytest.mark.parametrize("style", STYLES, ids=lambda s: s.name)
    def test_blank_rendered_preview_returns_none(self, style) -> None:  # type: ignore[no-untyped-def]
        # A pieceless preview box rendered WITH the style's gridlines and
        # noise must read as "no piece" — via the spread gate where the
        # gridlines sit below the uniformity floor, or via the shape-match
        # fallback where they poke above it (a lattice is no tetromino).
        image = render_board(np.zeros((4, 6), dtype=bool), style, cell_size=20)
        assert identify_next(image) is None

    def test_blank_light_preview_fixture_returns_none(self) -> None:
        # Real capture of the target game's empty preview box (white
        # theme): the old absolute score read the whole bright box as
        # foreground; it must read as "no piece", not garbage.
        image = np.asarray(Image.open(FIXTURES / "roas_stacker" / "blank_light_preview.png"))
        assert identify_next(image) is None

    def test_preview_pipeline_matches_grid_pipeline(self) -> None:
        # Sanity: classify_grid and identify_next agree on what is foreground.
        image = render_next_preview(ROTATIONS["O"][0].cells, STYLES[1], cell_size=22)
        occupancy, _ = classify_grid(image, rows=4, cols=6)
        assert occupancy.sum() == 4
        assert identify_next(image) == "O"


class TestIdentifyNextRealPreview:
    """Bug 2, on the committed crops of the real failing session.

    Measured on these crops before the fix: identify_next returned None on
    96 of 96 frames, so the tracker logged next=- all session and every
    hint the coach gave was 1-ply.
    """

    def test_the_real_crops_are_identified(self) -> None:
        # next_00043/00120 are a yellow-green O (2x2 of 19 px cells);
        # next_00063 is a blue horizontal I (4x1). Each sits well off
        # center in a 94x94 box, under a faint grey "NEXT" caption.
        assert identify_next(live_preview("00043")) == "O"
        assert identify_next(live_preview("00063")) == "I"
        assert identify_next(live_preview("00120")) == "O"

    def test_every_frame_of_the_live_window_is_identified(self) -> None:
        # The whole 96-frame window, in order: the preview holds the O
        # while the I falls, flips to I on frame 59 (the frame the I hard-
        # drops and the O spawns), and back to O when the I is dealt.
        seen = [identify_next(live_preview(f"{n:05d}")) for n in range(40, 136)]
        assert None not in seen
        runs = [(piece, sum(1 for _ in group)) for piece, group in _runs(seen)]
        assert runs == [("O", 19), ("I", 51), ("O", 26)]

    def test_the_caption_is_in_the_mask_and_out_of_the_blocks(self) -> None:
        # The diagnosis itself, pinned: the grey caption scores as far from
        # the white ground as the piece does, so it survives the threshold
        # and the raw bounding box is 45x79 for a piece that is 19x79 —
        # which is what broke every rotation's fit. The block filter is
        # what removes it; no threshold can.
        mask = preview_mask(live_preview("00063"))
        assert mask is not None
        assert bbox(mask) == (0, 12, 45, 91)
        blocks = _preview_blocks(mask)
        assert blocks is not None
        assert bbox(blocks) == (26, 12, 45, 91)

    def test_a_blank_box_has_no_blocks_at_all(self) -> None:
        # The other side of the same rule: with no piece in the box there
        # is nothing block-like to find, whatever the caption does.
        blank = np.asarray(Image.open(FIXTURES / "roas_stacker" / "blank_light_preview.png"))
        mask = preview_mask(blank)
        assert mask is None or _preview_blocks(mask) is None


def _runs(values: list[str | None]) -> list[tuple[str | None, list[str | None]]]:
    runs: list[tuple[str | None, list[str | None]]] = []
    for value in values:
        if runs and runs[-1][0] == value:
            runs[-1][1].append(value)
        else:
            runs.append((value, [value]))
    return runs


class TestIdentifyNextFurniture:
    """A preview box holds more than the piece, and is not built around it."""

    @pytest.mark.parametrize("style", STYLES, ids=lambda s: s.name)
    @pytest.mark.parametrize("piece", PIECES)
    def test_captioned_preview_recognized(self, piece: str, style) -> None:  # type: ignore[no-untyped-def]
        # Every style with the caption the real game draws. The caption is
        # IN the threshold mask for all of them (see the fixture test
        # above for the same thing on real pixels).
        for rot in ROTATIONS[piece]:
            image = render_next_preview(rot.cells, style, cell_size=20, label="NEXT")
            assert identify_next(image) == piece

    @pytest.mark.parametrize("style", STYLES, ids=lambda s: s.name)
    def test_caption_only_preview_returns_none(self, style) -> None:  # type: ignore[no-untyped-def]
        # An empty box still carries its caption: "NEXT" is four shapes in
        # a row, which is exactly the arrangement of a horizontal I.
        image = with_label(render_board(np.zeros((4, 6), dtype=bool), style, cell_size=20), style)
        assert identify_next(image) is None

    @pytest.mark.parametrize("box", [(4, 6), (6, 8), (8, 8)])
    @pytest.mark.parametrize("offset", [(0, 0), (1, 3), (2, 1)])
    def test_piece_anywhere_in_any_box(self, box, offset) -> None:  # type: ignore[no-untyped-def]
        # Nothing assumes the piece is centered, fills the crop, or that
        # the crop is square: the real one is none of those.
        for piece in PIECES:
            rot = ROTATIONS[piece][0]
            if offset[0] + rot.height > box[0] or offset[1] + rot.width > box[1]:
                continue
            image = render_next_preview(
                rot.cells, STYLES[4], cell_size=18, box_cells=box, offset=offset, label="NEXT"
            )
            assert identify_next(image) == piece

    @pytest.mark.parametrize("count", [2, 4])
    def test_a_row_of_hollow_blobs_is_not_a_piece(self, count: int) -> None:
        # Bold text is the adversarial case for a shape rule: 4 glyphs in a
        # row read as an I, 2 as half an O. A drawn cell is SOLID and
        # SQUARE, and a glyph is neither — measured, the sampled fill of a
        # hollow blob is 0.60 where every real piece cell reads 1.00.
        image = Image.new("RGB", (140, 90), (250, 250, 250))
        draw = ImageDraw.Draw(image)
        for index in range(count):
            left = 10 + index * 26
            draw.rectangle([left, 20, left + 18, 50], fill=(120, 120, 125))
            draw.rectangle([left + 5, 26, left + 13, 44], fill=(250, 250, 250))
        assert identify_next(np.asarray(image)) is None
