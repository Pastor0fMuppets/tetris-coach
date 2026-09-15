from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from PIL import Image, ImageDraw

from tetris_coach.core.pieces import PIECES, ROTATIONS
from tetris_coach.vision.grid import (
    _GHOST_SEPARATION,
    HINT_PAINT,
    MIN_SPREAD,
    _distance_scores,
    classify_grid,
    otsu_threshold_hist,
)
from tetris_coach.vision.pieces_vision import (
    _piece_from_mask,
    _preview_blocks,
    identify_next,
    match_cells,
)

from .synthetic import STYLES, label_color, render_board, render_next_preview, with_label

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


class TestIdentifyNextCaptionBar:
    """A caption/divider drawn as a SOLID bar, not as glyph strokes.

    The block filter drops furniture that is thin (a letter's stroke, a
    hollow border, a gridline lattice) or small. A solid label bar is
    neither, so it reaches the grid derivation as if it were a cell — and
    a bar wide enough to BRIDGE the gap between two of the piece's cells
    merges them into one band, which the median band size used to hide.
    Measured before this was refused: a 40x9 bar over a horizontal I of
    15 px cells reads as a confident J (and 'NEXT' captions are exactly
    what the live game draws over its preview).
    """

    @staticmethod
    def box(bar: tuple[int, int] | None, cell: int = 15, inset: int = 1) -> np.ndarray:
        """A white preview box with a horizontal I and an optional grey bar."""
        image = np.full((73, 106, 3), 252, dtype=np.uint8)
        for index in range(4):
            top, left = 21 + inset, 20 + index * cell + inset
            image[top : 21 + cell - inset, left : 20 + (index + 1) * cell - inset] = (60, 110, 200)
        if bar is not None:
            width, height = bar
            image[2 : 2 + height, 2 : 2 + width] = 170
        return image

    def test_the_piece_alone_is_read(self) -> None:
        assert identify_next(self.box(None)) == "I"

    def test_a_solid_bar_never_renames_the_piece(self) -> None:
        # The whole bar geometry space around the reported failure: every
        # one of these used to answer J on a contiguous window of it.
        wrong = {
            (width, height): identify_next(self.box((width, height)))
            for width in range(20, 65)
            for height in range(5, 16)
        }
        assert {answer for answer in wrong.values()} <= {None, "I"}
        assert identify_next(self.box((40, 9))) is None  # the reported crop

    @pytest.mark.parametrize("style", STYLES, ids=lambda s: s.name)
    @pytest.mark.parametrize("piece", PIECES)
    def test_no_style_reads_a_bar_as_another_piece(self, piece: str, style) -> None:  # type: ignore[no-untyped-def]
        # Same thing over the style matrix, with the bar in the corner a
        # caption occupies — clear of the piece, which is where a game
        # draws it. The answer may be None (the box is no longer legible),
        # never a different piece.
        for rot in ROTATIONS[piece]:
            image = render_next_preview(
                rot.cells, style, cell_size=16, box_cells=(6, 8), offset=(2, 3)
            )
            assert identify_next(image) == piece
            for bar in ((30, 7), (48, 11), (60, 9)):
                barred = image.copy()
                barred[1 : 1 + bar[1], 1 : 1 + bar[0]] = label_color(style)
                assert identify_next(barred) in (piece, None)


PALE = FIXTURES / "pale_preview"
PALE_BOARD = FIXTURES / "pale_piece"

# The pale_preview window's frames, and what is in the box on each. The two
# unnamed ones are the deal animation: the box is mid-swap, holding the
# outgoing piece's panel sliding off over the incoming one (00273 and 00280
# score their whole crop above the floor, background estimate included).
PALE_WINDOW = [
    ("00273", None),
    ("00280", None),
    *((f"00{n}", "T") for n in (290, 300, 477, 485, 495)),
    *((f"00{n}", "T") for n in range(565, 605, 5)),
]


def pale_preview(number: str, window: Path = PALE) -> np.ndarray:
    return np.asarray(Image.open(window / f"next_{number}.png"))[:, :, ::-1]


def band_split(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """The band, and the level the band pass splits out of it."""
    img = np.asarray(image)
    background = np.median(img.reshape(-1, img.shape[2]), axis=0)
    scores = _distance_scores(img, background)
    band = (scores >= _GHOST_SEPARATION) & (scores < MIN_SPREAD)
    level = otsu_threshold_hist(scores[band])
    return scores, band, float(level)


class TestIdentifyNextPalePiece:
    """Bug 3, on the committed crops of the session that went blind.

    A pale periwinkle piece in the box scores under the uniformity floor
    the threshold is anchored at, so the mask kept nothing and the box
    read as empty: measured over the 705-frame session behind
    ``tests/fixtures/pale_preview``, ``identify_next`` returned None on
    226 frames (32%), in runs up to 45. The same failure the BOARD had
    until ``vision.grid`` stopped letting a threshold decide the
    intermediate band.
    """

    def test_the_pale_previews_are_named(self) -> None:
        # The headline, on the frames the README names: next_00580 plainly
        # shows a T, and every readable frame of the window is a T.
        assert identify_next(pale_preview("00580")) == "T"
        assert [(n, identify_next(pale_preview(n))) for n, _ in PALE_WINDOW] == PALE_WINDOW

    def test_the_pale_piece_window_is_named_too(self) -> None:
        # The board-side window's own preview crops: 47 of 61, all T. The
        # 14 that are not are the game's end-of-round panel (00687-00700),
        # which covers the box as well as the board.
        seen = [identify_next(pale_preview(f"00{n}", PALE_BOARD)) for n in range(640, 701)]
        assert seen == ["T"] * 47 + [None] * 14

    def test_it_is_the_band_that_names_them_not_the_threshold(self) -> None:
        # The diagnosis, pinned. The thresholded mask is anchored at
        # MIN_SPREAD and the piece is UNDER it: 69 of 10098 pixels
        # survive, all of them caption, and the shape rules have nothing
        # to read. The answer comes from the band instead.
        image = pale_preview("00580")
        mask = preview_mask(image)
        assert mask is not None
        assert int(mask.sum()) == 69
        assert _piece_from_mask(mask) is None
        assert identify_next(image) == "T"

    def test_the_band_is_split_because_the_box_has_furniture_in_it(self) -> None:
        # Why the band is not simply handed over whole. A board cell's
        # patch mean never sees the box's own hairline border and
        # gridlines; in the crop they score 0.179-0.250, under the piece
        # at 0.254-0.349 and touching it. Taken whole the band merges the
        # piece's cells through that border and nothing block-like is
        # left; split at 0.252, the cells stand alone.
        scores, band, level = band_split(pale_preview("00580"))
        furniture, piece = band & (scores <= level), band & (scores > level)
        assert 0.17 < float(scores[furniture].max()) < level < float(scores[piece].min())
        assert _piece_from_mask(band) is None
        assert _piece_from_mask(piece) == "T"

    def test_the_caption_never_reaches_the_band_pass(self) -> None:
        # The guard that survives the change, and why it does not need a
        # rule of its own: the caption's core is SOLID class (0.586, well
        # above the floor), so only its antialiased skirt is in the band —
        # a hollow outline the block filter drops. The blocks the band
        # pass reads are the piece's four cells and nothing else.
        image = pale_preview("00580")
        scores, band, level = band_split(image)
        assert float(scores[:12, :40].max()) > MIN_SPREAD  # the caption, solid class
        blocks = _preview_blocks(band & (scores > level))
        assert blocks is not None
        assert bbox(blocks) == (11, 14, 52, 73)  # the piece; the caption is at y < 8

    def test_the_readable_frames_are_read_the_same_as_before(self) -> None:
        # The band pass is asked only where the threshold came back
        # empty-handed, so every crop that was readable is byte-identical.
        # Measured over every committed preview crop: the band, read on
        # its own, contradicts the threshold nowhere.
        for window in ("live_session", "ghost_session", "absorbed_piece"):
            for path in sorted((FIXTURES / window).glob("next_*.png")):
                image = np.asarray(Image.open(path))[:, :, ::-1]
                scores, band, level = band_split(image)
                assert _piece_from_mask(band & (scores > level)) is None
                assert identify_next(image) is not None


class TestIdentifyNextOwnPaint:
    """The one thing in the box that must never be read as a piece.

    The coach paints its hint over the game and captures the screen
    again, and the preview panel floats over the top corner of the
    playfield — so a hint drawn in that corner is drawn over the BOX. It
    is a tetromino of square cells, in the place a tetromino is expected,
    and over a light box it composites INTO the band this pass reads.
    """

    @staticmethod
    def box(cells: tuple[tuple[int, int], ...], color: tuple[int, int, int]) -> np.ndarray:
        """A white preview box with ``cells`` painted in ``color``."""
        image = np.full((96, 96, 3), 252, dtype=np.uint8)
        image[:, :, 2] = 251
        for r, c in cells:
            top, left = 8 + r * 19 + 1, 12 + c * 19 + 1
            image[top : top + 17, left : left + 17] = color
        return image

    @staticmethod
    def composite() -> tuple[int, int, int]:
        """What the shipped hint fill looks like over this box's ground."""
        assert HINT_PAINT is not None
        background = np.array([252.0, 252.0, 251.0])
        paint = np.asarray(HINT_PAINT.color, dtype=np.float64)
        composite = background + HINT_PAINT.opacity * (paint - background)
        return tuple(round(float(v)) for v in composite)  # type: ignore[return-value]

    def test_a_pale_piece_in_this_box_is_named(self) -> None:
        # The control: the same box, the same geometry, the session's own
        # periwinkle. The guard must not be what reads it.
        assert identify_next(self.box(ROTATIONS["T"][0].cells, (251, 224, 206))) == "T"

    def test_our_own_hint_is_refused_rather_than_named(self) -> None:
        # Every rotation of every piece, painted in our own fill: the box
        # is unreadable while our overlay is on it, and None is this
        # module's word for that. Naming it would report the piece the
        # coach is POINTING AT as the piece coming next.
        for piece in PIECES:
            for rot in ROTATIONS[piece]:
                assert identify_next(self.box(rot.cells, self.composite())) is None

    def test_it_is_the_paint_rule_and_not_the_shape_rules(self) -> None:
        # Pinned: the shape rules find a perfectly good piece there. It is
        # the color arithmetic that refuses it, and a session run with a
        # different --hint-color (or none) gets the shape answer back.
        image = self.box(ROTATIONS["J"][0].cells, self.composite())
        assert identify_next(image, own_paint=None) == "J"

    def test_the_composite_is_in_the_band_this_pass_reads(self) -> None:
        # The measurement that makes the rule necessary: our own fill over
        # a white box lands four thousandths from the session's pale
        # piece, on the very scale the band is defined by.
        scores, band, _level = band_split(self.box(((0, 0),), self.composite()))
        assert _GHOST_SEPARATION <= float(scores[band].max()) < MIN_SPREAD


def band_candidate(image: np.ndarray) -> np.ndarray:
    """Exactly the level the band pass hands to the shape rules."""
    scores, band, level = band_split(image)
    candidate = band & (scores > level)
    return candidate if candidate.any() else band


class TestIdentifyNextBandIsNotHandedARectangle:
    """What the band may not do: name a piece out of a lone rectangle.

    A band cell divided into cells by assertion — the FLUSH hypothesis —
    makes any solid rectangle a piece: a square is an O, a 4:1 bar an I.
    The threshold can afford that, because a solid class in a preview box
    is nearly always the piece. The band cannot: it is where everything
    the threshold refused arrives, and a box is full of rectangles that
    are not pieces.
    """

    # Panel/well shade pairs whose well lands in the band, with the score
    # the well reads at against the panel. The ordinary preview skin.
    WELLS: ClassVar = [
        ((250, 250, 250), (235, 235, 235), 0.243),
        ((250, 250, 250), (226, 226, 226), 0.307),
        ((30, 30, 34), (18, 18, 20), 0.223),
        ((30, 30, 34), (46, 46, 52), 0.256),
        ((120, 120, 120), (100, 100, 100), 0.280),
        ((200, 200, 205), (178, 178, 182), 0.296),
    ]
    # A piece pale enough to need the band, and a caption bar ABOVE it —
    # still in the band, but further from the ground than the piece is, so
    # the band's upper class is the BAR and the piece is dropped.
    PALE_PIECE: ClassVar = (232, 232, 232)  # 0.280 on a 252 ground
    PALE_BAR: ClassVar = (230, 230, 230)  # 0.294: above the piece, in the band

    @staticmethod
    def panelled(panel: tuple[int, int, int], well: tuple[int, int, int]) -> np.ndarray:
        """An EMPTY preview box: a panel around an inner well. No piece."""
        image = np.full((96, 96, 3), panel, dtype=np.uint8)
        image[20:76, 24:72] = well
        return image

    @staticmethod
    def with_piece(
        cells: tuple[tuple[int, int], ...], color: tuple[int, int, int], cell: int = 15
    ) -> np.ndarray:
        image = np.full((96, 96, 3), 252, dtype=np.uint8)
        for r, c in cells:
            top, left = 40 + r * cell + 1, 20 + c * cell + 1
            image[top : top + cell - 2, left : left + cell - 2] = color
        return image

    @pytest.mark.parametrize(("panel", "well", "score"), WELLS)
    def test_an_empty_panelled_box_is_not_an_o(self, panel, well, score: float) -> None:  # type: ignore[no-untyped-def]
        # The box's own skin, with nothing in it: the well is a rectangle
        # in the band, and an even division of a rectangle is an O.
        image = self.panelled(panel, well)
        scores, band, _level = band_split(image)
        assert _GHOST_SEPARATION <= float(scores[band].max()) < MIN_SPREAD
        assert round(float(scores[band].max()), 3) == pytest.approx(score, abs=0.01)
        assert identify_next(image) is None

    def test_a_caption_bar_above_the_piece_is_not_an_i(self) -> None:
        # The band keeps its UPPER class, and nothing says the box's
        # furniture sits BELOW the piece. When it sits above, the
        # candidate IS the furniture — and a solid bar is the picture of
        # a horizontal I. Swept over the bar geometries a caption
        # occupies: never a name, for any piece actually in the box.
        seen = set()
        for width in range(20, 65, 4):
            for height in range(5, 16, 2):
                for piece in PIECES:
                    image = self.with_piece(ROTATIONS[piece][0].cells, self.PALE_PIECE)
                    image[4 : 4 + height, 6 : 6 + width] = self.PALE_BAR
                    seen.add((piece, identify_next(image)))
        assert {answer for _piece, answer in seen} <= {None}

    def test_the_bar_is_what_the_band_keeps(self) -> None:
        # The diagnosis behind that test, pinned: both the bar and the
        # piece are in the band, the split puts the BAR on top, and the
        # piece is not in the candidate at all.
        image = self.with_piece(ROTATIONS["T"][0].cells, self.PALE_PIECE)
        image[4:13, 6:46] = self.PALE_BAR
        candidate = band_candidate(image)
        assert candidate[4:13, 6:46].all()  # the bar
        assert not candidate[40:, :].any()  # the piece, dropped
        assert _piece_from_mask(candidate) == "I"  # what the flush rule used to answer
        assert _piece_from_mask(candidate, flush=False) is None

    def test_a_lone_rectangle_is_never_a_piece(self) -> None:
        # The shape every one of these cases reduces to: one solid
        # rectangle in the band, which an even division reads as an O
        # (square) or an I (4:1). A preview box is full of them — a well,
        # a caption bar, a badge, the first visible sliver of a piece
        # still scrolling in — and none of them is a piece.
        named = {}
        for height in range(4, 40, 2):
            for width in range(4, 64, 2):
                image = np.full((96, 96, 3), 252, dtype=np.uint8)
                image[30 : 30 + height, 20 : 20 + width] = self.PALE_PIECE
                answer = identify_next(image)
                if answer is not None:
                    named[(width, height)] = answer
        assert named == {}

    @pytest.mark.parametrize("piece", PIECES)
    def test_a_piece_scrolling_in_is_not_named_from_a_sliver(self, piece: str) -> None:
        # A deal animation slides the next piece into the box. Its first
        # visible rows are a rectangle, and a rectangle is an O or an I —
        # named before the piece is whole, on consecutive captures, so
        # the tracker's two-frame debounce does not filter it.
        for cell in (12, 16, 20):
            for dy in range(-4 * cell, 0):
                image = np.full((96, 96, 3), 252, dtype=np.uint8)
                drawn = 0
                for r, c in ROTATIONS[piece][0].cells:
                    top, left = 30 + dy + r * cell + 1, 12 + c * cell + 1
                    bottom = min(96, top + cell - 2)
                    if bottom > max(0, top):
                        image[max(0, top) : bottom, left : left + cell - 2] = self.PALE_PIECE
                        drawn += 1
                answer = identify_next(image)
                assert answer in (None, piece)
                if drawn < 4:  # not all of it is in the box yet
                    assert answer is None

    def test_the_committed_crops_never_needed_the_flush_hypothesis(self) -> None:
        # The cost of the rule, measured rather than argued: over every
        # committed preview crop, the band's candidate reads the same
        # with the flush hypothesis and without it. What the rule costs a
        # real session here is nothing.
        for window in (
            "live_session",
            "ghost_session",
            "absorbed_piece",
            "pale_preview",
            "pale_piece",
            "spawn_latency",
            "ghost_beside_stack",
        ):
            for path in sorted((FIXTURES / window).glob("next_*.png")):
                image = np.asarray(Image.open(path))[:, :, ::-1]
                candidate = band_candidate(image)
                assert _piece_from_mask(candidate, flush=False) == _piece_from_mask(candidate)


class TestIdentifyNextOwnPaintOverAWell:
    """The same fill, on a box drawn in TWO shades: a panel and an inner well.

    The rule needs the color the paint landed ON, and the crop's own
    estimate of that is the per-channel median of every pixel. With a
    panel around an inner well — the ordinary preview skin — the median
    sits on whichever has more pixels, and a hint drawn inside the well
    is composited over the other one. Measured, the two composites stand
    ~21 uint8 units apart against a tolerance of 8.0: the rule said "not
    our paint" and the box was named as the placement on screen.
    """

    SHADES: ClassVar = [
        ((250, 250, 250), (235, 235, 235)),
        ((30, 30, 34), (18, 18, 20)),
        ((120, 120, 120), (100, 100, 100)),
        ((200, 200, 205), (186, 186, 190)),
    ]

    @staticmethod
    def box(
        panel: tuple[int, int, int],
        well: tuple[int, int, int],
        cells: tuple[tuple[int, int], ...],
        color: tuple[int, int, int] | None = None,
    ) -> np.ndarray:
        """A panelled box whose inner well holds ``cells`` in our own fill."""
        assert HINT_PAINT is not None
        ground = np.asarray(well, dtype=np.float64)
        paint = np.asarray(HINT_PAINT.color, dtype=np.float64)
        fill = np.round(ground + HINT_PAINT.opacity * (paint - ground)).astype(np.uint8)
        image = np.full((96, 96, 3), panel, dtype=np.uint8)
        image[6:62, 8:66] = well
        for r, c in cells:
            top, left = 10 + r * 17 + 1, 12 + c * 17 + 1
            image[top : top + 15, left : left + 15] = fill if color is None else color
        return image

    @pytest.mark.parametrize(("panel", "well"), SHADES)
    @pytest.mark.parametrize("piece", PIECES)
    def test_our_own_hint_is_refused_over_the_well(self, piece: str, panel, well) -> None:  # type: ignore[no-untyped-def]
        for rot in ROTATIONS[piece]:
            assert identify_next(self.box(panel, well, rot.cells)) is None

    @pytest.mark.parametrize(("panel", "well"), SHADES)
    def test_the_second_shade_is_what_used_to_defeat_it(self, panel, well) -> None:  # type: ignore[no-untyped-def]
        # The diagnosis, pinned to the arithmetic rather than to a
        # reading: the composite over the panel (where the median sits)
        # and the composite over the well (where the paint actually
        # landed) stand further apart than the tolerance, so the ground
        # the box averages to is not a ground anything was painted over.
        assert HINT_PAINT is not None
        composites = []
        for shade in (panel, well):
            ground = np.asarray(shade, dtype=np.float64)
            composites.append(ground + HINT_PAINT.opacity * (np.asarray(HINT_PAINT.color) - ground))
        assert float(np.linalg.norm(composites[0] - composites[1])) > HINT_PAINT.tolerance

    @pytest.mark.parametrize(("panel", "well"), SHADES)
    def test_a_real_piece_in_the_well_is_still_named(self, panel, well) -> None:  # type: ignore[no-untyped-def]
        # The control, and the cost of widening what the rule may match:
        # a game piece in the same well is named exactly as before.
        image = self.box(panel, well, ROTATIONS["J"][0].cells, color=(215, 15, 55))
        assert identify_next(image) == "J"


class TestIdentifyNextOwnPaintOnADarkBox:
    """The same hint fill, on a box dark enough to make it a SOLID class.

    The band is where the composite lands over a LIGHT box. It is not
    where it lands over a dark one: the composite is a distance from the
    box's own ground, and over black the shipped fill scores 0.374 —
    above MIN_SPREAD, so the box is read by the THRESHOLD, which used to
    name it. That named the placement the coach is pointing at as the
    piece coming next, on every dark theme.
    """

    GROUNDS: ClassVar = [(0, 0, 0), (4, 4, 6), (8, 8, 10), (12, 12, 16), (20, 20, 24)]

    @staticmethod
    def box(ground: tuple[int, int, int], cells: tuple[tuple[int, int], ...]) -> np.ndarray:
        """A dark preview box with ``cells`` painted in our own hint fill."""
        assert HINT_PAINT is not None
        background = np.asarray(ground, dtype=np.float64)
        paint = np.asarray(HINT_PAINT.color, dtype=np.float64)
        fill = background + HINT_PAINT.opacity * (paint - background)
        image = np.full((96, 96, 3), ground, dtype=np.uint8)
        for r, c in cells:
            top, left = 8 + r * 19 + 1, 12 + c * 19 + 1
            image[top : top + 17, left : left + 17] = np.round(fill).astype(np.uint8)
        return image

    @pytest.mark.parametrize("ground", GROUNDS)
    def test_the_composite_is_a_solid_class_here_not_a_band(self, ground) -> None:  # type: ignore[no-untyped-def]
        # The diagnosis: the same fill that scores 0.321 over white scores
        # 0.357-0.374 over these grounds, so the band pass — the only
        # place the refusal used to live — is never even asked.
        image = self.box(ground, ROTATIONS["O"][0].cells)
        scores, _band, _level = band_split(image)
        assert float(scores.max()) >= MIN_SPREAD

    @pytest.mark.parametrize("ground", GROUNDS)
    @pytest.mark.parametrize("piece", PIECES)
    def test_our_own_hint_is_refused_on_a_dark_box_too(self, piece: str, ground) -> None:  # type: ignore[no-untyped-def]
        for rot in ROTATIONS[piece]:
            assert identify_next(self.box(ground, rot.cells)) is None

    def test_it_is_the_paint_rule_and_not_the_shape_rules(self) -> None:
        # Pinned the same way the light box pins it: the shape rules find
        # a perfectly good piece, and it is the color arithmetic that
        # refuses it.
        image = self.box((0, 0, 0), ROTATIONS["T"][0].cells)
        assert identify_next(image, own_paint=None) == "T"
        assert identify_next(image) is None

    @pytest.mark.parametrize("ground", GROUNDS)
    def test_a_real_piece_on_the_same_dark_box_is_still_named(self, ground) -> None:  # type: ignore[no-untyped-def]
        # The control: the guard must refuse our fill, not dark boxes. A
        # game piece drawn on the same ground reads exactly as before.
        image = self.box(ground, ())
        for r, c in ROTATIONS["S"][0].cells:
            top, left = 8 + r * 19 + 1, 12 + c * 19 + 1
            image[top : top + 17, left : left + 17] = (215, 15, 55)
        assert identify_next(image) == "S"
