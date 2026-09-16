"""What a person saw, pinned against what the oracle says.

``tetris_coach.truth.oracle`` derives ground truth for the committed
capture windows so two trackers can be raced against the same answer
sheet. An answer sheet nobody checked is worth nothing, so the frames in
:class:`TestHandRead` were opened as images, magnified, and read cell by
cell by eye; the literals below are that reading, written down before the
oracle's own output was consulted. Twice the unaided eye was wrong at
full-board scale and right again at cell scale, both times off by one
column, and both times the oracle had it right -- which is the argument
for magnifying rather than for trusting either one.

The rest of the file checks the properties the oracle claims for itself,
over all 542 frames: that its landing-preview reading and its shape
reading never name different pieces, that every preview it reports as
incomplete is incomplete because this tool painted over it, and that the
committed JSON still says what a fresh derivation says.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tetris_coach.truth import oracle
from tetris_coach.truth.build import DEFAULT_OUTPUT, build
from tetris_coach.truth.windows import CONSECUTIVE, WINDOWS, load_window, overlap_mask

FIXTURES = Path(__file__).parent / "fixtures"


def cells(*pairs: tuple[int, int]) -> list[list[int]]:
    """The literal form the JSON uses, so a reading can be typed as pairs."""
    return [[r, c] for r, c in pairs]


@pytest.fixture(scope="module")
def truth() -> dict[str, Any]:
    return json.loads(DEFAULT_OUTPUT.read_text())  # type: ignore[no-any-return]


@pytest.fixture(scope="module")
def frames(truth: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (window["window"], frame["frame"]): frame
        for window in truth["windows"]
        for frame in window["frames"]
    }


class TestHandRead:
    """Frames a person opened and read, cell by cell.

    Chosen to cover every kind of thing these windows do: a piece in clear
    air, a piece clipped by the top edge, a piece clipped by the preview
    box, a piece rotated, a piece under the coach's own hint, the coach's
    rotation badge, the game's row-clear celebration, a piece that slides
    out of sight entirely, and a frame that is not the game at all.
    """

    def test_a_piece_in_clear_air(self, frames: dict[tuple[str, str], Any]) -> None:
        # Blue I lying across row 1; nothing else on the board; the game's
        # own landing outline four rows... eleven rows below it, on the floor.
        frame = frames[("live_session", "00043")]
        assert frame["verdict"] == "confident"
        assert frame["falling"]["piece"] == "I"
        assert frame["falling"]["cells"] == cells((1, 3), (1, 4), (1, 5), (1, 6))
        assert frame["settled"] == []
        assert frame["own_paint"] == []
        assert frame["ghost"] == cells((11, 3), (11, 4), (11, 5), (11, 6))

    def test_a_piece_under_the_coachs_own_paint(self, frames: dict[tuple[str, str], Any]) -> None:
        # The I has hard-dropped onto the exact square the hint was marking,
        # so those four cells are a real piece with cyan paint over them. A
        # rule that deleted painted cells would delete the stack here.
        frame = frames[("live_session", "00059")]
        assert frame["verdict"] == "confident"
        assert frame["settled"] == cells((11, 0), (11, 1), (11, 2), (11, 3))
        assert frame["own_paint"] == cells((11, 0), (11, 1), (11, 2), (11, 3))
        # Two cells of a green O at the top edge: the rest is above the capture.
        assert frame["falling"]["piece"] == "O"
        assert frame["falling"]["cells"] == cells((0, 4), (0, 5))
        assert frame["falling"]["clipped"] is True
        assert frame["ghost"] == cells((10, 4), (10, 5), (11, 4), (11, 5))

    def test_a_stack_of_two_pieces(self, frames: dict[tuple[str, str], Any]) -> None:
        # Green O resting on the blue I, new blue I at the top edge.
        frame = frames[("live_session", "00099")]
        assert frame["verdict"] == "confident"
        assert frame["falling"]["cells"] == cells((0, 3), (0, 4), (0, 5), (0, 6))
        assert frame["falling"]["piece"] == "I"
        assert frame["settled"] == cells(
            (9, 0), (9, 1), (10, 0), (10, 1), (11, 0), (11, 1), (11, 2), (11, 3)
        )
        assert frame["ghost"] == cells((10, 3), (10, 4), (10, 5), (10, 6))

    def test_the_coachs_rotation_badge(self, frames: dict[tuple[str, str], Any]) -> None:
        # A cyan disc with a "1" in it sits on (7, 9), over a vertical I hint
        # down the right wall. The disc is opaque: nothing under it is known.
        frame = frames[("live_session", "00121")]
        assert frame["verdict"] == "partial"
        assert frame["badge"] == cells((7, 9))
        assert frame["unreadable"] == cells((7, 9))
        assert frame["own_paint"] == cells((8, 9), (9, 9), (10, 9), (11, 9))
        assert frame["falling"]["cells"] == cells((0, 4), (0, 5))
        assert frame["settled"] == cells(
            (9, 0), (9, 1),
            (10, 0), (10, 1), (10, 2), (10, 3), (10, 4), (10, 5),
            (11, 0), (11, 1), (11, 2), (11, 3), (11, 4), (11, 5), (11, 6), (11, 7),
        )  # fmt: skip

    def test_the_hint_is_not_the_landing_preview(self, frames: dict[tuple[str, str], Any]) -> None:
        # Both are on this frame and they are in different places: the coach
        # wants the O at cols 2-3, the game previews it at cols 4-5 under the
        # piece. Two fixture READMEs read the first as the second.
        frame = frames[("ghost_session", "00061")]
        assert frame["verdict"] == "confident"
        assert frame["falling"]["piece"] == "O"
        assert frame["falling"]["cells"] == cells((0, 4), (0, 5), (1, 4), (1, 5))
        assert frame["settled"] == []
        assert frame["own_paint"] == cells((10, 2), (10, 3), (11, 2), (11, 3))
        assert frame["ghost"] == cells((10, 4), (10, 5), (11, 4), (11, 5))

    def test_a_preview_landing_on_top_of_the_stack(
        self, frames: dict[tuple[str, str], Any]
    ) -> None:
        frame = frames[("ghost_session", "00150")]
        assert frame["falling"]["cells"] == cells((0, 3), (0, 4), (0, 5), (0, 6))
        assert frame["ghost"] == cells((9, 3), (9, 4), (9, 5), (9, 6))
        assert frame["badge"] == cells((7, 9))
        assert frame["own_paint"] == cells((8, 9), (9, 9), (10, 9), (11, 9))
        assert frame["settled"] == cells(
            (10, 0), (10, 1), (10, 2), (10, 3), (10, 4), (10, 5),
            (11, 0), (11, 1), (11, 2), (11, 3), (11, 4), (11, 5),
        )  # fmt: skip

    def test_a_preview_beside_the_stack(self, frames: dict[tuple[str, str], Any]) -> None:
        # The window named for a ghost the old rule refused. The green O in
        # the corner is stack; the hint is at cols 0-3; the preview is at
        # cols 4-7 under the I, which is where the I will land.
        frame = frames[("ghost_beside_stack", "00147")]
        assert frame["verdict"] == "confident"
        assert frame["falling"]["cells"] == cells((1, 4), (1, 5), (1, 6), (1, 7))
        assert frame["settled"] == cells((11, 8), (11, 9))
        assert frame["own_paint"] == cells((11, 0), (11, 1), (11, 2), (11, 3))
        assert frame["ghost"] == cells((11, 4), (11, 5), (11, 6), (11, 7))

    def test_the_pale_periwinkle_t(self, frames: dict[tuple[str, str], Any]) -> None:
        # The piece the shipped classifier reads as background. Its badge is
        # on (8, 1) and its hint is the T at (9,1),(9,2),(9,3),(10,2).
        frame = frames[("absorbed_piece", "00298")]
        assert frame["falling"]["piece"] == "T"
        assert frame["falling"]["cells"] == cells((0, 4), (1, 3), (1, 4), (1, 5))
        assert frame["badge"] == cells((8, 1))
        assert frame["own_paint"] == cells((9, 1), (9, 2), (9, 3), (10, 2))
        assert frame["ghost"] == cells((10, 4), (11, 3), (11, 4), (11, 5))
        assert frame["settled"] == cells(
            (10, 0), (10, 1), (10, 8), (10, 9), (11, 0), (11, 1), (11, 8), (11, 9)
        )

    def test_a_rotated_piece_with_a_cell_above_the_capture(
        self, frames: dict[tuple[str, str], Any]
    ) -> None:
        # The I has been turned upright and only three of its four cells are
        # on screen. Three in a column fit an I, a J and an L equally; the
        # game's own preview shows all four, down col 3 at rows 6-9.
        frame = frames[("absorbed_piece", "00353")]
        assert frame["falling"]["piece"] == "I"
        assert frame["falling"]["cells"] == cells((0, 3), (1, 3), (2, 3))
        assert frame["falling"]["clipped"] is True
        assert frame["ghost"] == cells((6, 3), (7, 3), (8, 3), (9, 3))
        assert oracle.name_of_shape(frozenset((r, c) for r, c in frame["ghost"])) == "I"

    def test_lone_stack_cells_left_by_line_clears(self, frames: dict[tuple[str, str], Any]) -> None:
        # (11, 5) and (11, 8) are single pale periwinkle cells, all that is
        # left of earlier pieces. They are stack, not noise.
        frame = frames[("pale_piece", "00660")]
        assert frame["verdict"] == "confident"
        assert frame["falling"]["piece"] == "T"
        assert frame["falling"]["cells"] == cells((2, 4), (3, 3), (3, 4), (3, 5))
        assert frame["falling"]["clipped"] is False
        assert frame["settled"] == cells(
            (8, 0), (8, 9), (9, 0), (9, 9), (10, 0), (10, 9), (11, 0), (11, 5), (11, 8), (11, 9)
        )
        assert frame["ghost"] == cells((9, 4), (10, 3), (10, 4), (10, 5))

    def test_a_frame_that_is_not_the_game(self, frames: dict[tuple[str, str], Any]) -> None:
        # Not the end-of-round panel the README calls it: a different
        # application is on screen, an event-schedule page with its own
        # headings and photographs. Nothing here is a board.
        frame = frames[("pale_piece", "00690")]
        assert frame["verdict"] == "abstain"
        assert frame["board_visible"] is False
        assert frame["reasons"] == ["board-not-readable"]
        assert frame["settled"] == []
        assert "falling" not in frame

    def test_a_piece_entering_from_above(self, frames: dict[tuple[str, str], Any]) -> None:
        # Three cells in a row at the top edge: a T's bar, with its stem
        # still above the capture. The preview below shows the whole piece.
        frame = frames[("spawn_latency", "00216")]
        assert frame["verdict"] == "confident"
        assert frame["falling"]["piece"] == "T"
        assert frame["falling"]["cells"] == cells((0, 3), (0, 4), (0, 5))
        assert frame["falling"]["clipped"] is True
        assert frame["ghost"] == cells((9, 4), (10, 3), (10, 4), (10, 5))
        assert frame["own_paint"] == cells((11, 2), (11, 3), (11, 4), (11, 5))
        assert frame["settled"] == cells(
            (9, 8), (9, 9),
            (10, 0), (10, 1), (10, 8), (10, 9),
            (11, 0), (11, 1), (11, 2), (11, 3), (11, 4), (11, 5), (11, 8), (11, 9),
        )  # fmt: skip

    def test_the_hint_painted_over_the_preview(self, frames: dict[tuple[str, str], Any]) -> None:
        # The coach wants the T where the game says it will land, so its
        # paint covers two of the preview's four cells and erases the
        # outline there. What survives is (9, 4) and (10, 5).
        frame = frames[("spawn_latency", "00234")]
        assert frame["falling"]["cells"] == cells((0, 4), (1, 3), (1, 4), (1, 5))
        assert frame["own_paint"] == cells((9, 3), (10, 2), (10, 3), (10, 4))
        assert frame["ghost"] == cells((9, 4), (10, 5))
        assert frame["ghost_complete"] is False
        assert frame["ghost_obscured"] is True

    def test_the_row_clear_celebration(self, frames: dict[tuple[str, str], Any]) -> None:
        # A "ROAS +10% ROW CLEARED!" card over the middle of the board,
        # confetti falling across it, and row 11 complete and about to go.
        frame = frames[("spawn_latency", "00131")]
        assert frame["verdict"] == "abstain"
        assert "line-clear-animation:11" in frame["reasons"]
        assert "falling" not in frame
        assert cells((9, 3))[0] in frame["unreadable"]

    def test_a_piece_that_slides_behind_the_preview_box(
        self, frames: dict[tuple[str, str], Any]
    ) -> None:
        # The green O is at rows 0-1, cols 8-9 -- every cell of it inside the
        # square the NEXT panel floats over. Saying "no falling piece" here
        # would be a lie, so the oracle says nothing.
        frame = frames[("spawn_latency", "00154")]
        assert frame["verdict"] == "abstain"
        assert frame["reasons"] == ["piece-unaccounted-for"]
        assert "falling" not in frame
        assert frame["settled"] == cells((11, 8), (11, 9))


class TestWhatTheOracleClaims:
    """The properties it says hold, checked over every frame."""

    def test_the_preview_and_the_piece_never_name_different_things(
        self, truth: dict[str, Any]
    ) -> None:
        """Two readings of the same piece, one of them never used to name it.

        The piece is named from its own shape. The game's landing preview
        is a separate copy of it, and where the preview is whole it names a
        tetromino too. They agree on every frame in the corpus -- which is
        what makes the preview usable as a check rather than a crutch.
        """
        whole = 0
        for window in truth["windows"]:
            for frame in window["frames"]:
                if not frame.get("falling") or not frame.get("ghost_complete"):
                    continue
                preview = oracle.name_of_shape(frozenset((r, c) for r, c in frame["ghost"]))
                assert preview == frame["falling"]["piece"], (window["window"], frame["frame"])
                whole += 1
        assert whole >= 400

    def test_every_incomplete_preview_is_one_we_painted_over(self, truth: dict[str, Any]) -> None:
        """A missing preview cell is always a cell this tool painted.

        The preview marks where the piece will land and the solver often
        wants it in the same place, so the hint is drawn over the outline
        and erases it. If a preview were ever incomplete for some OTHER
        reason, the oracle's reading of it would be unexplained -- and this
        is where that would show up.
        """
        geometry = oracle.Geometry(rows=12, cols=10)
        for window in truth["windows"]:
            for frame in window["frames"]:
                if not frame.get("falling") or frame.get("ghost_complete"):
                    continue
                seen = frozenset((r, c) for r, c in frame["ghost"])
                painted = frozenset((r, c) for r, c in frame["own_paint"])
                if not seen:
                    assert painted, (window["window"], frame["frame"], "no preview, no paint")
                    continue
                assert oracle._completions(seen, painted, geometry), (
                    window["window"],
                    frame["frame"],
                )

    def test_no_piece_in_the_corpus_is_named_by_a_palette(self, truth: dict[str, Any]) -> None:
        """No frame in the corpus needs colour to put a name to a piece.

        ``basis`` records which channel named the episode: its own four
        cells, the game's own landing preview, or -- last resort -- the
        colour -> name map the other two produced. The last one is never
        reached here, so a race judged against this answer sheet is not
        being judged against a palette.

        ``ghost`` was unreached too until ``stray_after_clear``, whose last
        four frames hold an L that never descends past the top edge inside
        the window. Three cells name no tetromino, so shape abstains and
        the game's own preview of where it will land names it instead --
        which is the oracle using a channel neither tracker has, and
        exactly why the answer sheet is derived rather than read off one.
        """
        bases = {
            frame["falling"]["basis"]
            for window in truth["windows"]
            for frame in window["frames"]
            if frame.get("falling")
        }
        assert bases == {"shape", "ghost"}
        assert "colour" not in bases

    def test_colour_never_names_two_pieces(self, truth: dict[str, Any]) -> None:
        """The colour -> name map is a result, and it is consistent.

        Every window derives it independently, from shape. Nothing forces
        them to agree, and there is no compiled-in palette to make them.
        """
        combined: dict[str, str] = {}
        for window in truth["windows"]:
            assert window["notes"] == [], window["window"]
            for colour, name in window["colour_names"].items():
                assert combined.setdefault(colour, name) == name, colour
        assert combined == {
            "215,46,45": "I",
            "112,239,217": "O",
            "251,224,206": "T",
            "249,174,169": "J",
            "94,220,125": "S",
            "240,157,221": "Z",
            "130,175,243": "L",
        }

    def test_a_confident_frame_has_nothing_to_say_against_itself(
        self, truth: dict[str, Any]
    ) -> None:
        for window in truth["windows"]:
            for frame in window["frames"]:
                if frame["verdict"] == "confident":
                    assert "reasons" not in frame
                else:
                    assert frame["reasons"], (window["window"], frame["frame"])

    def test_unobservable_cells_are_never_reported_as_anything(self, truth: dict[str, Any]) -> None:
        """The capture has no evidence under the preview box, so nor has this."""
        for window in truth["windows"]:
            masked = {(r, c) for r, c in window["unobservable"]}
            for frame in window["frames"]:
                reported = {(r, c) for r, c in frame["settled"]}
                if frame.get("falling"):
                    reported |= {(r, c) for r, c in frame["falling"]["cells"]}
                assert not (reported & masked), (window["window"], frame["frame"])


class TestTheOracleOwesNothingToTheTracker:
    """Independence, as far as a test can enforce it."""

    def test_it_imports_no_vision_module(self) -> None:
        import tetris_coach.truth.build
        import tetris_coach.truth.oracle
        import tetris_coach.truth.windows

        for module in (
            tetris_coach.truth.oracle,
            tetris_coach.truth.windows,
            tetris_coach.truth.build,
        ):
            source = Path(module.__file__ or "").read_text()
            assert "tetris_coach.vision" not in source
            assert "from ..vision" not in source

    def test_the_hint_colour_is_the_one_the_overlay_paints(self) -> None:
        """Recognising our own paint is a fact about us, not about the game.

        The oracle writes the pen colour and the fill opacity out rather
        than importing them, because the module that owns them pulls in Qt.
        Written out, they can drift; this is what stops them.
        """
        from tetris_coach.overlay.renderer import HintStyle
        from tetris_coach.vision.grid import HINT_FILL_OPACITY

        red, green, blue = (int(HintStyle.color[i : i + 2], 16) for i in (1, 3, 5))
        assert oracle.HINT_PEN_BGR == (blue, green, red)
        # The fill is the READER's constant, not the painter's: the overlay
        # draws none by default now (HintStyle.fill_opacity is 0, the hint
        # is an outline in the band no reader samples), but these captures
        # have one in them and both the oracle and the rule that
        # un-composites it have to agree about which one.
        assert HintStyle.fill_opacity == 0.0
        assert oracle.HINT_FILL_OPACITY == HINT_FILL_OPACITY

    def test_the_unobservable_cells_are_the_engines(self) -> None:
        """Derived here from the two rectangles, and it had better match."""
        from tetris_coach.app import compute_overlap_mask
        from tetris_coach.capture.screen import Rect

        for spec in WINDOWS:
            engine = compute_overlap_mask(Rect(*spec.board_rect), Rect(*spec.next_rect), spec.rows)
            assert overlap_mask(spec) == engine == frozenset({(0, 8), (0, 9), (1, 8), (1, 9)})

    def test_the_paint_it_finds_is_the_paint_a_different_reader_finds(
        self, truth: dict[str, Any]
    ) -> None:
        """A second opinion on the coach's own overlay, written years apart.

        ``tests.layers.pen_stroked_cells`` was written for the shipped
        classifier's own tests. It reads the RGB PNG rather than the BGR
        array, at a different channel tolerance, and counts raw pen pixels
        anywhere in a cell rather than distinguishing the hint's outline
        from its rotation badge. Two independent measurements of the same
        769 painted cells, over 542 frames, and they agree exactly.
        """
        from .layers import pen_stroked_cells

        for window in truth["windows"]:
            for frame in window["frames"]:
                mine = {(r, c) for r, c in frame["own_paint"]}
                mine |= {(r, c) for r, c in frame.get("badge", [])}
                path = FIXTURES / window["window"] / f"board_{frame['frame']}.png"
                assert mine == pen_stroked_cells(path, window["rows"]), (
                    window["window"],
                    frame["frame"],
                )

    def test_a_sampled_window_is_not_offered_as_a_film(self) -> None:
        """``pale_preview`` is sampled, and the oracle reads windows as films.

        A piece is named from whichever frame of its episode shows it
        whole, and a resting piece is called settled because the frames
        after it never move it. Neither sentence means anything across
        frames 00273, 00280, 00290, 00300 -- so that window is marked and
        left out rather than quietly derived into nonsense.
        """
        from tetris_coach.truth.windows import by_name

        assert by_name("pale_preview").consecutive is False
        assert "pale_preview" not in {spec.name for spec in CONSECUTIVE}


class TestTheCommittedAnswerSheet:
    def test_it_is_what_deriving_it_again_produces(self, truth: dict[str, Any]) -> None:
        """Re-derive all six windows from the PNGs and compare.

        The JSON is checked in so a race is reproducible without spending a
        minute on 542 images. This is the price of that: if the oracle
        changes and the file is not rebuilt, here is where it is caught.
        Rebuild with ``python -m tetris_coach.truth.build``.
        """
        assert build(FIXTURES) == truth

    def test_it_covers_every_consecutive_window(self, truth: dict[str, Any]) -> None:
        derived = [window["window"] for window in truth["windows"]]
        assert derived == [spec.name for spec in CONSECUTIVE]
        for spec, window in zip(CONSECUTIVE, truth["windows"], strict=True):
            names, _ = load_window(FIXTURES, spec)
            assert [frame["frame"] for frame in window["frames"]] == names
