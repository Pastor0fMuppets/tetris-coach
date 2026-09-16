"""What the committed corpus can and cannot decide about the colour race.

The head-to-head in ``tests/test_race.py`` is run over eight windows of one
session of one game. This file pins the shape of that evidence, so that a
number from the race is never read as saying more than it can.

THE MIRROR PAIRS ARE IN IT NOW, THINLY. For six windows and nineteen
flights every one was an I, an O or a T: no S, Z, J or L falling piece
anywhere in the repository, which left the two MIRROR PAIRS -- the only
case where the two representations genuinely differ -- argued from
tetromino geometry and never once observed. ``hint_stutter`` and
``stray_after_clear`` were committed for a bug that gap was hiding and
close it on the way past: 24 flights now, one each of J, L, S and Z, and
a derived colour name for all seven tetrominoes.

One flight each is not a sample and the arithmetic below says so rather
than rounding it up. What it settles is that the mirror pairs are no
longer UNOBSERVED -- a claim about them can now be wrong about a real
frame.

What is demonstrated here instead is the mechanism, synthetically and
labelled as such: from a partial sighting shape cannot separate J from L
or S from Z, and a distinct colour can. That is an argument about
tetrominoes, which is sound wherever the colours are in fact distinct. It
is not evidence that this game's colours generalise to another game's, and
nothing in this repository is.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
from PIL import Image

from tetris_coach.core.pieces import ROTATIONS
from tetris_coach.race.measures import episodes, load_truth
from tetris_coach.vision.colour_palette import piece_from_cells
from tetris_coach.vision.colour_preview import identify_preview

from .colour_frames import BLUE_I, GREEN_O, render, tracker

FIXTURES = Path(__file__).parent / "fixtures"
WINDOWS = (
    "spawn_latency",
    "live_session",
    "ghost_session",
    "absorbed_piece",
    "pale_piece",
    "ghost_beside_stack",
    "hint_stutter",
    "stray_after_clear",
)
MIRRORS = (("J", "L"), ("S", "Z"))


def test_the_corpus_holds_all_seven_and_four_of_them_once() -> None:
    """Every flight the oracle saw, and how lopsided the letters are.

    The shape of the count is the point, not the coverage. Eleven of the
    24 flights are an I and four letters have exactly one flight each, so
    "all seven are represented" and "all seven are tested" are different
    sentences and only the first is true.
    """
    truth = load_truth()
    flights = collections.Counter(
        episode.piece for window in WINDOWS for episode in episodes(truth[window].frames)
    )
    assert dict(sorted(flights.items())) == {
        "I": 11,
        "J": 1,
        "L": 1,
        "O": 6,
        "S": 1,
        "T": 3,
        "Z": 1,
    }
    assert sum(flights.values()) == 24
    assert {piece for piece, n in flights.items() if n == 1} == {"J", "L", "S", "Z"}

    # And the colour->name map each window derives, which is the thing a
    # colour-first tracker is being credited with learning. Five rendered
    # colours were claimed for this game; all seven names now have one.
    payload = json.loads((FIXTURES / "oracle_truth.json").read_text())
    colours = {
        piece
        for window in payload["windows"]
        if window["window"] in WINDOWS
        for piece in window["colour_names"].values()
    }
    assert colours == {"I", "J", "L", "O", "S", "T", "Z"}


def test_what_every_next_box_in_the_repository_shows() -> None:
    """The other way a colour is learned, counted over every crop there is.

    Worth checking separately: the NEXT box names a colour without the
    board ever showing that piece, so a letter that appears only here is
    still evidence. Three do, all of them from the two windows committed
    for the hint stutter -- J 5 times, Z 9 and L 23 -- against the I, O and
    T the other six had between them. S is the one letter no preview crop
    in this repository has ever shown; the corpus' only S is a falling
    piece, on ``stray_after_clear`` 00545.
    """
    seen: collections.Counter[str] = collections.Counter()
    boxes = 0
    for directory in sorted(FIXTURES.iterdir()):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("next_*.png")):
            boxes += 1
            image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)[:, :, ::-1]
            reading = identify_preview(image)
            if reading is not None:
                seen[reading.piece] += 1
    assert boxes > 500, "this is over every NEXT crop in the repo"
    assert set(seen) == {"I", "J", "L", "O", "T", "Z"}
    assert {piece: seen[piece] for piece in ("J", "L", "Z")} == {"J": 5, "L": 23, "Z": 9}
    assert "S" not in seen


def test_shape_cannot_separate_a_mirror_pair_from_a_partial_sighting() -> None:
    """The mechanism the design rests on, over the pieces the corpus lacks.

    A piece entering from above shows its top row or two before the rest
    clears the ceiling. This enumerates every such partial sighting of every
    rotation of all seven pieces and asks how many pieces each one fits.

    J and L, and S and Z, are mirror images: a partial sighting of one is a
    partial sighting of the other, so no shape rule can name them before
    they have descended. This is the whole of the colour-first argument.
    The corpus now holds one flight of each of the four, which makes it
    checkable against a real frame; the enumeration below is what makes it
    GENERAL, over every rotation of all seven.
    """
    partial: dict[tuple[tuple[int, int], ...], set[str]] = collections.defaultdict(set)
    for piece, rotations in ROTATIONS.items():
        for rotation in rotations:
            cells = sorted(rotation.cells)
            height = max(r for r, _ in cells) + 1
            for visible in range(1, height):  # the piece is still clipped
                top = [(r, c) for r, c in cells if r < visible]
                min_c = min(c for _, c in top)
                partial[tuple(sorted((r, c - min_c) for r, c in top))].add(piece)

    ambiguous = {key: names for key, names in partial.items() if len(names) > 1}
    assert ambiguous, "partial sightings are ambiguous, which is the premise"
    for left, right in MIRRORS:
        shared = [names for names in ambiguous.values() if {left, right} <= names]
        assert shared, f"no partial sighting is shared by {left} and {right}"

    # ...while a COMPLETE sighting of either is unambiguous, which is why
    # the shipped tracker works at all once a piece has descended, and why
    # a complete sighting is allowed to overrule the palette.
    for piece, rotations in ROTATIONS.items():
        for rotation in rotations:
            assert piece_from_cells(frozenset(rotation.cells)) == piece


def test_two_colours_name_a_partial_sighting_that_two_shapes_cannot() -> None:
    """The same two-cell sighting, named or not named, by colour alone.

    Synthetic, and it stays so: ``hint_stutter`` holds a captured J and
    ``stray_after_clear`` a captured Z, but no committed frame shows a
    mirror pair's two halves rendered in two colours, which is the thing
    this asserts. What it shows is only that the tracker's naming does what
    it claims WHEN the colours differ -- which is the hypothesis, not
    evidence for it.
    """
    # Two cells side by side fit O, S, Z, J and L. Shape can say nothing.
    entering = frozenset({(0, 4), (0, 5)})
    assert piece_from_cells(entering) is None

    track = tracker()
    track.update(render({}))
    for colour, name in ((BLUE_I, "J"), (GREEN_O, "L")):
        index = track.palette.intern(np.array(colour) - track.palette.background)
        track.palette.name(index, name)
    for colour, name in ((BLUE_I, "J"), (GREEN_O, "L")):
        report = track.update(render(dict.fromkeys(entering, colour)))
        assert report.falling is not None
        assert report.falling.cells == entering
        assert report.falling.piece == name

    # And when the game renders the pair alike, the tracker says nothing
    # rather than guessing -- the honest answer, and the same one a
    # shape-first tracker gives.
    same = tracker()
    same.update(render({}))
    index = same.palette.intern(np.array(BLUE_I) - same.palette.background)
    same.palette.witness(index, "J")
    same.palette.witness(index, "L")
    report = same.update(render(dict.fromkeys(entering, BLUE_I)))
    assert report.falling is not None
    assert report.falling.piece is None


def test_what_generalising_would_take_is_written_down() -> None:
    """A reading list, asserted so it cannot quietly go missing.

    Everything the race cannot decide is recorded in one place, because the
    tempting misreading of "97% vs 79%" is that the question is settled.

    1. A window from a SECOND game. Every frame here is one session of ROAS
       Stacker: one light theme, flat colours, one background, one cell
       geometry, one hint overlay.
    2. MORE THAN ONE window containing each mirror pair. There is now one
       flight each of J, L, S and Z, from two windows of one session, so
       the mirror pairs -- the only case where the two representations
       genuinely disagree -- are observed rather than sampled. One flight
       cannot show a rule holds across rotations, speeds or stack shapes.
    3. A theme that is not flat light colour: a dark theme, a gradient or
       textured cell, a piece recoloured by level. The flatness premise is
       now checked per frame (``board_readable``) but it has only ever been
       measured against frames that satisfy it.
    4. A game that FILLS its landing preview. This one outlines it, so no
       rule in this repo has ever been tested against a ghost that a
       centre-patch sampler can see.
    """
    doc = test_what_generalising_would_take_is_written_down.__doc__
    assert doc is not None
    for needed in ("SECOND game", "each mirror pair", "not flat light colour", "FILLS"):
        assert needed in doc
