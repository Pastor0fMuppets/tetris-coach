"""How often the NEXT box can be read, over every committed window.

One measurement, in one place, because preview readability is not a
property of any single fixture: every window that carries preview crops
contributes, and a change to the reader shows up here as a number before
it shows up anywhere as a behavior.

What each window is for lives in its own module; this one only asks what
the box said, capture by capture, and what the reader got right and wrong
over the lot of them. The reading is run-length encoded, so a flip that
moves by a frame is as visible as a piece that changes name.

Measured before the intermediate band reached the preview path, and
after (``identify_next``, every committed ``next_*.png``):

    window               crops   named before   named after
    absorbed_piece          81         81            81
    ghost_beside_stack      48         48            48
    ghost_session           95         95            95
    live_session            96         96            96
    pale_piece              61          0            47
    pale_preview            15          0            13
    spawn_latency          161        132           150
    ------------------------------------------------------
    total                  557        452           530

The 27 still unnamed are the two things that really are unreadable: a
game panel over the whole screen (pale_piece 00687-00700, the
end-of-round summary; pale_preview 00273/00280, a deal animation mid
swap) and the end-of-round wipe that blanks the box along with the field
(spawn_latency 00121-00125 and 00152-00157). None of them holds a piece
a reader could name.

No crop's name CHANGED: every piece named before is named the same now,
and the gain is 78 crops that used to say nothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from tetris_coach.vision.pieces_vision import identify_next

FIXTURES = Path(__file__).parent / "fixtures"

# (piece, consecutive captures) over each window's crops in frame order.
# A window's preview is a series of runs by construction — a piece sits in
# the box for a whole tenure — so this both counts the reading and shows
# where every deal lands.
READING: dict[str, list[tuple[str | None, int]]] = {
    "absorbed_piece": [("I", 66), ("O", 15)],
    "ghost_beside_stack": [("O", 48)],
    "ghost_session": [("I", 92), ("O", 3)],
    "live_session": [("O", 19), ("I", 51), ("O", 26)],
    # Was [(None, 61)]: a pale periwinkle T, in the box for the window.
    "pale_piece": [("T", 47), (None, 14)],
    # Was [(None, 15)]: the same T, in the window committed for it.
    "pale_preview": [(None, 2), ("T", 13)],
    # Was [..., (None, 18), ("O", 25)]: the T that was dealt, unread for
    # its whole tenure, which is what dated that deal's flip a deal late.
    "spawn_latency": [
        ("O", 41),
        (None, 5),
        ("O", 26),
        (None, 6),
        ("I", 40),
        ("T", 18),
        ("O", 25),
    ],
}


def crops(window: str) -> list[np.ndarray]:
    """A window's preview crops in frame order, as the pipeline hands them."""
    return [
        np.asarray(Image.open(path))[:, :, ::-1]
        for path in sorted((FIXTURES / window).glob("next_*.png"))
    ]


def read(window: str) -> list[str | None]:
    return [identify_next(crop) for crop in crops(window)]


def encode(values: list[str | None]) -> list[tuple[str | None, int]]:
    runs: list[tuple[str | None, int]] = []
    for value in values:
        if runs and runs[-1][0] == value:
            runs[-1] = (value, runs[-1][1] + 1)
        else:
            runs.append((value, 1))
    return runs


def test_every_window_with_preview_crops_is_measured_here() -> None:
    # The measurement covers the committed windows rather than a chosen
    # few: a new window with preview crops has to be added to the table.
    with_crops = {
        path.name for path in FIXTURES.iterdir() if path.is_dir() and any(path.glob("next_*.png"))
    }
    assert with_crops == set(READING)


def test_the_reading_of_every_window() -> None:
    assert {window: encode(read(window)) for window in READING} == READING


def test_the_box_is_read_on_530_of_557_committed_crops() -> None:
    # The headline number, and the one the entering-piece accelerator
    # depends on: it names a spawn from the piece that just LEFT the box,
    # which needs the box readable on the captures either side of a deal.
    named = sum(count for runs in READING.values() for piece, count in runs if piece is not None)
    total = sum(count for runs in READING.values() for _piece, count in runs)
    assert (named, total) == (530, 557)  # was 452 of 557


def test_nothing_that_was_readable_reads_differently() -> None:
    # The no-regression half, measured rather than asserted: the four
    # windows whose boxes were fully readable before are untouched, name
    # for name and frame for frame.
    for window in ("absorbed_piece", "ghost_beside_stack", "ghost_session", "live_session"):
        assert None not in read(window)
        assert encode(read(window)) == READING[window]
