"""Unit tests for the parallel colour-first tracker's own logic.

Which component is in flight, which cells are settled board, and which
transitions that adds up to — all on synthetic frames from
``tests/colour_frames``. The same tracker replayed over real captured
sessions is ``tests/test_colour_tracker_sessions.py``.
"""

from __future__ import annotations

import numpy as np

from tetris_coach.vision.colour_tracker import ColourTracker, Event

from .colour_frames import (
    BLUE_I,
    COLS,
    GREEN_O,
    PALE_T,
    ROWS,
    WHITE,
    blend,
    preview,
    render,
    tracker,
)

# -- naming a piece ---------------------------------------------------


def test_two_visible_cells_are_named_by_colour_alone() -> None:
    """The whole point: a piece entering from above is named at once.

    Two adjacent cells fit O, S, Z, J and L, so shape must hold; colour
    does not have to.
    """
    track = tracker()
    track.update(render({}), preview([(0, 0), (0, 1), (1, 0), (1, 1)], GREEN_O))
    report = track.update(
        render({(0, 4): GREEN_O, (0, 5): GREEN_O}),
        preview([(0, 0), (0, 1), (1, 0), (1, 1)], GREEN_O),
    )
    assert report.next_piece == "O"
    assert report.falling is not None
    assert report.falling.piece == "O"
    assert report.falling.cells == frozenset({(0, 4), (0, 5)})
    assert not report.falling.complete
    assert report.stack_rows == (0,) * ROWS


def test_a_complete_sighting_names_a_colour_with_no_preview_at_all() -> None:
    """Shape is the fallback for a game whose NEXT box is unreadable."""
    track = tracker()
    piece = {(4, 3): PALE_T, (5, 2): PALE_T, (5, 3): PALE_T, (5, 4): PALE_T}
    assert track.update(render(piece)).falling is not None
    report = track.update(render(piece))
    assert report.falling is not None and report.falling.piece == "T"
    # ... and the colour is now known, so a clipped sighting is named too.
    clipped = track.update(render({(0, 6): PALE_T}))
    assert clipped.falling is not None and clipped.falling.piece == "T"


def test_an_unknown_colour_is_still_tracked_unnamed() -> None:
    track = tracker()
    report = track.update(render({(0, 4): (10.0, 90.0, 200.0), (0, 5): (10.0, 90.0, 200.0)}))
    assert report.falling is not None
    assert report.falling.piece is None
    assert report.falling.cells == frozenset({(0, 4), (0, 5)})


# -- what is not board content ---------------------------------------


def test_a_translucent_ghost_is_not_content() -> None:
    """A landing preview is the piece's own colour at low opacity.

    NOTE: no committed fixture contains a game-drawn ghost — every "ghost"
    in those windows turned out to be this tool's own hint paint — so this
    is the only place the rule is exercised.
    """
    ghost = blend(GREEN_O, 0.3)
    frame = {(1, 4): GREEN_O, (1, 5): GREEN_O, (11, 4): ghost, (11, 5): ghost}
    track = tracker()
    track.update(render({(1, 4): GREEN_O, (1, 5): GREEN_O}))  # learn the opaque colour
    report = track.update(render(frame))
    assert report.falling is not None
    assert report.falling.cells == frozenset({(1, 4), (1, 5)})
    assert report.stack_rows == (0,) * ROWS


def test_cells_the_capture_cannot_see_are_not_content() -> None:
    """A UI panel floating over the board corner shows the NEXT piece."""
    hidden = frozenset({(0, 8), (0, 9), (1, 8), (1, 9)})
    image = render({cell: PALE_T for cell in hidden} | {(11, 0): BLUE_I})
    report = ColourTracker(rows=ROWS, cols=COLS, unobservable_cells=hidden).update(image)
    assert report.falling is None
    assert report.stack_rows[0] == 0 and report.stack_rows[1] == 0
    assert report.stack_rows[11] == 0b1


# -- falling versus settled -------------------------------------------


def test_a_hovering_piece_never_becomes_stack() -> None:
    """This game has no gravity: a piece sits at the top until dragged.

    Time alone must not settle it — nothing holds it up. This is the
    absorbed-piece failure: once the shipped tracker commits a piece in
    flight into the stack, the piece's own descent reads as stack cells
    vanishing, and it cannot get out.
    """
    track = tracker()
    frame = render({(0, 3): PALE_T, (1, 2): PALE_T, (1, 3): PALE_T, (1, 4): PALE_T})
    for _ in range(50):
        report = track.update(frame)
    assert report.falling is not None and report.falling.piece == "T"
    assert report.falling.floating
    assert report.stack_rows == (0,) * ROWS


def test_a_resting_piece_settles_and_reports_a_lock() -> None:
    track = tracker(settle_frames=3)
    falling = {(9, 4): GREEN_O, (9, 5): GREEN_O, (8, 4): GREEN_O, (8, 5): GREEN_O}
    for _ in range(4):
        track.update(render(falling))
    landed = {(10, 4): GREEN_O, (10, 5): GREEN_O, (11, 4): GREEN_O, (11, 5): GREEN_O}
    first = track.update(render(landed))
    assert first.falling is not None and first.falling.piece == "O"
    assert not first.falling.floating
    for _ in range(3):
        report = track.update(render(landed))
    assert report.falling is None
    assert Event.PIECE_LOCKED in report.events
    assert report.stack_rows[11] == 0b110000


def test_a_hard_drop_between_captures_still_reports_one_lock() -> None:
    """A piece that jumps the whole board in one capture locks once.

    Not on the jump itself: this game is drag-to-drop, so a piece on the
    floor may still be slid sideways, and nothing about the jump says it is
    down for good. The lock is reported when the evidence arrives — it holds
    still, or the next piece is dealt.
    """
    track = tracker()
    for _ in range(3):
        track.update(render({(0, 4): GREEN_O, (0, 5): GREEN_O}))
    dropped = render({(10, 0): GREEN_O, (10, 1): GREEN_O, (11, 0): GREEN_O, (11, 1): GREEN_O})
    events = [event for _ in range(6) for event in track.update(dropped).events]
    assert events.count(Event.PIECE_LOCKED) == 1
    assert track.update(dropped).stack_rows[11] == 0b11


def test_the_next_piece_appearing_locks_the_one_that_dropped() -> None:
    track = tracker()
    for _ in range(3):
        track.update(render({(0, 4): GREEN_O, (0, 5): GREEN_O}))
    report = track.update(
        render(
            {
                (10, 0): GREEN_O,
                (10, 1): GREEN_O,
                (11, 0): GREEN_O,
                (11, 1): GREEN_O,
                (0, 4): BLUE_I,
                (0, 5): BLUE_I,
                (0, 6): BLUE_I,
                (0, 7): BLUE_I,
            }
        )
    )
    assert Event.PIECE_LOCKED in report.events
    assert Event.PIECE_SPAWNED in report.events
    assert report.falling is not None and report.falling.piece == "I"
    assert report.stack_rows[11] == 0b11


def test_stack_hanging_over_its_own_hole_is_not_a_falling_piece() -> None:
    """A clear leaves settled cells above empty ones. Sideways support counts."""
    board = {(11, c): BLUE_I for c in range(9)}
    board[(10, 9)] = GREEN_O  # overhang: (11, 9) is a hole under it
    board[(10, 8)] = GREEN_O
    track = tracker()
    for _ in range(5):
        report = track.update(render(board))
    assert report.falling is None
    assert report.stack_rows[10] == 0b1100000000


def test_a_spawn_is_reported_once_per_piece() -> None:
    track = tracker()
    frames = [render({(0, 4): GREEN_O, (0, 5): GREEN_O})] * 3
    frames += [render({(1, 4): GREEN_O, (1, 5): GREEN_O})] * 3
    events = [event for frame in frames for event in track.update(frame).events]
    assert events.count(Event.PIECE_SPAWNED) == 1


def test_a_line_clear_is_reported() -> None:
    track = tracker()
    full = {(11, c): BLUE_I for c in range(COLS)}
    for _ in range(5):
        track.update(render(full))
    report = track.update(render({}))
    assert Event.LINES_CLEARED in report.events
    assert report.cleared_rows == 1
    assert report.stack_rows == (0,) * ROWS


# -- the property the whole design is for -----------------------------


def test_one_unreadable_frame_costs_exactly_one_frame() -> None:
    """No commit, no memory to wedge: the next frame is read from scratch.

    The shipped tracker answers a frame it cannot explain by holding its
    committed state, and four of them by resyncing onto the observed board
    — which is how a piece in flight gets absorbed and the tracker sticks.
    Here a garbage frame produces a garbage answer, and the frame after it
    is right again.
    """
    track = tracker()
    good = render({(2, 3): PALE_T, (3, 2): PALE_T, (3, 3): PALE_T, (3, 4): PALE_T})
    for _ in range(5):
        before = track.update(good)
    noise = np.random.default_rng(0).integers(0, 255, good.shape, dtype=np.uint8)
    track.update(noise)
    after = track.update(good)
    assert before.falling is not None and after.falling is not None
    assert after.falling.piece == before.falling.piece == "T"
    assert after.falling.cells == before.falling.cells
    assert after.stack_rows == before.stack_rows


def test_the_background_memory_survives_a_board_that_fills_up() -> None:
    """Past half full, a per-frame median would call a piece the background."""
    track = tracker()
    track.update(render({}))
    board: dict[tuple[int, int], tuple[float, ...]] = {}
    for r in range(4, ROWS):
        for c in range(COLS):
            board[(r, c)] = BLUE_I
    for _ in range(5):
        report = track.update(render(board))
    assert track.palette.background is not None
    assert np.allclose(track.palette.background, np.array(WHITE), atol=2.0)
    assert report.stack_rows[11] == (1 << COLS) - 1
    assert report.stack_rows[0] == 0


def test_the_preview_teaches_the_palette_a_colour_it_has_not_seen() -> None:
    track = tracker()
    track.update(render({}), preview([(0, 1), (1, 0), (1, 1), (1, 2)], blend(PALE_T, 0.87)))
    report = track.update(render({(0, 7): PALE_T}))
    assert report.falling is not None and report.falling.piece == "T"


# -- a theme where colour says nothing --------------------------------


def test_a_monochrome_theme_falls_back_to_shape() -> None:
    """Game-agnosticism: one colour for every piece is still trackable.

    Colour is the identity when the game gives it; when it does not, the
    tracker is back where the shipped one always is, naming a complete
    sighting by its shape. What survives the loss of colour is everything
    structural: the piece is still the floating component, the settled
    board is still re-derived every frame, and a piece resting on the
    stack still settles into it.
    """
    grey = (120.0, 120.0, 120.0)
    track = tracker()
    stack = {(11, c): grey for c in range(6)}
    flying = {(7, 8): grey, (8, 8): grey, (8, 9): grey, (9, 9): grey}  # an S, upright
    for _ in range(4):
        report = track.update(render(stack | flying))
    assert len(track.palette.classes) == 1
    assert report.falling is not None
    assert report.falling.piece == "S"
    assert report.falling.cells == frozenset(flying)
    assert report.stack_rows[11] == 0b111111

    landed = {(9, 8): grey, (10, 8): grey, (10, 9): grey, (11, 9): grey}
    for _ in range(5):
        report = track.update(render(stack | landed))
    assert report.falling is None
    assert report.stack_rows[11] == 0b1000111111


def test_a_monochrome_theme_keeps_falling_back_after_the_first_piece() -> None:
    """The second piece of a one-colour game, and the third.

    Showing ONE piece cannot tell a fallback from a bootstrap: a tracker
    that consults shape only while the colour is anonymous passes that test
    and then names every later piece after the first. So this shows three
    pieces down one colour class, and the class names none of them.
    """
    grey = (120.0, 120.0, 120.0)
    track = tracker()
    shapes = {
        "S": {(7, 8), (8, 8), (8, 9), (9, 9)},
        "L": {(4, 2), (5, 2), (6, 2), (6, 3)},
        "J": {(4, 6), (5, 6), (6, 5), (6, 6)},
    }
    for expected, cells in shapes.items():
        for _ in range(4):
            report = track.update(render(dict.fromkeys(cells, grey)))
        assert len(track.palette.classes) == 1, "still one colour, as the theme intends"
        assert report.falling is not None
        assert report.falling.piece == expected, f"{expected} named {report.falling.piece}"

    # The class has been retired from naming, so it no longer lends its
    # name to a partial sighting either. Silence is the honest answer for a
    # game whose colours carry no identity: two cells of grey are an O, an
    # S, a Z, a J or an L, and the tracker knows only that.
    assert track.palette.classes[0].ambiguous
    assert track.palette.piece_of(0) is None
    report = track.update(render({(0, 4): grey, (0, 5): grey}))
    assert report.falling is not None
    assert report.falling.cells == frozenset({(0, 4), (0, 5)})
    assert report.falling.piece is None


def test_a_colour_that_is_contradicted_once_never_names_again() -> None:
    """Two tetrominoes rendered alike: neither gets the other's name.

    Not only the monochrome case -- a game that draws J and L in the same
    colour, or recolours a piece by level, lands here too. The palette is
    allowed to be contradicted exactly once, by the strongest evidence
    there is (all four cells at once), and the answer to a contradiction is
    to stop naming rather than to pick a winner.
    """
    shared = (90.0, 180.0, 60.0)
    track = tracker()
    for _ in range(4):
        report = track.update(render(dict.fromkeys({(4, 2), (5, 2), (6, 2), (6, 3)}, shared)))
    assert report.falling is not None and report.falling.piece == "L"
    assert track.palette.piece_of(0) == "L"

    # The NEXT box, which is weaker evidence, may not rename a class...
    track.palette.name(0, "J")
    assert track.palette.piece_of(0) == "L"

    # ...but a complete sighting on the board retires it.
    for _ in range(4):
        report = track.update(render(dict.fromkeys({(4, 6), (5, 6), (6, 5), (6, 6)}, shared)))
    assert report.falling is not None and report.falling.piece == "J"
    assert track.palette.piece_of(0) is None
    track.palette.name(0, "L")
    assert track.palette.piece_of(0) is None, "a retired class cannot be revived by the box"
