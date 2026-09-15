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
    CELL,
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


def test_a_piece_landing_beside_settled_cells_of_its_own_colour() -> None:
    """The absorbed-piece failure, arriving by the colour road.

    Colour separates a piece from the stack -- except from stack of its own
    colour, which is one 4-connected component with it and cannot be told
    apart by the one signal this design is built on. This game deals three
    I pieces in a row in two of the committed windows, so it is not exotic.

    Before the fix the arriving I was reported as NO PIECE and swallowed
    into the stack with no event raised at all: the coach would have gone
    on hinting for the piece before it.

    What separates them is time, not colour. The arriving cells changed in
    the last few frames and the ones they touch did not.
    """
    track = tracker()
    settled = {(11, c): BLUE_I for c in range(4)}
    for _ in range(5):
        before = track.update(render(settled))
    assert before.falling is None and before.stack_rows[11] == 0b1111

    arrived = settled | {(11, c): BLUE_I for c in range(4, 8)}
    report = track.update(render(arrived))
    assert report.falling is not None
    assert report.falling.piece == "I"
    assert report.falling.cells == frozenset({(11, 4), (11, 5), (11, 6), (11, 7)})
    assert report.stack_rows[11] == 0b1111, "the stack is what was there before it"
    assert Event.PIECE_SPAWNED in report.events

    # It settles when it has held still, exactly as any resting piece does.
    for _ in range(3):
        report = track.update(render(arrived))
    assert report.falling is None
    assert report.stack_rows[11] == 0b11111111
    assert Event.PIECE_LOCKED in report.events


def test_a_piece_coming_to_rest_on_its_own_colour_keeps_its_settling_grace() -> None:
    """It used to vanish on the frame it landed, ahead of its own grace."""
    track = tracker()
    settled = {(11, c): BLUE_I for c in range(4)}
    for _ in range(5):
        track.update(render(settled))
    for row in (7, 8, 9):
        report = track.update(render(settled | {(row, c): BLUE_I for c in range(4)}))
        assert report.falling is not None and report.falling.floating

    landed = settled | {(10, c): BLUE_I for c in range(4)}
    for _ in range(3):
        report = track.update(render(landed))
        assert report.falling is not None, "still the piece, resting"
        assert report.falling.piece == "I"
        assert not report.falling.floating
        assert report.stack_rows[10] == 0
    report = track.update(render(landed))
    assert report.falling is None
    assert report.stack_rows[10] == 0b1111
    assert Event.PIECE_LOCKED in report.events


# -- what is not board content ---------------------------------------


def test_a_filled_translucent_ghost_would_be_read_as_its_piece() -> None:
    """The rule that used to catch this is gone, and this is what that costs.

    A magnitude floor relative to a colour class's brightest sighting did
    catch a filled preview -- and it also erased any piece the game drew
    dimmer than that sighting (see the test below), which is the worse
    failure by far: a misnamed piece is a bad hint, a vanished piece is a
    hint for a board that does not exist.

    So the cost is recorded rather than hidden. A game that fills its
    landing preview would have four phantom cells in the stack here. No
    frame in this repo does -- see
    ``test_colour_tracker_sessions.test_the_game_draws_a_ghost_and_it_is_an_outline``
    for what this game actually draws -- so there is nothing to tune such a
    rule against, and the structural fact a real rule would need (a preview
    sits below its piece, in its piece's columns) is not measurable here.
    """
    ghost = blend(GREEN_O, 0.3)
    frame = {(1, 4): GREEN_O, (1, 5): GREEN_O, (11, 4): ghost, (11, 5): ghost}
    track = tracker()
    track.update(render({(1, 4): GREEN_O, (1, 5): GREEN_O}))  # learn the opaque colour
    report = track.update(render(frame))
    assert report.falling is not None
    assert report.falling.cells == frozenset({(1, 4), (1, 5)})
    assert report.stack_rows[11] == 0b110000, "the filled preview is read as stack"


def test_a_piece_dimmer_than_its_own_brightest_sighting_is_still_on_the_board() -> None:
    """Two shades of one hue: the dimmer one must not disappear.

    Perpendicular distance to a ray makes two shades of one colour the SAME
    class by construction -- same direction, different magnitude -- so a
    rule that called anything below half a class's peak a ghost erased the
    dimmer piece outright. Not misnamed: absent. It was neither the falling
    piece nor part of the stack, so the solver was handed a board without
    it and drew a hint for a game that was not being played.

    A theme with two shades of one colour, or a piece recoloured by level,
    is not exotic, and the margin on real data was not wide either: the
    pale periwinkle T's measured peak is 52.9, which puts its erasure floor
    at 26.5 against an EMPTY_DIST of 12.
    """
    track = tracker()
    cells = {(4, 3), (5, 2), (5, 3), (5, 4)}
    for _ in range(3):
        bright = track.update(render(dict.fromkeys(cells, PALE_T)))
    assert bright.falling is not None and bright.falling.piece == "T"
    peak = track.palette.classes[0].peak

    for _ in range(3):
        dim = track.update(render(dict.fromkeys(cells, blend(PALE_T, 0.45))))
    assert dim.falling is not None, "the dim T is on the board"
    assert dim.falling.piece == "T"
    assert dim.falling.cells == frozenset(cells)
    assert len(track.palette.classes) == 1, "and it is the same colour, at 45% of the peak"
    assert track.palette.classes[0].peak == peak


def test_a_frame_that_is_not_a_board_is_refused_whole() -> None:
    """Nothing is read from it, and -- more importantly -- nothing written.

    A design whose claim is that a bad frame costs one frame has to mean
    it, and a palette class that only ever grows is the one place it did
    not: reading a non-board frame interned junk colours permanently.
    """
    track = tracker()
    board = {(11, c): BLUE_I for c in range(4)} | {(3, 6): GREEN_O, (3, 7): GREEN_O}
    for _ in range(4):
        good = track.update(render(board))
    classes = len(track.palette.classes)
    background = track.palette.background
    assert good.board_visible and good.falling is not None

    noise = np.random.default_rng(0).integers(0, 255, (ROWS * 20, COLS * 20, 3), dtype=np.uint8)
    blind = track.update(noise)
    assert not blind.board_visible
    assert blind.falling is None, "no piece is invented out of a frame that is not a board"
    assert blind.events == ()
    assert len(track.palette.classes) == classes, "not one junk colour interned"
    assert track.palette.background is background

    # ...and the frame the board comes back reads as itself, not as a lock
    # or a line clear caused by the gap.
    back = track.update(render(board))
    assert back.board_visible
    assert back.stack_rows == good.stack_rows
    assert Event.LINES_CLEARED not in back.events


def test_a_blind_frame_holds_the_stack_so_the_board_returning_is_not_a_clear() -> None:
    """The count carried across the gap is what stops a phantom event.

    ``_events`` reads a line clear off a drop in the number of settled
    cells. If a refused frame reported an empty board, the frame after it
    would look like several rows vanishing at once.
    """
    track = tracker()
    board = {(11, c): BLUE_I for c in range(10)} | {(10, c): BLUE_I for c in range(8)}
    for _ in range(4):
        before = track.update(render(board))
    blank = np.random.default_rng(1).integers(0, 255, (ROWS * 20, COLS * 20, 3), dtype=np.uint8)
    for _ in range(3):
        blind = track.update(blank)
    assert not blind.board_visible
    assert blind.stack_rows == before.stack_rows, "the last real stack is what it still holds"
    after = track.update(render(board))
    assert after.events == () and after.cleared_rows == 0


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
    """The row is completed by a piece landing in it, as it is in a game.

    The frames where the completed row is still on screen are refused (a
    clear is playing over them, see the test below), so the clear is
    reported on the frame the board comes back -- which is the frame the
    coach has something to say about again. The arithmetic still works
    because the tracker allows for the four cells the piece that completed
    the row was carrying as the piece in flight.
    """
    track = tracker()
    nearly = {(11, c): BLUE_I for c in range(6)}
    piece = {(4, c): GREEN_O for c in range(6, 10)}
    for _ in range(4):
        track.update(render(nearly | piece))
    landed = {(11, c): GREEN_O for c in range(6, 10)}
    flash = track.update(render(nearly | landed))
    assert not flash.board_visible, "a completed row on screen is a clear playing"
    report = track.update(render({}))
    assert Event.LINES_CLEARED in report.events
    assert report.cleared_rows == 1
    assert report.stack_rows == (0,) * ROWS


def test_a_completed_row_still_on_screen_is_refused() -> None:
    """The flash a clear plays over the row is not a board, and reads as one.

    ROAS Stacker recolours a completed row and then flashes it. Every cell
    that changes colour counts as having JUST changed, and a cell that has
    just changed is the evidence this tracker names a piece from -- so a
    couple of cells of a row that is being taken away outranked the real
    piece and became "the piece" (measured on spawn_latency 00127-00133:
    an ``I`` drawn over four of the seven frames while the player held an
    O). A finished row is never a resting state of a Tetris board, so
    seeing one is enough to know the frame is mid-animation.
    """
    track = tracker()
    board = {(11, c): BLUE_I for c in range(COLS)}
    report = track.update(render(board))
    assert not report.board_visible
    assert report.refused_because == "a completed row is still on screen"
    assert report.falling is None

    # One cell short of complete is an ordinary board and is read.
    del board[(11, 4)]
    assert track.update(render(board)).board_visible


def test_a_row_the_panel_covers_can_never_read_as_complete() -> None:
    """A row with a hidden cell is not known to be full: the panel could hide a gap.

    The row here is drawn full, but two of its cells are behind the game's
    own UI, so the frame is read as an ordinary board rather than refused
    as a clear. Erring this way costs a frame of a real clear; erring the
    other way would refuse every frame of a game whose panel sits over a
    column that fills up.
    """
    track = tracker(unobservable_cells=frozenset({(11, 8), (11, 9)}))
    report = track.update(render({(11, c): BLUE_I for c in range(COLS)}))
    assert report.board_visible
    assert report.stack_rows[11] == 0b11111111


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
    # Every row one cell short of complete: a completed row would mean a
    # clear is playing and the frame would be refused (see
    # ``test_a_completed_row_still_on_screen_is_refused``). 72 of 120
    # cells is still well past the half the median would trip over.
    for r in range(4, ROWS):
        for c in range(COLS - 1):
            board[(r, c)] = BLUE_I
    for _ in range(5):
        report = track.update(render(board))
    assert track.palette.background is not None
    assert np.allclose(track.palette.background, np.array(WHITE), atol=2.0)
    assert report.stack_rows[11] == (1 << (COLS - 1)) - 1
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


def card(offset: float) -> tuple[float, ...]:
    """A flat popup card drawn ``offset`` uint8 units off the board's ground."""
    return (WHITE[0] - offset, WHITE[1], WHITE[2])


def test_a_card_near_the_board_colour_erases_nothing() -> None:
    """The cover neither the flatness nor the airborne premise can see.

    ROAS Stacker's "ROW CLEARED" card sits 7 uint8 units off the board
    background (``truth/oracle.py``), and the content floor is 12. So a
    cell the card covers is not content resting on nothing -- it is
    NOTHING, and both of the other premises are satisfied by a board that
    has apparently just gone empty. Measured before this premise existed:
    the card over a 16-cell stack gave ``accepted=True`` and a stack of
    zero, and because the frame was ACCEPTED the engine's stale counter
    was reset on every one of them, so the 45-frame withdrawal that exists
    for this very popup was never reached.

    What gives it away is that the empty board reads AS the background
    rather than near it: over the whole committed corpus not one
    unpainted observable cell sits even 1.0 away.
    """
    track = tracker()
    board = {(11, c): BLUE_I for c in range(8)} | {(10, c): GREEN_O for c in range(4)}
    for _ in range(3):
        before = track.update(render(board))
    assert before.stack_rows[11] == 0b11111111

    covered = board | {(r, c): card(7.0) for r in (9, 10, 11) for c in range(COLS)}
    report = track.update(render(covered))
    assert not report.board_visible
    assert report.refused_because == "cells are neither the board's ground nor content"
    assert report.stack_rows == before.stack_rows, "the last real board is what it holds"


def test_a_card_just_above_the_content_floor_is_refused_too() -> None:
    """The same card one unit brighter fills instead of erasing.

    Above :data:`EMPTY_DIST` every cell it covers is content, grounded and
    plausible: measured, 14 phantom stack cells and a hint three rows
    higher up the board. Nothing about the colour can refuse that -- what
    refuses it is that the card spans the board's full width, which reads
    as a completed row, and a completed row is never a resting state.

    A cover narrower than the board AND clear of the background AND
    touching the stack is the case that still gets through; it is named in
    ``ColourTracker._blind`` rather than left to be found live.
    """
    track = tracker()
    board = {(11, c): BLUE_I for c in range(8)} | {(10, c): GREEN_O for c in range(4)}
    for _ in range(3):
        before = track.update(render(board))
    covered = board | {(r, c): card(12.1) for r in (9, 10, 11) for c in range(COLS)}
    report = track.update(render(covered))
    assert not report.board_visible
    assert report.refused_because == "a completed row is still on screen"
    assert report.stack_rows == before.stack_rows


def test_a_ghost_outline_does_not_look_like_a_cover() -> None:
    """The band premise must not fire on the game's own landing preview.

    The preview is an outline: the centre of the cell is pure background,
    so the sampled patch barely moves and the four cells it marks stay
    well under the band. Even if one did land in it, a preview is a
    tetromino and the budget is a tetromino.
    """
    track = tracker()
    piece = {(4, c): GREEN_O for c in range(2)} | {(5, c): GREEN_O for c in range(2)}
    ghost = {(r, c): blend(GREEN_O, 0.04) for r in (10, 11) for c in range(2)}
    report = track.update(render(piece | ghost))
    assert report.board_visible
    assert report.falling is not None and report.falling.piece == "O"


def test_a_flat_panel_floating_over_the_board_is_not_a_board() -> None:
    """The cover the flatness premise cannot see, and the one that matters live.

    ROAS Stacker draws a "ROW CLEARED" card over its own playfield after a
    clear, and a leaderboard at game over. A web page fails the flatness
    test because a page is photographs and text; a card drawn in one flat
    colour passes it by construction, and used to arrive as a slab of
    phantom stack -- with the solver then planning around rows that are a
    popup.

    What refuses it is Tetris rather than rendering: on a board, one piece
    is in flight and everything else is held up by what is under it, so
    more than a tetromino resting on nothing is not a board. Measured, the
    cost of that premise over the whole committed corpus is zero (see
    ``tests/test_colour_tracker_sessions.py``).
    """
    track = tracker()
    board = {(11, c): BLUE_I for c in range(4)}
    for _ in range(4):
        before = track.update(render(board))
    classes = len(track.palette.classes)
    background = track.palette.background
    assert before.board_visible and before.stack_rows[11] == 0b1111

    popup = board | {(r, c): GREEN_O for r in range(3, 7) for c in range(1, 9)}
    covered = track.update(render(popup))
    assert not covered.board_visible, "32 cells resting on nothing is not a board"
    assert covered.falling is None
    assert covered.stack_rows == before.stack_rows, "the last real board is what it holds"
    assert covered.events == ()
    assert len(track.palette.classes) == classes, "the card's colour is not a piece colour"
    assert track.palette.background is not None
    assert background is not None
    assert np.array_equal(track.palette.background, background)

    back = track.update(render(board))
    assert back.board_visible and back.stack_rows == before.stack_rows
    assert Event.LINES_CLEARED not in back.events


def test_the_budget_for_what_rests_on_nothing_is_a_piece_and_a_bit() -> None:
    """A slab over the void is not a board; a stray cell beside a piece is.

    Both directions are the point. A whole piece hovering with an empty
    board under it is the ordinary frame of this game -- no gravity, so it
    hangs there until the player drags it -- and refusing THAT would be a
    coach that goes blind every time a piece spawns. One component larger
    than a tetromino cannot be a piece, and what it is in practice is a
    cover.

    The TOTAL is deliberately looser than the per-component budget. At
    exactly one tetromino it also refused a real piece with one stray cell
    beside it, and a stray cell or two is what a board rectangle a few
    pixels off produces: measured on the 8 committed live ROAS Stacker
    frames, a selection that leaves the NEXT panel covering two unmasked
    cells reads 6 airborne on every frame, and refused every frame of the
    session in silence. Accuracy there should degrade, not stop.
    """
    piece = {(4, c): BLUE_I for c in range(4)}
    hovering = tracker().update(render(piece))
    assert hovering.board_visible
    assert hovering.falling is not None and hovering.falling.piece == "I"

    # A stray cell beside the piece: read, and the piece is still named.
    stray = tracker().update(render(piece | {(7, 7): GREEN_O}))
    assert stray.board_visible
    assert stray.falling is not None and stray.falling.piece == "I"

    # Five cells of one colour resting on nothing are not a tetromino.
    slab = tracker().update(render({(4, c): BLUE_I for c in range(5)}))
    assert not slab.board_visible
    assert slab.refused_because == "more than a tetromino is resting on nothing"

    # Nor are nine cells over the void, however they are broken up.
    scattered = {(2, 0), (2, 1), (4, 4), (4, 5), (6, 8), (6, 9), (8, 2), (8, 3), (8, 4)}
    many = tracker().update(render(dict.fromkeys(scattered, GREEN_O)))
    assert not many.board_visible


def test_a_cover_is_a_gap_in_the_motion_evidence_too() -> None:
    """What the board did behind a cover was not seen, so it is not evidence.

    A cell's age says how long it has held its colour, and the tracker uses
    it to tell an arriving piece from the stack it landed on. Across frames
    the tracker REFUSED, that age is a fiction: nothing was seen, so
    everything that changed behind the cover looks like it changed just
    now.

    Measured before the fix, on a flat card held over the board for twenty
    frames while a column grew underneath it: the frame the board came back
    reported those three settled cells as the piece in flight, raised a
    PIECE_SPAWNED, and handed the solver a board they were missing from.
    The real piece, resting on the stack, was not reported at all.
    """
    track = tracker()
    before = {(11, c): BLUE_I for c in range(8)} | {(10, c): BLUE_I for c in range(3)}
    for _ in range(6):
        track.update(render(before))

    card = before | {(r, c): GREEN_O for r in range(2, 6) for c in range(1, 9)}
    for _ in range(20):
        assert not track.update(render(card)).board_visible

    grown = {(9, 0): BLUE_I, (8, 0): BLUE_I, (7, 0): BLUE_I}
    resting = {(10, 5): PALE_T, (10, 6): PALE_T, (10, 7): PALE_T, (9, 6): PALE_T}
    back = track.update(render(before | grown | resting))
    assert back.board_visible
    assert back.falling is None, "nothing here was seen to move"
    assert back.events == ()
    for cell in grown | resting:
        assert back.stack_rows[cell[0]] >> cell[1] & 1, f"{cell} is on the board"

    # A piece the capture can see is in flight needs no history at all, so
    # the frame after the cover still reports one that floats.
    flying = track.update(render(before | grown | {(4, c): GREEN_O for c in range(4, 8)}))
    assert flying.falling is not None and flying.falling.piece == "I"


def test_a_refused_frame_does_not_move_the_next_piece_either() -> None:
    """Nothing is written has to include the box, or the palette lies.

    The box is a different region of the screen, so a cover over the board
    need not cover it — but reading it teaches the palette a COLOUR as well
    as a name, and a frame the gate refuses has its palette rolled back. A
    name kept off such a frame would point at a class that no longer
    exists, so the box is read after the gate rather than before it.
    """
    track = tracker()
    board = {(11, c): BLUE_I for c in range(4)}
    track.update(render(board), preview([(0, 1), (1, 0), (1, 1), (1, 2)], blend(PALE_T, 0.87)))
    assert track.update(render(board)).next_piece == "T"
    classes = len(track.palette.classes)

    card = board | {(r, c): GREEN_O for r in range(3, 7) for c in range(1, 9)}
    covered = track.update(
        render(card), preview([(0, 0), (0, 1), (0, 2), (0, 3)], blend(BLUE_I, 0.87))
    )
    assert not covered.board_visible
    assert covered.next_piece == "T", "the box reading came with a frame that is not a board"
    assert len(track.palette.classes) == classes


def test_the_next_box_is_read_with_this_session_own_paint() -> None:
    """--hint-color has to reach the box, not just the board.

    The NEXT panel floats over the top corner of the playfield in the game
    this is used on, so a hint drawn in that corner is drawn over the BOX.
    A reader told the wrong paint colour does not recognize the fill, and
    what it then sees in the box is our own tetromino-shaped paint.
    """
    from tetris_coach.vision.grid import HINT_FILL_OPACITY, HINT_PAINT, OwnPaint

    magenta = OwnPaint.for_hint_color("#ff00ff")
    assert magenta is not None
    box = preview([(0, 0), (0, 1), (1, 0), (1, 1)], GREEN_O).astype(np.float64)
    fill = np.array(magenta.color, dtype=np.float64)
    under = box[: 3 * CELL, :]
    box[: 3 * CELL, :] = under + HINT_FILL_OPACITY * (fill - under)
    painted = box.round().astype(np.uint8)

    told = ColourTracker(rows=ROWS, cols=COLS, paint=magenta)
    assert told.update(render({}), painted).next_piece == "O"
    # The default paint is a different colour, so the fill is not
    # recognized and the box is unreadable rather than misread.
    for paint in (HINT_PAINT, None):
        deaf = ColourTracker(rows=ROWS, cols=COLS, paint=paint)
        assert deaf.update(render({}), painted).next_piece is None
