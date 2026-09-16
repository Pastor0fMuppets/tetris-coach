"""``tetris-coach`` entry point.

``--demo`` runs a terminal self-play demonstration exercising the whole
non-GUI stack (core + solver): the solver plays a simulated game against a
random piece sequence with 2-ply lookahead, printing the board.

Without ``--demo``, the macOS overlay flow starts: select the board and
next-piece regions, then run the capture -> vision -> solve -> overlay loop.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from typing import TYPE_CHECKING

from .capture.screen import Rect
from .core.board import DEFAULT_HEIGHT, WIDTH, Board
from .core.pieces import PIECES
from .region_select import MIN_BOARD_SIZE, MIN_PREVIEW_SIZE
from .solver.search import Move, best_move
from .vision.grid import DEFAULT_HINT_COLOR, DEFAULT_NEXT_HINT_COLOR
from .vision.pieces_vision import SPAWN_ROWS

if TYPE_CHECKING:  # pragma: no cover - import-time cost, not behaviour
    from .app import CoachConfig


def _rect_error(rect: Rect, min_size: tuple[int, int], what: str) -> str | None:
    """Explain why ``rect`` is unusable for ``what``, or ``None`` if fine."""
    min_w, min_h = min_size
    if rect.width < min_w or rect.height < min_h:
        return (
            f"{what} selection is {rect.width}x{rect.height} logical px; at "
            f"least {min_w}x{min_h} is needed for vision to resolve it. "
            "Run again and drag a rectangle over the full region "
            "(a plain click selects nothing)."
        )
    return None


def _render_demo_board(board: Board, move: Move | None) -> str:
    """Widen ``Board.__str__`` (the one grid renderer) to 2-char cells,
    with the hinted placement overlaid as ``[]``."""
    hint_cells = set(move.cells) if move is not None else set()
    lines = []
    for r, text_row in enumerate(str(board).splitlines()):
        cells = [
            "[]" if (r, c) in hint_cells else "##" if ch == "#" else " ."
            for c, ch in enumerate(text_row)
        ]
        lines.append("|" + "".join(cells) + "|")
    lines.append("+" + "-" * (2 * WIDTH) + "+")
    return "\n".join(lines)


def run_demo(pieces: int, seed: int, delay: float, rows: int = DEFAULT_HEIGHT) -> int:
    """Solver self-play in the terminal; returns a process exit code."""
    rng = random.Random(seed)
    board = Board([0] * rows)
    current = rng.choice(PIECES)
    upcoming = rng.choice(PIECES)
    placed = 0
    lines_cleared = 0
    started = time.perf_counter()
    search_times: list[float] = []

    while placed < pieces:
        t0 = time.perf_counter()
        move = best_move(board, current, next_piece=upcoming)
        search_times.append(time.perf_counter() - t0)
        if move is None:
            print(_render_demo_board(board, None))
            print(f"Topped out after {placed} pieces ({lines_cleared} lines).")
            return 1
        if delay > 0:
            # Show the placement hint on the pre-drop board, like the overlay
            # would, then apply it.
            sys.stdout.write("\x1b[2J\x1b[H")
            print(_render_demo_board(board, move))
            print(
                f"piece {placed + 1}/{pieces}: {current} (next {upcoming})  lines: {lines_cleared}"
            )
            time.sleep(delay)
        board = move.board
        lines_cleared += move.lines_cleared
        placed += 1
        current = upcoming
        upcoming = rng.choice(PIECES)

    elapsed = time.perf_counter() - started
    print(_render_demo_board(board, None))
    print(f"Placed {placed} pieces, cleared {lines_cleared} lines in {elapsed:.1f}s (seed {seed}).")
    if search_times:
        search_times.sort()
        p50 = search_times[len(search_times) // 2]
        p95 = search_times[min(len(search_times) - 1, int(len(search_times) * 0.95))]
        print(f"2-ply search time: p50 {p50 * 1000:.1f} ms, p95 {p95 * 1000:.1f} ms.")
    return 0


def coach_config(args: argparse.Namespace) -> CoachConfig:
    """The engine configuration these arguments ask for.

    Split out of :func:`_run_overlay`, which cannot run off macOS, so that
    what the flags MEAN is testable headlessly -- ``--tracker`` in
    particular, since it decides which tracker reads every frame.
    """
    from .app import CoachConfig

    return CoachConfig(
        poll_rate=args.poll_rate,
        hint_color=args.hint_color,
        next_hint_color=args.next_hint_color,
        show_next_hint=not args.no_next_hint,
        hint_fill_opacity=args.hint_fill,
        rows=args.rows,
        debug=args.debug,
        dump_dir=args.dump_frames,
        dump_limit=args.dump_limit,
        tracker=args.tracker,
    )


def _run_overlay(args: argparse.Namespace) -> int:  # pragma: no cover - macOS only
    try:
        from .app import run
        from .region_select import select_region
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to start GUI mode: {exc}", file=sys.stderr)
        return 1
    try:
        board_rect = select_region(
            "Drag a rectangle over the Tetris board (Esc to cancel)",
            min_size=MIN_BOARD_SIZE,
        )
        if board_rect is None:
            print("Cancelled.", file=sys.stderr)
            return 1
        error = _rect_error(board_rect, MIN_BOARD_SIZE, "Board")
        if error is not None:
            print(error, file=sys.stderr)
            return 1
        next_rect = select_region(
            "Drag a rectangle over the next-piece box (Esc to skip)",
            min_size=MIN_PREVIEW_SIZE,
        )
        if next_rect is not None:
            error = _rect_error(next_rect, MIN_PREVIEW_SIZE, "Next-piece")
            if error is not None:
                print(error, file=sys.stderr)
                return 1
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        print("On non-macOS hosts, try: python -m tetris_coach.cli --demo", file=sys.stderr)
        return 1
    print(f"Board region: {board_rect}", flush=True)
    print(f"Next-piece region: {next_rect}", flush=True)
    run(board_rect, next_rect, config=coach_config(args))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tetris-coach",
        description="Game-agnostic Tetris training overlay (macOS) and solver demo.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="run a terminal self-play demo (no GUI required)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=DEFAULT_HEIGHT,
        help="board height in rows for the demo and the overlay "
        f"(width is always {WIDTH}; default {DEFAULT_HEIGHT})",
    )
    parser.add_argument("--pieces", type=int, default=200, help="demo: number of pieces to play")
    parser.add_argument("--seed", type=int, default=0, help="demo: RNG seed")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="demo: seconds between frames (0 = fast, print only the result)",
    )
    parser.add_argument(
        "--poll-rate", type=float, default=15.0, help="overlay: captures per second"
    )
    parser.add_argument(
        "--hint-color",
        default=DEFAULT_HINT_COLOR,
        help="overlay: colour of the hint for the piece in play, drawn as a "
        "bold solid outline (Qt colour string)",
    )
    parser.add_argument(
        "--next-hint-color",
        default=DEFAULT_NEXT_HINT_COLOR,
        help="overlay: colour of the second hint -- where the NEXT piece goes "
        "if you follow the first one -- drawn as a dashed outline. It is the "
        "dashes that tell the two apart at a glance; the colour is so that "
        "they also differ where each sits over a similar piece. With "
        "--hint-fill it must also be far enough from --hint-color for the "
        "coach to tell its own two marks apart, or the run is refused",
    )
    parser.add_argument(
        "--no-next-hint",
        action="store_true",
        help="overlay: show one target only. The second hint is where the "
        "piece after this one goes on the board the first placement leaves "
        "behind; some players would rather have the one square to aim at",
    )
    parser.add_argument(
        "--hint-fill",
        type=float,
        default=0.0,
        metavar="OPACITY",
        help="overlay: fill the current hint's cells at this opacity (0-1, "
        "default 0 = outline only). The outline is drawn in the outer band "
        "of each cell, which is the part of a cell vision never samples, so "
        "by default the coach cannot read its own drawing at all. A fill is "
        "inside that sample, and a correct reading then depends on the paint "
        "being recognized and removed again -- the path three of this tool's "
        "own bugs came down",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="overlay: print a terminal debug view of what vision sees "
        "(the board, the falling and next piece, the transitions) on every "
        "frame whose reading changed, plus a throttled one-line status",
    )
    parser.add_argument(
        "--tracker",
        choices=("colour", "shape"),
        default="colour",
        help="overlay: which tracker reads the frames. colour (the default) "
        "names each piece by the colour it is drawn in and re-derives the "
        "board from every frame; shape matches occupancy against a committed "
        "stack memory. Measured on this project's six capture windows the "
        "colour reader drew no hint for the wrong piece where shape drew six, "
        "and was never late where shape was up to 3 frames behind; shape is "
        "kept as an escape hatch for a theme the colour rules cannot read",
    )
    parser.add_argument(
        "--dump-frames",
        metavar="DIR",
        default=None,
        help="overlay: save captured board/next frames as PNGs into DIR "
        "(debugging aid for region/scaling/tracking problems)",
    )
    parser.add_argument(
        "--dump-limit",
        type=int,
        default=700,
        help="overlay: how many consecutive frames --dump-frames saves "
        "before thinning to every 100th (default 700, ~47s at 15 fps)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.rows < SPAWN_ROWS + 1:
        print(
            f"--rows must be at least {SPAWN_ROWS + 1} (got {args.rows}): "
            f"pieces spawn within the top {SPAWN_ROWS} rows, so a shorter "
            "board has no room to play.",
            file=sys.stderr,
        )
        return 2

    if not 0.0 <= args.hint_fill <= 1.0:
        print(
            f"--hint-fill must be between 0 and 1 (got {args.hint_fill}): it is "
            "the opacity the hint's fill is drawn at, and the reader is told "
            "the same number so that what it un-composites is what was painted.",
            file=sys.stderr,
        )
        return 2

    # The two hint colours are the reader's business as well as the eye's:
    # with a fill configured, a second hint in (nearly) the first's colour
    # is read back as a fill that was never painted, and un-compositing it
    # invents a tetromino out of bare board. Refused rather than warned
    # about, because the cost is a phantom piece and the fix is one flag.
    from .app import hint_color_conflict

    conflict = hint_color_conflict(coach_config(args))
    if conflict is not None:
        print(conflict, file=sys.stderr)
        return 2

    if args.demo:
        return run_demo(pieces=args.pieces, seed=args.seed, delay=args.delay, rows=args.rows)
    return _run_overlay(args)


if __name__ == "__main__":
    raise SystemExit(main())
