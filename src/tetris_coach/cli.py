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

from .core.board import HEIGHT, WIDTH, Board
from .core.pieces import PIECES
from .solver.search import Move, best_move


def _render_demo_board(board: Board, move: Move | None) -> str:
    hint_cells = set(move.cells) if move is not None else set()
    lines = []
    for r in range(HEIGHT):
        row = board.rows[r]
        cells = []
        for c in range(WIDTH):
            if (r, c) in hint_cells:
                cells.append("[]")
            elif row >> c & 1:
                cells.append("##")
            else:
                cells.append(" .")
        lines.append("|" + "".join(cells) + "|")
    lines.append("+" + "-" * (2 * WIDTH) + "+")
    return "\n".join(lines)


def run_demo(pieces: int, seed: int, delay: float) -> int:
    """Solver self-play in the terminal; returns a process exit code."""
    rng = random.Random(seed)
    board = Board()
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
                f"piece {placed + 1}/{pieces}: {current} "
                f"(next {upcoming})  lines: {lines_cleared}"
            )
            time.sleep(delay)
        board = move.board
        lines_cleared += move.lines_cleared
        placed += 1
        current = upcoming
        upcoming = rng.choice(PIECES)

    elapsed = time.perf_counter() - started
    search_times.sort()
    p50 = search_times[len(search_times) // 2]
    p95 = search_times[int(len(search_times) * 0.95)]
    print(_render_demo_board(board, None))
    print(
        f"Placed {placed} pieces, cleared {lines_cleared} lines "
        f"in {elapsed:.1f}s (seed {seed})."
    )
    print(f"2-ply search time: p50 {p50 * 1000:.1f} ms, p95 {p95 * 1000:.1f} ms.")
    return 0


def _run_overlay(args: argparse.Namespace) -> int:  # pragma: no cover - macOS only
    try:
        from .app import CoachConfig, run
        from .region_select import select_region
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to start GUI mode: {exc}", file=sys.stderr)
        return 1
    try:
        board_rect = select_region("Drag a rectangle over the Tetris board (Esc to cancel)")
        if board_rect is None:
            print("Cancelled.", file=sys.stderr)
            return 1
        next_rect = select_region("Drag a rectangle over the next-piece box (Esc to skip)")
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        print("On non-macOS hosts, try: python -m tetris_coach.cli --demo", file=sys.stderr)
        return 1
    config = CoachConfig(
        poll_rate=args.poll_rate,
        hint_color=args.hint_color,
        debug=args.debug,
    )
    run(board_rect, next_rect, config=config)
    return 0


def main(argv: list[str] | None = None) -> int:
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
        "--pieces", type=int, default=200, help="demo: number of pieces to play"
    )
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
        "--hint-color", default="#00e5ff", help="overlay: hint color (Qt color string)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="overlay: show a debug window with what vision sees",
    )
    args = parser.parse_args(argv)

    if args.demo:
        return run_demo(pieces=args.pieces, seed=args.seed, delay=args.delay)
    return _run_overlay(args)


if __name__ == "__main__":
    raise SystemExit(main())
