"""Write the derived truth out as JSON, so a race need not re-derive it.

``python -m tetris_coach.truth.build`` regenerates
``tests/fixtures/oracle_truth.json`` from the committed capture windows.
The file is checked in: deriving it takes a minute and reads 542 PNGs, and
a race between two trackers should be comparing them against the same
frozen answer sheet rather than against whatever the oracle happens to say
today. ``tests/test_truth_oracle.py`` fails if the two drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tetris_coach.truth.oracle import FrameTruth, WindowTruth, derive_window
from tetris_coach.truth.windows import CONSECUTIVE, WindowSpec, load_window

SCHEMA = 1
DEFAULT_FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures"
DEFAULT_OUTPUT = DEFAULT_FIXTURES / "oracle_truth.json"


def frame_to_json(frame: FrameTruth) -> dict[str, Any]:
    out: dict[str, Any] = {
        "frame": frame.frame,
        "verdict": frame.verdict,
        "board_visible": frame.board_visible,
        "settled": [list(cell) for cell in frame.settled_cells],
        "ghost": [list(cell) for cell in frame.ghost_cells],
        "own_paint": [list(cell) for cell in frame.own_paint_cells],
    }
    if frame.reasons:
        out["reasons"] = list(frame.reasons)
    if frame.falling_piece is not None:
        out["falling"] = {
            "piece": frame.falling_piece,
            "basis": frame.falling_basis,
            "clipped": frame.falling_clipped,
            "colour": frame.colour_of_falling,
            "cells": [list(cell) for cell in frame.falling_cells],
            "from_occupancy_alone": frame.colour_free_cells,
        }
    if frame.ghost_cells or frame.ghost_obscured:
        out["ghost_complete"] = frame.ghost_complete
        out["ghost_obscured"] = frame.ghost_obscured
    if frame.badge_cells:
        out["badge"] = [list(cell) for cell in frame.badge_cells]
    if frame.unreadable_cells:
        out["unreadable"] = [list(cell) for cell in frame.unreadable_cells]
    return out


def window_to_json(spec: WindowSpec, truth: WindowTruth) -> dict[str, Any]:
    return {
        "window": truth.window,
        "rows": truth.geometry.rows,
        "cols": truth.geometry.cols,
        "board_rect": list(spec.board_rect),
        "next_rect": list(spec.next_rect),
        "unobservable": [list(cell) for cell in sorted(truth.geometry.unobservable)],
        "palette": {
            "background": list(truth.palette.background),
            "content": [list(colour) for colour in truth.palette.content],
        },
        "colour_names": dict(truth.colour_names),
        "notes": list(truth.notes),
        "counts": {
            verdict: truth.count(verdict) for verdict in ("confident", "partial", "abstain")
        },
        "frames": [frame_to_json(frame) for frame in truth.frames],
    }


def build(fixtures: Path = DEFAULT_FIXTURES) -> dict[str, Any]:
    windows = []
    for spec in CONSECUTIVE:
        names, images = load_window(fixtures, spec)
        truth = derive_window(spec.name, names, images, spec.geometry())
        windows.append(window_to_json(spec, truth))
    return {
        "schema": SCHEMA,
        "generated_by": "tetris_coach.truth.build",
        "about": (
            "Per-frame ground truth for the consecutive capture windows, derived "
            "from the pixels by tetris_coach.truth.oracle and independent of any "
            "tracker. Cells are [row, col]. 'settled' is the resting stack, "
            "'falling' the piece in play (cells are what is VISIBLE of it; "
            "'clipped' says more of it lies outside the capture), 'ghost' the "
            "game's landing-preview outline, 'own_paint' the coach's own hint "
            "overlay caught in the capture. A frame with verdict 'abstain' is one "
            "the oracle will not be quoted on; 'partial' means the piece is "
            "answered but some cells are hidden."
        ),
        "windows": windows,
    }


def dumps(payload: dict[str, Any]) -> str:
    """Pretty at the top, one line per frame.

    Fully indented, every ``[row, col]`` pair costs four lines and the file
    runs to half a megabyte of mostly brackets; fully compact, it is one
    unreadable line and every regeneration is a one-line diff. A frame to a
    line is the useful middle: a person can read a window's history down
    the page, and a change to one frame shows up as a change to one line.
    """
    lines = ["{"]
    head = {key: value for key, value in payload.items() if key != "windows"}
    for key, value in head.items():
        lines.append(f" {json.dumps(key)}: {json.dumps(value)},")
    lines.append(' "windows": [')
    for position, window in enumerate(payload["windows"]):
        lines.append("  {")
        for key, value in window.items():
            if key == "frames":
                continue
            lines.append(f"   {json.dumps(key)}: {json.dumps(value)},")
        lines.append('   "frames": [')
        frames = window["frames"]
        for index, frame in enumerate(frames):
            comma = "," if index < len(frames) - 1 else ""
            lines.append("    " + json.dumps(frame, separators=(",", ":")) + comma)
        lines.append("   ]")
        lines.append("  }" + ("," if position < len(payload["windows"]) - 1 else ""))
    lines.append(" ]")
    lines.append("}")
    return "\n".join(lines) + "\n"


def main(output: Path = DEFAULT_OUTPUT, fixtures: Path = DEFAULT_FIXTURES) -> None:
    payload = build(fixtures)
    output.write_text(dumps(payload))
    total = {"confident": 0, "partial": 0, "abstain": 0}
    for window in payload["windows"]:
        for key, value in window["counts"].items():
            total[key] += int(value)
        print(
            f"{window['window']:20s} {len(window['frames']):4d} frames  "
            + "  ".join(f"{k}={v}" for k, v in window["counts"].items())
        )
    print(f"{'total':<21}{sum(total.values()):4d} frames  " + str(total))
    print("written:", output)


if __name__ == "__main__":  # pragma: no cover
    main()
