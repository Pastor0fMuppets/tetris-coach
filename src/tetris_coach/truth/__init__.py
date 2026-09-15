"""Ground-truth oracle over the committed capture windows.

This package is a REFEREE, not a tracker. Nothing in the shipped pipeline
imports it and it imports nothing from :mod:`tetris_coach.vision`: its job
is to say what is really on the board in each fixture frame so two
trackers can be raced against the same answer sheet.

It is pure ``numpy`` and has no GUI, no capture and no macOS dependency,
so it runs anywhere the tests do.
"""

from tetris_coach.truth.oracle import (
    FrameTruth,
    Geometry,
    Palette,
    WindowTruth,
    derive_window,
    measure_frame,
)

__all__ = [
    "FrameTruth",
    "Geometry",
    "Palette",
    "WindowTruth",
    "derive_window",
    "measure_frame",
]
