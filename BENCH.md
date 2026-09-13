# Search-time benchmark

Full 2-ply search (`solver.search.best_move` with a known next piece): every
(rotation, column) placement of the current piece, each scored by the best
next-piece placement on the resulting board. Pure CPython on bitboards, no
pruning. Budget: **< 50 ms** (enforced by
`tests/test_search.py::TestBenchmark::test_two_ply_search_under_50ms`).

## Apple M2 (macOS 14.5, 8 GB, CPython 3.12.14)

300 runs per scenario, wall-clock (`time.perf_counter`):

| Scenario                                   | p50     | p95     | max      |
| ------------------------------------------ | ------- | ------- | -------- |
| Empty board, T then T (worst-case ~34x34)  | 6.4 ms  | 6.5 ms  | 6.6 ms   |
| Mid-game board, T then T                   | 7.5 ms  | 7.7 ms  | 11.0 ms  |
| Mid-game board, I then I                   | 1.9 ms  | 2.0 ms  | 2.1 ms   |
| Self-play, 500 moves, random pieces        | 3.3 ms  | 7.0 ms  | 7.5 ms   |

pytest-benchmark (same machine, T x T on the mid-game board): mean 7.5 ms,
median 7.5 ms over 134 rounds.

Headroom vs. the 50 ms budget is ~6x at p95, so the precompute-ahead loop in
`app.py` (solve during the fall of the previous piece) comfortably hides the
search latency entirely.

## CI hardware

Not yet measured on CI. Re-run and append results with:

```sh
python -m pytest tests/test_search.py::TestBenchmark -q  # asserts the budget
python -m tetris_coach.cli --demo --pieces 500           # prints p50/p95
```

# Vision stage

Per-frame vision costs on synthetic frames: a 600x1200 board image
(Retina-scale capture of a 10x20 grid, classic-dark style with gridlines and
per-pixel noise) and a 200x200 next-piece preview box. Apple M2 (macOS 14.5,
8 GB, CPython 3.12.14), 100 runs each, wall-clock, before/after measured
back-to-back on the same machine with `python -m tests.bench_vision`
(reported in BENCH.md rather than asserted: wall-clock tests are flaky; the
suite's enforced benchmark covers the solver).

| Stage                                     | before p50 | before p95 | after p50 | after p95 |
| ----------------------------------------- | ---------- | ---------- | --------- | --------- |
| Grid: `classify_grid` at 600x1200         | 35.0 ms    | 75.4 ms    | 4.9 ms    | 5.1 ms    |
| Preview: `identify_next` at 200x200       | 14.5 ms    | 15.3 ms    | 1.8 ms    | 1.9 ms    |
| Preview cache gate (`np.array_equal`)     | —          | —          | 0.011 ms  | 0.011 ms  |

What changed — with identical classification across a 613-case synthetic
sweep (all styles x cell sizes x boards/pieces: occupancy, confidence, and
identified pieces byte-equal before vs. after):

- `cell_scores` ran `score_map` over the entire board image (full-size
  float32 copies per frame — the allocation churn behind the before column's
  wild p95), then discarded the ~75% of pixels outside each cell's sampled
  patch. It now scores only the 200 sampled sub-rects, reduces channel
  max/min on the raw uint8 planes, and reads per-pixel scores from a
  precomputed 256x256 (channel max, channel min) table.
- `identify_next` fed every preview pixel to the exact per-sample Otsu loop
  documented for small value sets. Pixel-scale inputs now use histogram
  Otsu (`np.histogram` bins=256 + vectorized cumulative sums), same
  threshold within one bin width; the 200 cell scores in `classify_grid`
  keep the exact small-N path.
- The preview image is byte-identical on ~14 of 15 frames, so `CoachEngine`
  caches the last preview frame and its identified piece and gates
  `identify_next` on `np.array_equal`, ~0.01 ms in the steady state.

At the default 15 fps poll rate the vision stages drop from ~50 ms per
frame (a whole frame budget on Retina regions) to ~5 ms in the steady
state.
