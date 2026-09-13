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

Current pipeline: background-distance scoring — each cell's central-patch
mean color is scored by its sqrt-compressed Euclidean distance from a
per-frame background estimate (top-row cell-color median for the board,
whole-image pixel median for the preview box), then split by Otsu. The
"before" column is the previous absolute-score pipeline (per-pixel
max(brightness, saturation) via a 256x256 LUT), which assumed a dark
background.

| Stage                                     | before p50 | before p95 | after p50 | after p95 |
| ----------------------------------------- | ---------- | ---------- | --------- | --------- |
| Grid: `classify_grid` at 600x1200         | 4.9 ms     | 5.1 ms     | 2.2 ms    | 2.3 ms    |
| Preview: `identify_next` at 200x200       | 1.8 ms     | 1.9 ms     | 2.0 ms    | 2.0 ms    |
| Preview cache gate (`np.array_equal`)     | 0.011 ms   | 0.011 ms   | 0.006 ms  | 0.006 ms  |

The grid stage got cheaper: a per-channel patch mean plus one distance per
cell is less work than LUT-indexing every sampled pixel. The preview stage
pays a little more for its whole-image median background estimate. Both
stages keep histogram Otsu for pixel-scale inputs and the exact small-N
Otsu for the 200 cell scores, and `CoachEngine` still gates
`identify_next` behind the byte-identical preview cache (~14 of 15 frames
in the steady state).

At the default 15 fps poll rate the vision stages cost ~4 ms per frame
worst case (~2 ms in the cache-hit steady state) against the ~67 ms frame
budget.

For history: the original pipeline scored the whole board image per frame
(35.0/75.4 ms p50/p95 grid, 14.5/15.3 ms preview) before patch-only
sampling and histogram Otsu brought it to the "before" column above.
