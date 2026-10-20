# Dimensionality migration: `(batch, *dim, time, *seq)` → `(batch, *time, *dim)`

Working plan for moving nnodely 2 from its old tensor layout to a
channels-last one, in which the sequence (rollout) axes are time axes.
Each step ends with a check that must pass before the next one starts.

**Status (2026-10-20):** the migration is functionally complete. See
"Summary and next steps" at the end.
- Steps 1-5 are done: the convention, every layer, the data side and every
  test are on `(batch, *time, *dim)`.
- Verified: TF full suite 824 passed, 1 skipped; JAX 760 + 35 slow passed;
  torch 763 + 27 slow passed (skips are backend-only ONNX/XLA tests). ruff
  clean, `pyright src tests` 0 errors, `sphinx-build -W` passes, every guide
  snippet runs, and the README example produces the Fibonacci series.
- Step 7 (docs): the guide pages, README and CHANGELOG are done. An agent was
  still updating `NNODELY_AI_GUIDE.md` when this was written; check
  `git diff NNODELY_AI_GUIDE.md` and run its recipes.
- Decisions 1 (`Seq*` → `Time*`), 2 (Fir features) and 3 (semantic `axis`)
  are still open.

## Old layout

Every stream was a tensor `(batch, *dim, time, *seq)`:

- `dim`: one or more feature axes;
- `time`: the one axis of the sample window;
- `seq`: one rollout axis per Loop, appended in the order the Loops were
  applied, so the last one was the outermost Loop.

An operand with fewer axes got its missing axes at the end.

## New convention (step 2)

Every stream is a tensor `(batch, *time, *dim)`:

- `time` (T\*): the time axes `(T_k, ..., T_1)`, **outermost first**.
  - The **inner time** `T_1` is the sample window: temporal operations (a
    window, Fir, Derivative, Integrate, TimeSelect...) act along it.
  - The **outer time** `T_k` is where a Loop rolls out: it cycles over it,
    and its result gains its rollout as a new outer time.
  - `()` is a stream without time: a number, a Parameter or Constant
    declared without time, an Input read whole.
- `dim` (D\*): one or more feature axes.

`Shape(dim, time)` holds the two tuples; there is no `seq` any more, on
`Shape`, `Stream`, `Layer` or `Parameter`. `time` takes a tuple, or an int
for one axis.

**Inputs.** `x.sw(n, seq=...)`, `x.last(seq=...)` and `x.next(seq=...)`
declare the time axes outside the window (an int or a tuple, outermost
first): `x.sw(3, seq=(5, 2))` has time `(5, 2, 3)`. `seq` is only this
argument. Every window of an Input has the same outer time, which its first
window sets. A later window may leave it out, which inherits it, but may not
change it (`ValueError`). Windows still differ freely in their inner time.

**Values.** `Parameter`/`Constant`: without a value, `time` gives the sizes
of all the time axes; with a value, it names one or more of its axes,
outermost first, and the rest are dims. `.value` is stored in the tensor
order `(*time, *dim)`.

**Arithmetic** (`Shape.broadcast`, used by every `BinaryOp`):
- Streams with time axes must have exactly the same ones; different time
  is never blended (`ValueError`). A dynamic axis matches any length and stays
  dynamic.
- A stream without time broadcasts along any time.
- The dims must be the same, or one of them a scalar (every axis of size 1),
  which broadcasts to the other's as numpy aligns them: `(1, 1)` with `(3,)`
  gives `(1, 3)`. A vector never broadcasts to a matrix, even one whose rows
  it fits: `(3,)` against `(2, 3)` is refused.
- One operand must already have the result's shape, and the others
  broadcast into it. Two operands that would both be broadcast are refused:
  a timeless spatial matrix and the window of a scalar, as in `x3 + x1tm`,
  would merge a temporal quantity with a spatial matrix. A time of length 1
  is not a broadcast, so `x3 + x1t` is allowed.

At run time `BinaryOpImpl` lines operands up with `align`: missing time axes
go right after the batch, and the missing leading axes of a scalar dim in
front of the dim axes.

## Decisions

1. **Time ordering.** Settled: outermost first. Still open: whether the
   `Seq*` layers become `Time*` layers whose default `axis=-1` ("the rollout
   of the last Loop") becomes `axis=0`. I recommend it.
2. **Where `features` goes in `Fir(features>1)`** (and LocalModel cells).
   Open. It is still the leading dim axis. Channels-last makes the trailing
   dim axis natural, as in `Dense`, and free in the GEMM. I recommend
   trailing.
3. **What `axis` means in `Softmax`/`PReLU`/`BatchNorm`.** Open. It is a raw
   tensor axis with batch as 0, so `BatchNorm`'s default `axis=1` is now a
   time axis for timed streams. I recommend a semantic dim axis.
4. **How `seq` is spelled.** Settled: it doesn't exist; there are time
   axes, inner and outer. Only the windows' `seq=` argument remains, to
   declare an Input's outer time; a window that omits it inherits the
   Input's. `Parameter.value` is stored in the tensor order `(*time, *dim)`.
   Numbers and values declared without `time` have no time axis.

## Steps

1. **Centralize the shape, without changing it.** `core/shape.py` owns the
   layout; every layer asks it for its axes.
   → *verified:* TF full 810 passed, 1 skipped; JAX fast 746 passed;
   torch fast 749 passed; `pyright src` 0 errors. **Done.**
2. **The new `Shape`** (see "New convention"). Adapted in this step:
   - `core/shape.py`: the new `Shape`, `Shape.broadcast` and `align`, and
     `OUTER_TIME_AXIS = 1`; `expand_to_rank` and `outer_time_view` now
     follow the new layout;
   - `Stream`, and `Layer`, whose `output_shape()` now returns a `Shape`
     (a `TypeError` names any layer that still returns a tuple);
   - `BinaryOp` and `Identity`, `Input`, `SampleWindow`;
   - `Parameter`/`Constant` values, `Output`, `Fir`, `Linear`, `PReLU`;
   - in the core, the minimizer's dynamic-input check and `seq_weights`
     length, and `Modely.rollback` (feedback along the inner time).

   A model of Inputs, windows, Fir, Linear, arithmetic and Parameters builds
   and runs in the new layout (`test_broadcast_values_follow_the_shapes`).
   → *verified:* `tests/test_shape.py` 32 passed on TF, JAX and torch,
   including Fir filtering the inner time at every outer step; ruff clean;
   `pyright` clean on the structural modules. **Done.**
3. **Adapt the layers still on the old `Shape` API.** **Done 2026-10-20.** They still read `.seq`,
   `time_axis`, `window` or the `rollout_*` names, which no longer exist
   (`grep -rnE "\.seq\b|seq_rank|seq_axes|resolve_seq|\.window\b|\.time_axis\b|rollout|outer_steps" src`
   lists them, in partitions, Loop, Integrate, Derivative, OdeNet,
   LocalModel, Roll, Fuzzify, EquationLearner, DataLoader). By failure
   count in the fast suite:
   - `output_shape()` still returning tuples: Derivative, IntegrateStep,
     Differentiate, the Fuzzify memberships (Triangular, Rectangular,
     Gaussian), the `Seq*` layers, LocalModel cells, OdeNet;
   - time read as one int (`.time == 1`, `init.shape.time`): Loop, Roll,
     `Modely.rollback`, Integrate, Derivative, TimeSelect, TimeRange,
     EquationLearner;
   - `Shape.replace(seq=...)` and 3-way `.dimensions` unpacking: Loop,
     SeqStack, SeqSelect, SeqRange, SeqConcatenate. With `seq` gone, a Loop
     consumes `outer_time_axis` and adds `with_outer_time`, and the `Seq*`
     layers become operations on the outer time axes;
   - `Loop._state_shape`/`_align_to_shape`: numpy right alignment of the
     layout tuples, which in the new layout is the same rule as `align`;
   - `BatchNorm` (decision 3), `EquationLearner`, `Interpolation`;
   - internal `Input(..., seq=...)`, which no longer exists: `Integrate`'s
     loop body and `EquationLearner` must declare it on their windows.

   → *verify:* each layer's own tests, restructured as in step 5.
4. **Data in and out.** **Done 2026-10-20.** `DataLoader._windows` builds `(N, *dim, time, *seq)`
   with `sliding_window_view`, which appends axes: it must build
   `(N, *time, *dim)`, reading the outer time from `node.shape.time[:-1]`
   instead of `node.seq`. Then check the padding and mask
   (`OUTER_TIME_AXIS`), normalization, minimizer masks and `seq_weights`,
   validation, and export.
   → *verify:* `test_dataset`, `test_minimize`, `test_validation`,
   `test_train`.
5. **Adapt the old tests: structure only, same results.** **Done
   2026-10-20**; the changes that touched a test's meaning are listed below. Arrays fed to
   models and loaders are transposed to `(B, *time, *dim)`. Asserted shapes
   are reordered (`.dimensions` is now `(time, dim)`). Numbers in arithmetic
   are timeless. `Input(seq=)` becomes `sw(..., seq=)`; `Parameter(seq=)`
   becomes `time=(...)`. Every asserted *value* stays the same.
   → *verify:* full TF, fast JAX and torch, as green as before step 2.
6. **Decisions 2 and 3** once the suite is green: Fir features trailing,
   semantic `axis` for Softmax/PReLU/BatchNorm, the `Seq*` → `Time*` merge
   (`TimeConcatenateImpl`, `ConcatenateImpl` and `SeqConcatenateImpl` are
   then one implementation). Bump the `.nnodely` format `VERSION`.
7. **Docs and messages.** `README`, `docs/guide/*.rst`, `CHANGELOG.md`,
   `NNODELY_AI_GUIDE.md`, and the docstrings and error messages that spell
   the old layout (`grep -rn "\*dim, time" src docs`).
   → *verify:* `sphinx -W` build, all guide snippets run.

Through every step, `.keras` exports are not portable across the migration:
Keras layer configs and input shapes both change. 2.0 is unreleased, so no
compatibility code.

## Step 1: what `core/shape.py` owns

Keras layers (`*Impl`) don't derive axes from ranks. The symbolic layer
passes them the tensor axis that its `Shape` resolves. Fir, Linear and
`FuzzyProductImpl` permute every axis: they rebuild their input's `Shape`
from the tensor, compute in a fixed view and permute back. `transpose` skips
permutations that change nothing, so in the new layout Linear's
permutations cost no op.

| What | API (as of step 2) |
|---|---|
| Axes | `time`, `dim`, `time_rank`, `dim_rank`, `rank`, `tuple`, `dimensions` (= `(time, dim)`) |
| Tensor axes | `time_axes`, `dim_axes`, `inner_time_axis`, `outer_time_axis` |
| Inner and outer time | `inner_time`, `outer_time` (sizes), `with_outer_time(n)` (a Loop's result), `without_outer_time()` (a Loop's body) |
| Checked axis | `resolve_dim`, `resolve_time` (from the outermost; `-1` is the inner time), `resolve_axis` |
| Reading a tensor shape back | `Shape.from_tensor_shape(shape, time_rank= \| dim_rank=)` |
| Deriving shapes | `replace(dim=, time=)` |
| Arithmetic | `Shape.broadcast(shapes, owner)`, `align(x, ...)` |
| Raw tensors | `BATCH_AXIS`, `OUTER_TIME_AXIS`, `outer_time_view`, `expand_to_rank`, `slice_axis`, `transpose`, `inverse_permutation` |

## Steps 3-5: what changed (2026-10-20)

Layers: partitions (Time* on the inner time, Seq* on the outer time axes
outermost first, default `axis=0`; `Concatenate` repeats a timeless input),
Loop (rolls out along the outer time; `LoopImpl` takes `rolled_*` flags, no
per-axis lists), Roll, rollback, OdeNet (points as a new outer time),
Derivative/IntegrateStep (`init` one sample or timeless), Integrate (outer
time, internal Inputs windowed), LocalModel (`CellWeight` keeps the "time 1
weighs every sample" broadcast; `FuzzyProductImpl` works channels-last, no
transposes), Fuzzify, EquationLearner, BatchNorm (default first dim axis),
DataLoader (windows built `(N, *time, *dim)`).

Simplified because of the layout: `LinearImpl` (no transposes: one
`tensordot`, or one `moveaxis` for `axis=k`), `LoopImpl` (the rollout is
always tensor axis 1), `FuzzyProductImpl`, Integrate (the scalar `init` is
spread by the Loop), Fuzzify (number centers no longer need a time).

Tests whose meaning had to change (all values kept):
- Models refused by the new arithmetic, restructured: a window times a
  one-sample stream (`test_calculus` mixed derivative: the Fir output is
  repeated with `TimeConcatenate`; `test_dataset` step alignment: two
  Outputs); a vector added to a matrix (`test_save_export`
  `_weighted_layers_model`: two Outputs); a one-sample divisor along a
  sequence (`test_shape_convention`: the divisor read whole).
- Raw tensor axes retargeted to the same semantic axis: `Softmax(axis=-1)`
  → `axis=1`, `BatchNorm(axis=-1)` → `axis=1`, custom Keras layers summing
  over `axis=1` → `axis=-1`.
- `Seq*` explicit axes renumbered outermost first.
- Regexes reworded with their message (Integrate init, Fuzzify centers,
  TimeConcatenate, Seq* concatenate/stack).
- `test_parameters_constants`: "time is one axis" no longer exists (time may
  name several axes); a Fir kernel declared with `time=` no longer matches
  the kernel layout, given plainly instead.
- `test_my_localmodel` fed 4 samples to inputs read with `.last()`; it now
  feeds the one sample read.

## Summary and next steps (2026-10-20)

### What was done
- `core/shape.py` owns the convention. `Shape(dim, time)` holds `time` as a
  tuple, outermost first; the inner time is the window, the outer time is
  the rollout. `Shape.broadcast` implements the arithmetic rules and `align`
  their runtime counterpart.
- `seq` was removed from `Shape`, `Stream`, `Layer`, `Input` and
  `Parameter`; it survives only as `sw/last/next(seq=...)`.
- Every layer was adapted (see "Steps 3-5"). Code got simpler where the
  layout allowed it: `LinearImpl`, `LoopImpl`, `FuzzyProductImpl`, Integrate
  and Fuzzify.
- All 23 test files were converted for shape only. The tests whose meaning
  had to change are listed in "Steps 3-5"; no asserted value changed.
- New tests in `tests/test_shape.py` (32) pin the convention, including your
  arithmetic specification.

### Edge cases found, not covered by tests
1. **Raw Input used whole, then windowed.** `x * 2.0` declared before
   `x.sw(3)` keeps a stale timeless shape, but runs on the `(B, 3, 1)` union
   window. This existed before ("raw Input stale shape") and is more visible
   now. Fix: freeze the Input shape at its first use, or refuse a later
   window.
2. **Loop over a timeless body input.** A seed `x.last(seq=3)` (time
   `(3, 1)`) does not roll out a timeless input: it needs time `(3,)`. The
   error ("cannot determine the rollout length") should say so.
3. **`SeqStack` of timeless streams.** The new axis becomes the inner time
   `(n,)`, a window and not a sequence. Refuse it, or document it.
4. **`validate()` and `train()` disagree.** `validate()` accepts a timeless
   data target against a time-1 source, because it compares after
   flattening; `train()` refuses the same pair. Make validate check shapes
   like `_align_target`.
5. **ONNX export of `Concatenate` with a timeless input.** It repeats the
   input with `zeros_like + add`. It runs and saves, but no ONNX test covers
   it.
6. **A Fir/Linear kernel Parameter declared with `time=`.** It is stored
   `(*time, *dim)`, while kernels are laid out `[*dim, window, features]`,
   so it no longer matches. Kernels should be given as plain values (all
   dim). This calls for a kernel layout that follows the convention
   (decision 2).
7. **`.keras` exports.** Files exported before the migration do not import:
   Impl configs and input layouts changed. `.nnodely` saves made with the
   old layout are not compatible either (weights layout); bump the format
   `VERSION` (step 6).
8. **The local `examples/` folder** (untracked) still uses the old API:
   about 33 more pyright errors than before.

### Assessment of the new convention
Better overall:
- Channels-last matches Keras and data files, `(time, features)`: Linear
  is one `tensordot`, Keras defaults (`axis=-1`) mean "per feature", and the
  DataLoader no longer transposes.
- One concept (inner and outer time) replaces `time` + `seq`. A Loop always
  rolls axis 1, and masks and `seq_weights` run along axis 1.
- Timeless values (numbers, Parameters) say what they are.
- The strict broadcast rule caught at least one model that added a vector to
  a matrix by accident (`test_save_export`).

Costs to weigh:
- The strict time rule removes convenient broadcasts: a 1-sample value
  against a window (`fir(...) * x.sw(2)`, `x.sw(3) + y.last()`) or against
  a sequence. These now need `TimeConcatenate` repetition, a timeless
  operand, or a dedicated layer (LocalModel's `CellWeight`). Consider letting
  an inner time of length 1 broadcast along a window: it covers the common
  physics pattern without blending different windows.
- Raw tensor axes changed meaning silently: Softmax, PReLU and BatchNorm
  `axis`, and custom Keras layers that sum over `axis=1`. That makes
  decision 3 (semantic axes) urgent.
- Raw Inputs became timeless: a stream that used to be one sample is now no
  time at all, a beginner pitfall. Many tests had to switch to `.last()`.

### Next steps, in order
1. Review the `NNODELY_AI_GUIDE.md` update and run its recipes.
2. Decide 1-3, then implement:
   - `Seq*` → `Time*` with `axis` over all time axes (default 0);
   - Fir features trailing, plus a kernel layout that follows the stream
     (`FirImpl` loses its channel rotation);
   - semantic `axis` for Softmax, PReLU and BatchNorm.
3. Decide on broadcasting an inner time of length 1 (see above).
4. Fix edge cases 1-4. Add an ONNX test for case 5.
5. Bump the `.nnodely` format `VERSION`. Note in the CHANGELOG that `.keras`
   files are not portable across this change.
6. Migrate `examples/`.
7. Commit in reviewable pieces: `core/shape.py` + core, layers, tests, docs.
