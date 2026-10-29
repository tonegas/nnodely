# nnodely — reference for AI code generation

Target reader: an LLM or coding agent that has to write, debug or explain Python code
using **nnodely 2.0.0** (Model-Structured Neural Networks on Keras 3; backends TensorFlow,
PyTorch, JAX; Python 3.10–3.13). The file is dense on purpose, and exact signatures are
given. Items marked **[verified]** were checked by running code against the source tree
(TensorFlow backend, Keras 3.12.1). The recipes in §15 run once `numpy` is imported as `np` (as in R1) and their data arrays (`th`, `om`, `tq`, `ref`, `times`, `xs`, `Fs`) are defined.

Use this order:
1. Read §1 (hard rules) before writing anything.
2. Map the user's equations to blocks with §2.
3. Compute the shape of every stream with §3 before wiring.
4. Choose the training scheme: one-step (§8), multi-step (§10), physics residual (§11).
5. Start from the closest recipe in §15.
6. Run the checklist in §17 before returning code.

---

## 1. Hard rules

1. **The lifecycle order is fixed:** declare streams → `Modely(name, inputs, outputs)` (or `ModelyLoop`, §10.4) → `minimize(...)` / `rollback(...)` → `build()` → `DataLoader(model, ...)` → `train` / `validate` / `model(...)` / export. `DataLoader`, `train`, `validate`, inference and export all need a built model. Objectives and rollback must be declared **before** `build()`, and `rollback` after `build` raises. `build()` returns the model, so `Modely(...).build()` chains.
2. **`Input(name, *, dim=None, seq=None, sw=None)`**: all keyword-only, so `Input("x", 3)` is an error. An Input is **declared once with the shape of its data and never changes**. `sw` is its window: `sw=n` the last `n` samples up to the current one, `sw=(past, future)` adds future samples, `T = past + future`. `seq` is its sequence, the axes outside the window (§3): `Input("s", seq=H, sw=1)`, `-1` or `"dyn"` for a dynamic length; **a sequence needs a window** (`ValueError` otherwise).
3. **Without `sw` an Input has no time axis**: one value per sample, `Shape(S=(), T=None, D=dim)`, data `(batch, *dim)`. There is no `sw()`, `next()` or `last()` on an Input [verified]: layers take what they need of its window, `TimeSelect(-1)(x)` the newest sample, `TimeRange(0, n)(x)` a part. To read a signal both through a window and as one sample, declare the union window and select: `x = Input("x", sw=(5, 1))`, `TimeRange(0, 5)(x)` the past, `TimeSelect(-2)(x)` the current sample, `TimeSelect(-1)(x)` the next one.
4. **Stream operators:** `+ - * / **` work between streams and Python numbers; numbers become `Constant`s without a window, applied to every sample. Windows must be equal, or one of them a single sample, `T=1`; a stream without a window read from data does not combine with one that has a window, while a value without one (a number, a Parameter, a Constant) combines with any stream (§3). Unary minus is not supported: `-x` raises `TypeError` [verified], so write `-1.0 * x` or `Negative()(x)`. Also unsupported: `2 ** x`, `@`, indexing `x[0]`, comparisons. For indexing use `Select` / `Range` (dim), `TimeSelect` / `TimeRange` (window) or `SeqSelect` / `SeqRange` (sequence).
5. **Objective targets** (`minimize(name, source, target, loss)`). `target` is one of:
   (a) an `Input` (`Input("y", sw=1)`, its current sample, or `Input("y", seq=H, sw=1)` for a trajectory), or a part of its window (`TimeSelect(-1)(y)` of `Input("y", sw=(1, 1))`, the next sample);
   (b) any stream computed from the data, such as `Sin()(y)` or `y * 2` [verified: the loss is computed against `sin(y)` and `2y`];
   (c) a number (`0` and `0.0` alike), which becomes a `Constant`;
   (d) `None`, which drives `source` to 0.
6. **Objectives are identified by name.** Several objectives may read the same source, and each is logged under its own name [verified]. Registering a name again replaces the objective it names, with a warning.
7. **Per-objective weights:** `minimize(..., gain=0.5)` weighs an objective in the total loss; its logged loss stays unweighted [verified]. `seq_weights=` weighs the steps of a rollout or window (§8).
8. **Numbers as targets:** `target=0` and `target=0.0` are the same; a number becomes a one-sample `Constant`, broadcast to the source's shape [verified]. A target read from data must have exactly the source's shape.
9. **Names are keys, not identities.** Input and Output names must be unique within a model: they are the keys of data dicts, results and history. Layer names do not decide weight sharing: two layer objects are two layers with their own weights, even when created with the **same explicit `name=`**; Keras then receives them as `twin` and `twin_1` [verified]. Pass `name=` to look a layer up later (`model.model.get_layer(name)`). Auto names are `Linear1`, `Fir2`, ... from a global counter that skips names in use.
10. **Weight sharing means reusing the same layer object:** `lin = Linear(features=4)`, then `lin([a])` and `lin([b])` share weights. A `Linear(...)` created *inside* a function that runs several times (for example the `f` of `Ode`, which runs once per RK stage) creates new weights on every call. Create weight-bearing layers and `Parameter`s outside `f` and call them inside.
11. **Body models** passed to `Loop`, `Roll` or `OdeNet` keep their own objectives to themselves: the outer model neither trains them nor needs their target inputs [verified]. Put the objectives on the outer model.
12. **Inference reads only what the outputs need:** `model(data)` takes the Inputs in `model.inference_inputs`. Target-only Inputs can be left out and extra keys are ignored [verified]. A missing needed input raises `ValueError` naming it.
13. **`validate()` scores every objective as training does**, including residual (`target=None`) and number targets: its loss is `train()`'s `val_<name>_loss`, its metrics are per channel and leave padded rollout steps out (§13) [verified].
14. **`save()` / `load()`** (nnodely format) **supports every model**, including `Loop`, `Roll` and `OdeNet`, whose bodies are saved in sub-folders, and `ModelyLoop`. A closed Input is saved with the stream it is closed with and its `init`, so a loaded model called in a `ModelyLoop` feeds it back as the saved one does [verified]. `save` warns where that cannot hold: a trained weight read only by such a stream, which the model does not compute, is saved as declared; and a call giving a stream for a closed Input its model declares keeps nothing of its feedback. Weights are saved and loaded by default; `save(path, weights=False)` stores the architecture alone and `load(path, weights=False)` ignores saved weights, and either way the model is initialized as `build()` does. A loaded model's layers are its own: they never share weights with the model it was saved from or with another load of the same file. The method an `OdeNet.set_method()` switched to is saved with it. A `Parameter` or `Constant` read by several models of the tree (Loop bodies, blocks, the outer model) is one object: saved under one id in each model's file, it is loaded as one object, listed once in `parameters`, and frozen everywhere at once [verified]. Objectives are not saved: a loaded model has no minimizers; call `minimize(...)` and `build()` again before `train()` [verified].
15. **`Softmax()` normalizes each sample as a whole**: its values over every sequence, window and dim axis sum to one [verified]. For a softmax over the features use `Softmax(axis=-1)` [verified]: `axis` is a raw tensor axis (batch 0, then the sequence, the window, the dims last), so `-1` is the last dim axis. `BatchNorm()` already normalizes per feature (its default axis is the first dim axis).
16. **`Differentiate(respect_to=x)`**: `x` must be the `Input` object. The result has the shape of `x`, its whole window: zero for the samples the differentiated relation does not read [verified]. It does not work inside a model that is used as a composed block (§12).
17. **`train()` defaults to `batch_size=1` and `epochs=10`.** Always pass both (typical batch sizes are 32–256).
18. **Set `KERAS_BACKEND` before the first import** of `keras` or `nnodely`. With no value set, the backend is TensorFlow. A backend that is not installed raises `ImportError` naming the extra to install.
19. **`Parameter` initializes to `random_normal` when no value is given.** For physical constants pass `value=<initial guess>`.
20. **`dt` is always explicit** for `IntegrateStep`, `Integrate`, `DerivativeStep` and `Derivative`; nothing is inferred from data. Windows are counted in samples, not seconds.
21. **Using a `Loop` node directly** gives the trajectory of its **first callback output**. Unpack it (`a, b = Loop(...)`) to get every body output, in `f.outputs` order [verified].
22. **A model called twice as a block** returns outputs with the same names both times. Combine them, or wrap each in an `Output` with a distinct name; otherwise `build` raises `two different outputs named ...`.
23. **`DataLoader(..., delimiter=, header=)` are passed to `pd.read_csv`** as its `sep` and `header`. Missing (n/a) cells are kept as NaN, with a warning naming the input and the count [verified].
24. **Inference arrays are always batched:** `(batch, *seq, time, *dim)`, a batch of 1 for one sample; an Input without a window is `(batch, *dim)`. `x.shape` gives the per-sample shape, as declared. Flat lists raise; an unbatched array raises, or is silently misread with its first axis taken as the batch [verified]. `loader[i]` has no batch axis: pass `{k: v[None] for k, v in loader[i].items()}`, or `loader.as_dict()` for every sample.
25. **`x.close(stream, init=None)` closes an Input, and only a `ModelyLoop` feeds it back** (§10.4). `stream` is one sample of `x` (its dim and sequence, `T=1`; no window for an Input without one) and is returned: `x_next = x.close(IntegrateStep(dt=dt)(x, v_next))`. Inside the step `x` holds the value at the end of the previous step. Built as a `Modely`, the same graph is the one-step model and its closed Inputs are read from data, with a `UserWarning` for those it does not declare [verified]. `Integrate`, `Derivative` and `Buffer` close Inputs of their own, which a `ModelyLoop` feeds back; built as a `Modely`, these start from the layer's `init` (zero without one), as at the first step of a `ModelyLoop`, and are never read from data [verified].
26. **A layer is applied by calling it on streams.** A layer object used as a stream itself - `b + Exp()`, `Output("y", Sin())` - reads nothing, and raises `ValueError` ("was never called on a stream") [verified]: write `Exp()(b)` or `Exp(b)`. `Fir` and `Linear` read one stream; given several they raise ("reads one stream, got 2"): join them first with `Concatenate` or `TimeConcatenate`.

---

## 2. Mapping equations to blocks

| Physical or modeling concept | nnodely construct |
|---|---|
| Measured signal `x(t)`, one sample per step | `x = Input("x", sw=1)`; vector signal: `Input("x", dim=3, sw=1)` |
| A static value per sample (no time: PINN coordinate, operating point) | `Input("x")`, no window; it does not combine with windowed streams |
| Past n samples `x[t-n+1..t]` | `x = Input("x", sw=n)` |
| Past n samples and the next one | `x = Input("x", sw=(n, 1))`: `TimeRange(0, n)(x)` the past, `TimeSelect(-2)(x)` the current `x[t]`, `TimeSelect(-1)(x)` the next `x[t+1]` (typical one-step target) |
| Trajectory of H steps (rollout data, seed, target) | `Input("x", seq=H, sw=1)`: H consecutive samples as a sequence axis; `seq=-1` follows the data |
| Linear combination of past samples (ARX / FIR term) | `Fir(features=1)([x])` with `x = Input("x", sw=n)` |
| Static affine map / MLP layer on features | `Linear(features=k)([s])` plus an activation |
| Unknown physical constant (mass, stiffness, g/l) | `Parameter("m", value=guess)` |
| Known constant | `Constant("g", value=9.81)`, or just a number in arithmetic |
| Nonlinearity `sin`, `exp`, `abs`, saturation | `Sin()(s)`, `Exp()(s)`, `Abs()(s)`, `Clamp(min, max)(s)` |
| Lookup table / characteristic curve (fixed) | `Interpolation(x_points, y_points)(s)` |
| Gain scheduling / operating regimes | `Fuzzify(centers, function=Triangular)` → `LocalModel(lambda i: Fir(features=1))([x], [membership])`, `x` with `sw=n` |
| Symbolic regression of an unknown law | `EquationLearner(["sin", "multiply", ...], linear_out=Linear(1))` |
| `du/dx` exact (PINN residual, Sobolev) | `Differentiate(order=1, respect_to=x_input)(u)` |
| `dx/dt` from a sampled window | `DerivativeStep(dt=dt, window=...)(init, x)`, `x` with `sw=n` |
| `dx/dt` step by step, in a model declared one step at a time | `Derivative(dt=dt, init=x0)(x)` with `x = Input("x", sw=1)`, in a `ModelyLoop`: the inverse of `Integrate` |
| `∫ rate dt` over a window / one Euler state update | `IntegrateStep(solver="euler", dt=dt)(state, rate)` |
| `∫ rate dt` step by step (one rate sample per step) | `Integrate(solver="euler", dt=dt, init=x0)(rate)` with `rate = Input("rate", sw=1)`, in a `ModelyLoop` |
| A state `x[k]` computed from `x[k-1]`; several coupled states | `x = Input("x", sw=1)`, `x_next = x.close(<stream of x and the others>)`, built with `ModelyLoop` (§10.4) |
| The last n values of a signal computed step by step (a controller's commands read by a plant) | `Buffer(sw=n)(u)`, in a `ModelyLoop` |
| One explicit RK step of `dx/dt = f(x, u, θ)` | `Ode(f, [states], dt, method="rk4", args=(...))` |
| Learned vector field integrated over reported times | `OdeNet(f=field_model, states={...}, t=times_stream)` |
| Multi-step / simulation-error training | `ModelyLoop` (states declared one step at a time), `Loop` (a body rolled out over a sequence) or `Modely.rollback` (k-step ahead) |
| Physical law as a soft constraint | `model.minimize("law", residual_stream)` (no target) |
| Reuse a trained sub-model | `sub_model([stream_a, stream_b])` (composition) |

---

## 3. Shapes and windows

**Semantic shape** (no batch axis): `(*seq, time, *dim)`, printed `Shape(S=seq, T=time, D=dim)`; a stream without a window is `(*seq, *dim)`, printed `T=None`.
- `seq` (`S`): a tuple of **sequence axes**, outermost first, possibly empty. A `Loop` rolls out along the outermost one; a collected rollout is prepended as a new outermost sequence axis. Declared with `Input(..., seq=, sw=)`. A dynamic axis (`seq=-1` or `"dyn"`) is `None` internally, `dyn` when printed.
- `time` (`T`): the **window** of consecutive samples, an int, or `None` for no window. `Shape(time=(5, 4))` or `time=0` raises `TypeError`. `Fir`, `DerivativeStep`, `IntegrateStep`, `Roll`, `rollback` and the `Time*` layers act on it, and raise `ValueError` on a stream without one ("acts on the window of its stream ... has none"). A **single sample** is `T=1`. A **value** - a number, a `Parameter` / `Constant` declared without time, or a relation of those alone - has no window and applies to every sample of any stream.
- `dim` (`D`): feature axes. Default `(1,)`; `dim=3` gives `(3,)`; `dim=(3,2)` gives a matrix per sample.

**Runtime tensor**: `[batch, *seq, time, *dim]`, or `[batch, *seq, *dim]` without a window: the sequence axes are tensor axes `1 … seq_rank`, the window is axis `1 + seq_rank` (`time_axis` is `None` without one), the dims are the last axes. The **outer axis** (tensor axis 1) is the outermost sequence axis, or the window without a sequence: rollout masks and `seq_weights` run along it. Every data array passed to a model or produced by the `DataLoader` follows this layout.

Stream attributes: `s.shape` (a `Shape(dim, time, seq)`, tuple-like over its axes), `s.seq` (a tuple), `s.time` (an int or `None`), `s.dim`, `s.name`, `s.shape.tuple`, `s.shape.dimensions` (= `(seq, time, dim)`), `s.shape.seq_rank`, `s.shape.dim_rank`, `s.shape.outer` (the outer axis's length), and the tensor axes `s.shape.seq_axes`, `s.shape.time_axis`, `s.shape.dim_axes` [verified: `Input("w", dim=3, seq=5, sw=10)` prints `Shape(S=(5,), T=10, D=(3,))`, `seq_axes` `(1,)`, `time_axis` 2, `dim_axes` `(3,)`].

**The window of an `Input`**, declared once:

| Declaration | Samples | `T` |
|---|---|---|
| `Input("x", sw=n)` | `x[t-n+1] … x[t]` | `n` |
| `Input("x", sw=(p, f))` | `x[t-p+1] … x[t], x[t+1] … x[t+f]` | `p+f` |
| `Input("x", sw=(0, 1))` | `x[t+1]` | `1` |
| `Input("x")` | `x[t]`, without a time axis | `None` |
| `Input("x", seq=s, sw=...)` | `s` consecutive windows, the last ending at `t` (§9) | as above, with `S=s` |

`sw` is an int ≥ 1 or a pair `(past, future)` of ints ≥ 0 with at least one sample (`ValueError` otherwise). `seq=` (an int or a tuple, outermost first; `-1` or `"dyn"` dynamic) is the Input's sequence and **needs `sw`**: `Input("x", seq=(4, -1), sw=3)` is `S=(4, dyn), T=3` [verified]. An Input never changes: its `shape` is the shape of its data, before and after any build, and `x.past`, `x.future` are its window. Layers take parts of it: `TimeSelect(i)(x)` one sample (`T=1`; `-1` the newest), `TimeRange(a, b)(x)` a part. Windows are ordered oldest first. Every model of an Input reads the same window: for different windows of one signal, declare their union and select from it, or declare two Inputs and feed the same column under both names.

**Shape rules of common layers** [verified]. `v = Input("v", dim=3, sw=4)`, `S=(), T=4, D=(3,)`:

| Expression | Output shape |
|---|---|
| `Fir(features=2)([v])` | `S=(), T=1, D=(3, 2)`: each of the 3 features filtered alone over its 4 samples onto 2 channels, a new last dim axis; one output sample; the sequence is kept (an Input with `seq=5, sw=4` gives `S=(5,), T=1`). `Fir(features=1)` keeps the dim, `D=(3,)`; on a scalar the channels replace it, `Fir(features=2)([Input("x", sw=4)])` is `S=(), T=1, D=(2,)`. A stream without a window raises |
| `Linear(features=5)([v])` | `S=(), T=4, D=(5,)`: per-sample map of the whole dim, sequence and window kept (none stays none); a `D=(2, 3)` input gives `D=(5,)` too, and `axis=k` maps only dim axis `k` (`Linear(5, axis=1)` on `D=(2, 3)` gives `D=(2, 5)`), with a matrix per row unless `shared_kernel=True` |
| `Sum()([v])` | `S=(), T=4, D=(1,)`: sums dim axes (keepdims) |
| `Select(1)([v])` | `S=(), T=4, D=(1,)` |
| `Range(0, 2)([v])` | `S=(), T=4, D=(2,)` |
| `TimeSelect(-1)([v])` | `S=(), T=1, D=(3,)`: the window is always kept |
| `SeqSelect(-1)([Input("s", seq=5, sw=4)])` | `S=(), T=4, D=(1,)`: the sequence axis is removed |
| `TimeRange(0, 2)([v])` | `S=(), T=2, D=(3,)` |
| `Concatenate()([TimeSelect(-1)(v), w])`, `w = Input("w", dim=2, sw=1)` | `S=(), T=1, D=(5,)`: along dim axis 0 |
| `TimeConcatenate()([TimeRange(0, 2)(v), TimeRange(1, 4)(v)])` | `S=(), T=5, D=(3,)` |
| `Fuzzify(centers=[0,1,2], function=Triangular)([g])`, `g = Input("g", sw=1)` | `S=(), T=1, D=(3,)`: one feature per center |
| `Fuzzify(centers=[0,1,2], function=Triangular)([Input("g4", sw=4)])` | `S=(), T=4, D=(3,)` |
| `Fuzzify(centers=Parameter("c", value=[0,1,2]), function=Triangular)([g])` | `S=(), T=1, D=(3,)`: trainable centers |
| `LocalModel(lambda i: Fir(features=1))([t], [Fuzzify(...)([g])])`, `t = Input("t", sw=10)` | `S=(), T=1, D=(1,)` |
| `DerivativeStep(dt=dt)(x)`, `x` with `sw=n` | `S=(), T=n, D=(1,)`: window length preserved |
| `Derivative(dt=dt)(r)`, `Integrate(dt=dt)(r)`, `r = Input("r", sw=1)` | `S=(), T=1, D=(1,)`: one value per step of a `ModelyLoop`; a signal with a sequence or a longer window raises |
| `Buffer(sw=3)(u)`, `u = Input("u", sw=1)` | `S=(), T=3, D=(1,)`: `[u[k-2], u[k-1], u[k]]` |
| `Differentiate(respect_to=t_input)(u)` | the shape of `t_input` |
| `IntegrateStep(dt=dt)(a)`, `a` with `sw=n` | `S=(), T=n, D=(1,)` |
| Elementwise layers (activations, trig, `Exp`, `Clamp`, `Interpolation`, `BatchNorm`) | unchanged, with a window or without |

**Arithmetic broadcasting** (`+ - * / **`, `Shape.broadcast`) [verified]:
- **Window**: equal, or one of them a single sample, `T=1`, which applies to every sample of the other's window: `x + TimeSelect(-1)(x)` with `x` of `sw=3` gives `T=3`; `Input("a", sw=3) + Input("b", sw=2)` raises `ValueError` ("windows of 2 and 3"). A **value** without a window applies to every sample of any stream. A stream **without a window read from data** (an Input without `sw`, or a relation of it) does not combine with one that has a window (`ValueError`, "a stream with no time"); two of them combine, without a window.
- **Sequence**: the same - a dynamic axis matches any length and stays dynamic - or one of them a single step (every sequence axis 1, or none), padded with leading axes as numpy does: `s + x` with `s = Input("s", seq=4, sw=1)` and `x = Input("x", sw=1)` gives `S=(4,), T=1` [verified].
- **Dim**: equal, or one of them a scalar (every axis 1), which broadcasts numpy-style: `(1, 1)` with `(3,)` gives `(1, 3)`. A vector never broadcasts to a matrix: `(3,)` with `(2, 3)` raises.
- **One operand must already have the result's shape**, a value without a window counting as `T=1`; there is no two-way broadcast. A `(2, 3)` Constant plus `Input("x", sw=4)` raises (each would be broadcast); plus `Input("x", sw=1)` gives `S=(), T=1, D=(2, 3)`. `Input("s", seq=4, sw=1) + Input("y", sw=3)` raises too.
- Examples: `v * Input("x", sw=4)` gives `S=(), T=4, D=(3,)`; `v * 2`, `v * Parameter("g", value=2.0)` and `v + Constant("c", value=[1., 2., 3.])` give `S=(), T=4, D=(3,)`.
- At run time the missing sequence axes of an operand are inserted right after the batch, a value's missing window right before its dims, and the missing leading dims of a scalar before its dims. Parameters and Constants carry the batch axis like any stream.

**Parameter / Constant shapes** [verified]: by default every axis of a `value` is a dim axis and the value has **no window**: `value=2.0` gives `S=(), T=None, D=(1,)`; `value=[1,2,3]` gives `D=(3,)`; `value=[[1,2]]` gives `D=(1, 2)`; an array of shape `(2,3,2)` gives `D=(2, 3, 2)`. Given with a `value`, `dim` / `time` / `seq` are **axes of the value**, not sizes. `time` names one axis, the window (a tuple raises `TypeError`), `seq` one or several, outermost first, and the axes they leave are the dim unless `dim` names them: `value=[[1,2]], time=1` gives `S=(), T=2, D=(1,)`; `value=ones((2,3,4)), dim=0, time=1, seq=2` gives `S=(4,), T=3, D=(2,)`; `dim=(2, 0), time=1` gives `S=(), T=3, D=(4, 2)`; `dim=(1, 0)` alone transposes a `(2, 3)` value into `D=(3, 2)`. When `dim` is given, `dim`, `time` and `seq` together must take every axis once (`ValueError` otherwise). Without `value` they are sizes: `dim=3` gives `S=(), T=None, D=(3,)`, `dim=(2,3), time=4` gives `S=(), T=4, D=(2, 3)`, `dim=2, time=4, seq=3` gives `S=(3,), T=4, D=(2,)`. `.value` (the value as declared, which training does not change) and the variable (`.param`, the current one) are laid out `(*seq, time, *dim)`, without the window when no `time` is given: a `(3, 2)` value is a `(3, 2)` variable, its stream `S=(), T=None, D=(3, 2)`; `value=ones((2, 3)), time=1` is `S=(), T=3, D=(2,)` and a `(3, 2)` variable.

---

## 4. Public API and imports

```python
import os; os.environ["KERAS_BACKEND"] = "torch"   # or "tensorflow" / "jax"; BEFORE imports
from nnodely import (
    Modely, ModelyLoop, DataLoader, Input, Output, Parameter, Constant, set_seed, get_seed,
    # layers with weights
    Linear, Fir, LocalModel, EquationLearner, BatchNorm,
    # fuzzy / lookup
    Fuzzify, Triangular, Rectangular, Gaussian, Interpolation, KerasFunction,
    # dim axes / window / sequence axes
    Select, Range, Concatenate, Sum,
    TimeSelect, TimeRange, TimeConcatenate, TimeReverse,
    SeqSelect, SeqRange, SeqConcatenate, SeqStack, SeqReverse,
    # calculus / recurrence
    Differentiate, DerivativeStep, Derivative, Integrate, IntegrateStep, Ode, OdeNet, Loop, Roll, Buffer,
    # activations
    ReLU, LeakyReLU, ELU, PReLU, Sigmoid, Tanh, Softmax, Swish, GELU, Softplus,
    # math
    Sin, Cos, Tan, Asin, Acos, Atan, Atan2, Exp, Log, Log10, Sqrt, Abs, Floor, Ceil,
    Deg2Rad, Sign, Negative, Clamp,
    # training printers (callbacks)
    TinyPrinter, LegacyPrinter, NNodelyPrinter,
)
# Not exported at top level:
from nnodely.core.layer import Layer, Identity                # base class for custom layers
from nnodely.core.validation import ValidationResult
import nnodely; nnodely.__version__
```

Install: `pip install "nnodely[torch]"` (or `[tensorflow]` / `[jax]`); add `onnx` for ONNX export (`"nnodely[torch,onnx]"`).

---

## 5. Core objects

### Input
`Input(name: str, *, dim: int | tuple | None = None, seq: int | str | tuple | None = None, sw: int | tuple | None = None)`
- Declared once with the shape of its data, and never changed (§3): `sw=n` the last `n` samples, `sw=(past, future)` adds future samples; without `sw` no time axis.
- No methods take windows: there is no `.sw()`, `.next()` or `.last()`. `TimeSelect` / `TimeRange` take parts of the window.
- Attributes: `.shape` (the per-sample shape of its data, as declared), `.past`, `.future` (its window; `(1, 0)` without one), `.seq`, `.dim`, `.time` (`None` without a window), `.name`, `.update` and `.init` (what `close` set; `None` while open).
- `close(stream, init=None)` → `stream`: feeds `stream` back into the Input at the next step, in a `ModelyLoop` (§10.4). `stream` is one sample of the Input: `T=1` (no window for an Input without one), its dim and sequence; otherwise `ValueError` ("closed with the value of one step"). An Input reading future samples cannot be closed, nor one already closed [verified]. `init`, the value before the first step: a number, a `Parameter` / `Constant`, an `Input` (its first step is read), or `None`.
- `seq=N` gives a fixed-length sequence (trajectory): `Input("x", seq=N, sw=1)` is `S=(N,), T=1`. A tuple gives several sequence axes, outermost first. `seq=-1`, or `seq="dyn"` as a shape prints it (`seq=(4, "dyn")` in a tuple), gives a dynamic length, resolved by `DataLoader(seq_length=...)` and followed by a `Loop` at call time; only one dynamic axis per input is allowed. A sequence needs `sw`.
- An Input used only as a target (for example `Input("y_meas", sw=1)`) is read by training and validation (it is in `model.train_inputs`), not by inference (§1.12).

### Output
`Output(name: str, stream: Stream)`: gives a stream a public name. It is the key in results, history and `rollback` / `Loop` mappings.

### Parameter (trainable)
`Parameter(name=None, *, value=None, initializer="random_normal", time=None, dim=None, seq=None, trainable=True)`
- With `value`: the shape comes from the value, all its axes in the dim (one sample, `S=(), T=1`) unless `dim` / `time` / `seq` name which axes of the value play each part (see §3).
- Without `value`: `dim` / `time` / `seq` are the sizes (`time` an int, `seq` an int or a tuple, outermost first; omitted, the Parameter is one sample), and `initializer` draws the value.
- `initializer` is any Keras initializer name or object.
- After build: `.param` (a Keras variable) and `.value_numpy`, laid out as the value (`(*seq, time, *dim)`, no window axis without `time`).
- `p.freeze()` / `p.unfreeze()` (each returns `p`), or `p.trainable = False`, keep the value fixed through training or release it; `trainable=False` creates it frozen. Before or after build: the next `train()` follows. The flag is saved with the model. A Parameter shared by several models is frozen for all of them.
- At runtime it has a batch axis like any stream, the same value for every sample; one of a single sample applies to every sample of any stream's sequence and window.

### Constant (fixed)
`Constant(name=None, *, value, time=None, dim=None, seq=None)`: the shape comes from `value`, its axes named by `dim` / `time` / `seq` as for a Parameter (§3). It is a Parameter that training does not change. After build: `.constant` and `.value_numpy`. Numbers in arithmetic become cached anonymous Constants.

### Layer calling convention
Configure first, then call on a stream or a list of streams:
- `Linear(features=4)([s])` and `Linear(features=4)(s)` are both accepted.
- Multi-input layers take a list: `Concatenate()([a, b])`.
- A layer whose constructor takes only `name` (`Sin Cos Tan Asin Acos Atan Atan2 Exp Log Log10 Sqrt Abs Floor Ceil Deg2Rad Sign Negative Sigmoid Tanh Swish Softplus`) can be applied as it is created: `Sin(x)` is `Sin()(x)`, `Atan2([y, x])` is `Atan2()([y, x])`. A string first argument is still the name: `Sin("s")` configures. For a named layer write `Sin(name="s")(x)`. Layers with settings (`ReLU`, `Softmax`, `Sum`, `Clamp`, `Linear`, ...) are always configured first.
- `LocalModel` takes the inputs and a list of activations: `LocalModel(...)(inputs, activations)`.

The weights of `Fir` and `Linear` are `Parameter`s, predecessors of the layer in the graph:
- Pass them as `kernel=` / `bias=`: a `Parameter` (of the weight's shape, axes of size 1 aside), or a Keras initializer by name or object, from which the layer makes one named `"<layer>_kernel"` / `"<layer>_bias"`. `bias=False` leaves the bias out. Known weights are set when declaring, with no assign after build: `Fir(features=1, kernel=Parameter(value=K), bias=False)`.
- Layers given the same Parameter share it. Freezing it (`p.freeze()`, or `model.freeze()`) fixes the layer; `model.parameters` lists the made ones too.
- After build, `node.kernel` / `node.bias` on the *returned stream node* are the Keras variables of those Parameters, in the Parameter's shape (`.assign(np.array(...))` works); `bias` is None without a bias.
- `Fir` kernel `(*dim, window, features)`, bias `(*dim, features)`; without the `*dim` for a scalar input or `shared_kernel=True`. Window rows are the samples of the window, oldest first: `Fir(features=2)([x])` with `x = Input("x", sw=4)` has kernel `(4, 2)`, on `D=(2, 3)` `(2, 3, 4, 2)` [verified].
- `Linear` kernel `(*dim, features)`, bias `(features,)`. With `axis=k`: kernel `(*other_dim_axes, dim[k], features)`, bias `(*other_dim_axes, features)`, the other axes dropped with `shared_kernel=True`. On `D=(2, 3)`: `axis=1` gives `(2, 3, 5)` / `(2, 5)`, `axis=0` gives `(3, 2, 5)` / `(3, 5)` [verified].
- `LocalModel` cells with known weights: a factory giving each cell its Parameters, `LocalModel(lambda i: Fir(features=1, kernel=Parameter(value=K[i])))`; the returned stream has no `kernel`.

---

## 6. Layer reference

Legend: **in → out** gives semantic shapes `(*seq, time, *dim)`. `d` = dim, `S` = the sequence axes, `T` = the window.

### 6.1 With weights
| Signature | in → out | Notes |
|---|---|---|
| `Linear(features=1, kernel="glorot_uniform", bias=True, axis=None, shared_kernel=False, name=None)` | `(*S, T, *d)` → `(*S, T, features)`; with `axis=k`, dim axis `k` becomes `features` | Per sample of the window and every step of the sequence. Without `axis` the whole dim is mapped (every feature feeds every output). With `axis=k` only that axis is mapped, with a matrix and bias per position of the other dim axes, or one for all with `shared_kernel=True` (which requires `axis`). `bias=True` draws the bias with glorot, not zeros. Weights: §5. |
| `Fir(features=1, kernel="glorot_uniform", bias=True, shared_kernel=False, name=None)` | `(*S, T, *d)` → `(*S, 1, *d, features)` | Each dim element is filtered alone over its window (elements never mixed), with a kernel of its own or one for all with `shared_kernel=True`; applied at every step of the sequence. The channels are a new last dim axis. `features=1`, the default, keeps the dim (`(*S, 1, *d)`); on a scalar the channels replace it (`(*S, 1, features)`). `bias=True` is zeros. Weights: §5. |
| `LocalModel(input_function=None, output_function=None, vectorized=False, name=None)` called as `(inputs, activations: list)` | `a_k: (T, n_k)`: dim `(n_k,)` and no sequence, T the cells' window or 1 → shape of one output cell | The activations are multiplied into `N = Π n_k` joint memberships `mu`, row-major (cell `(i1, i2)` is `i1*n2 + i2`), and a `CellWeight` layer (`<name>_weighted`) weighs the cells with them: a 1-sample membership weighs every sample of a window. Activations with a sequence raise. Computes `Σ_i output_function_i(mu_i * input_function_i([x...]))`. Each function is: `None` (no function: one input weighted as it is / the weighted cells only summed); a weightless elementwise layer instance (`Sin()`, `Tanh()`, `ReLU()`, ...), applied once; any other `Layer` instance, applied to every cell with the same weights; or a callable, which is a factory `f((i1, i2, ...)) -> layer` called once per cell. With `vectorized=True` a callable or a layer (other than a weightless elementwise one) is instead called once on all cells and must return `(*cell, N)` (`(N,)` for scalar cells), e.g. `Fir(features=N)`: its N channels are the N cells. Functions receive a **list** of streams; all cells must have the same shape. Lists of functions are not accepted. |
| `EquationLearner(functions: list, *, linear_in: Linear \| None = None, linear_out: Linear \| None = None, name=None)` called on `stream` or `[streams]` | `(*S, T, d)` → `(*S, T, n_functions)`, or `linear_out`'s output | `functions` items can be a name (`identity sin cos tan asin acos atan relu leaky_relu elu prelu sigmoid tanh swish gelu softplus add subtract multiply divide power`), a Layer class or instance, a callable over streams (arity inferred), or `(callable, arity)`. `linear_in` projects to Σarity arguments; its `features` must equal that sum. Inputs must have dim rank 1 (several inputs are concatenated). Internally it is a composed `Modely`. |
| `BatchNorm(axis=None, momentum=0.99, epsilon=1e-3, center=True, scale=True, name=None)` | unchanged | Per-feature statistics, along the stream's first dim axis by default [verified]. An explicit `axis` is a raw tensor axis (batch 0, then the sequence, the window, the dims last). Uses the inference branch in `validate` and inference. |
| `Parameter(...)` | – | See §5. |

### 6.2 Fuzzy and lookup (no weights)
| Signature | in → out | Notes |
|---|---|---|
| `Fuzzify(centers, function, bounds=False, normalize=False, name=None)` called on a scalar stream | `(*S, T, 1)` → `(*S, T, n)` | One layer: its Keras layer (`FuzzifyImpl`) computes the bounds, every membership - the placed ones in one vectorized operation over all centers - and the normalization at once; its preds are `[x, *center streams, *Parameters of its function layers]`. `function` is required. `centers` are numbers (fixed, sorted), or move with the model: one stream of dim `(n,)` with one sample, `T=1` - a `Parameter` vector (trained) or a computed stream such as `Fir(features=n)` (every sample its own centers) - or a list of scalar streams `(1,)` and numbers (kept fixed). Moving centers are not sorted, used in the order given (give them ascending); with a sequence they have the signal's. A matrix, a window longer than 1 or a non-scalar list item raises `ValueError`. `function` is one function for every center or a list of one per center (paired before the centers are sorted). A membership class `Triangular` / `Rectangular` / `Gaussian` (or its name as a string) is placed on each center and sized by its neighbors as computed: moving centers take their sets with them, widths included, and the gradient reaches them through both (a step, `Rectangular`, gives none). Triangles reach the neighboring centers (lopsided for uneven centers) and sum to 1 inside the center range, rectangles reach halfway to the neighboring centers, half-open, so exactly one holds every value between the centers, Gaussians have sigma = (left gap + right gap) / 6; the outer sets reuse the one gap they have, a lone center gets 1 on both sides; a gap at 0 or below (centers that meet or cross) is held at 1e-7, no NaN. Any other nnodely layer class (made anew per center, named `<name>_<Class>_<index>`) or instance (shared) - `Triangular(radius=P)` among them - is applied to `x - center` inside the Fuzzify's Keras layer, the Parameters it reads becoming preds of the Fuzzify: `function=Sin` gives `sin(x - c)`. The memberships are put side by side as the functions give them: the output dim is the sum of their dims (`Linear(features=2)` gives 2 per center), its window theirs (a `Fir` on a window gives `T=1`); results that cannot be put side by side raise `ValueError`. So is a custom function, a `keras.layers.Layer` subclass (made anew per center) or instance (shared), wrapped in a `KerasFunction` so the model saves, loads and exports; its class is registered with Keras for this session - in another, define it (with `@keras.saving.register_keras_serializable()`) before loading. Its `call` reads one argument, `x - c`: a `call` needing more (those with a default, such as `training=None`, aside) raises `TypeError`, and so does calling a Fuzzify on more than one stream. Keras weights of a custom function are not Parameters: one Keras instance with weights shared between a Fuzzify and another layer does not load back. `bounds`: `False` (default) reads the signal as it is; `True` holds it between the lowest and the highest center (moving centers as computed, per sample), so a value beyond the centers is in the outer set instead of in none (triangles/rectangles all 0); `(left, right)` holds it between two numbers, `left <= right` - anything else raises `ValueError`. `normalize=False` (default) keeps the values of the functions; `normalize=True` divides each sample's memberships by their sum along the centers axis (per time step, per sequence step), so they add up to 1; an all-zero row stays 0, no NaN. It is meant for memberships that are not negative, not for values that can be (`Sin`, `Identity`, custom functions, `x - c`). A repeated call applies the same layers. `plot(x_range=None, points=401, ax=None)` draws the memberships with matplotlib as the layer computes them now - Parameters at their current values, the trained ones after training - over the centers and the outer gap beyond each end, on `ax` or a new figure, and returns the axes; it works on a Fuzzify not applied yet and on the stream it gives; centers computed from the inputs raise `ValueError`. |
| `KerasFunction(function, name=None)` called on `stream` or `[streams]` | shape read off the Keras layer's result | A custom function: `function` is a `keras.layers.Layer` subclass (made here, no arguments) or instance (its weights shared wherever it is used). The streams, laid out `(batch, *seq, time, *dim)` (features on the last axis), are passed to `call` as separate arguments: `call(self, x, y)` takes two. Saves, loads and exports (the class is registered with Keras for this session; in another, define it with `@keras.saving.register_keras_serializable()` before loading). Its weights are Keras weights, trained with the model but not `Parameter`s: `model.freeze()` / `model.parameters` do not reach them - set the layer's `trainable`. |
| `Triangular(radius=1.0, name=None)` | unchanged | 1 at 0, linear down to 0 at `radius` from it, 0 beyond; the signal is shifted to center it, `Triangular(radius=r)([x - c])`. `radius` is a number `>= 0`, a `(left, right)` pair, or a `Parameter` of dim `(1,)` or `(2,)` without time - a pred of the layer, trained with the model; anything else raises `ValueError`. A width at 0 or below is held at 1e-7: the set collapses, no NaN. Elementwise on any stream. |
| `Rectangular(radius=1.0, name=None)` | unchanged | 1 on `[-left, right)`, 0 elsewhere; `radius` as `Triangular`'s. A step has no gradient: a Parameter read by rectangles alone does not train (TF and torch warn "Gradients do not exist"), one read elsewhere too trains there, and the rectangles follow it [verified]. |
| `Gaussian(sigma=1.0, name=None)` | unchanged | `exp(-(x / sigma)^2 / 2)`; `sigma` a number `> 0` or a `Parameter` of dim `(1,)` without time, held at 1e-7 if trained to 0 or below. |
| `Interpolation(x_points, y_points, mode="linear", name=None)` | elementwise | `mode` is `"linear"` (piecewise) or `"polynomial"` (the unique global polynomial). Queries are clamped to `[x_min, x_max]`. Not trainable. |

### 6.3 Activations (elementwise, no weights except PReLU)
`ReLU(max_value=None, negative_slope=0.0, threshold=0.0)`, `LeakyReLU(negative_slope=0.3)`, `ELU(alpha=1.0)`, `PReLU(shared_axes=None)` (trainable: one slope per feature, shared along the window and every sequence axis, a dynamic one included [verified]; `shared_axes` counts raw tensor axes: batch 0, the sequence `1 … seq_rank`, the window, dims last), `Sigmoid()`, `Tanh()`, `Swish()`, `GELU(approximate=True)`, `Softplus()`, `Softmax(axis=None)` (the whole sample; `axis=-1` for the features, a raw tensor axis, §1.15). All accept `name=`. Called on a list of streams, an activation (or a math or trigonometric layer, `Clamp`, a membership function) is applied to each and returns a list, `Sin([a, b])` → `[sin(a), sin(b)]`; one with settings applied before it is configured, `ReLU(x)`, raises `TypeError`.

### 6.4 Math (elementwise)
`Sin Cos Tan Asin Acos Atan Exp Log Log10 Sqrt Abs Floor Ceil Deg2Rad Sign Negative` take no arguments except `name=`. Neither does `Atan2`, called on `[y, x]`: `atan2(y, x)`, the angle of the point `(x, y)` in `[-pi, pi]`, with its quadrant; `y` and `x` broadcast as the arithmetic operators do (§3), e.g. a `Constant` against a sequence. `Clamp(min=None, max=None)`: a `None` bound is unbounded. `Sum(axis=None)` sums the dim axes and keeps them with length 1 (`axis=None` sums all dim axes, an int picks one). `Floor`, `Ceil` and `Sign` have zero gradient.

### 6.5 Axis manipulation
| Signature | Effect |
|---|---|
| `Select(idx, axis=0, keep_dim=False)` | Picks one index on a dim axis, **the axis dropped**: `D=(4, 3)` → `D=(3,)`; `keep_dim=True` keeps it with length 1, `D=(1, 3)`. A stream always has a dim, so a vector gives `D=(1,)` either way. Negative idx allowed; an index out of range raises. |
| `Range(start, end, axis=0)` | Slice `[start, end)` on a dim axis. |
| `Concatenate(axis=0)([a, b, ...])` | Concatenates along a dim axis. The inputs must have the same sequence and window; a single sample (`T=1`) or a value without a window (a Parameter, a number) is repeated along those of the others; a stream without a window read from data does not join one with a window [verified]. |
| `TimeSelect(idx)` | One sample of the window, **kept with length 1** (`-1` = newest sample): `T=5` → `T=1`. Negative idx counts from the end. |
| `TimeRange(start, end)` | Slice `[start, end)` of the window, kept; `None` leaves a bound open, negative bounds count from the end. |
| `TimeConcatenate()([a, b, ...])` | Joins along the window: `T=2` + `T=3` → `T=5`; the sequence and dim must match. |
| `TimeReverse()` | Reverses the samples of the window, newest first. |
| `SeqSelect(idx, axis=0)` | One step of a sequence axis, **the axis removed**: `S=(5,), T=1` → `S=(), T=1`. Negative idx counts from the end of the data, also on a dynamic axis, so `SeqSelect(-1)` is the last step of the outermost rollout, of any length. |
| `SeqRange(start, end, axis=0)` | Slice `[start, end)` of a sequence axis; **removed when the range is one step long**. `None` leaves a bound open. On a dynamic axis a negative bound or an open end gives a dynamic axis. |
| `SeqConcatenate(axis=0)([a, b, ...])` | Joins along an existing sequence axis; the dim, the window and the other sequence axes must match. A dynamic input makes the axis dynamic. |
| `SeqStack(axis=0)([a, b, ...])` | Stacks equal-shape streams into a new sequence axis: 4 × `S=(), T=1` → `S=(4,), T=1`. `axis` is the new axis's position among the result's sequence axes: `0` the new outermost, `-1` the new innermost. To join samples into a window use `TimeConcatenate`: 3 × `S=(), T=1` → `S=(), T=3`. |
| `SeqReverse(axis=0)` | Reverses the steps of a sequence axis. |

The `Time*` layers act on the window, `T`, of which a stream has one: they take no `axis` and always keep it. The `Seq*` layers act on a sequence axis, their `axis` counted **outermost first**: `0`, the default, is the rollout of the outermost (last applied) `Loop`. On `s` with `S=(3, 5), T=1` [verified]: `SeqSelect(-1)` → `S=(5,), T=1`, `SeqSelect(0, axis=1)` → `S=(3,), T=1`, `SeqRange(1, 3)` → `S=(2, 5), T=1`, `SeqRange(1, 3, axis=1)` → `S=(3, 2), T=1`, `SeqRange(1, 2)` → `S=(5,), T=1`, `SeqStack()([s, s])` → `S=(2, 3, 5), T=1`, `SeqStack(axis=-1)([s, s])` → `S=(3, 5, 2), T=1`, `SeqConcatenate()([s, s])` → `S=(6, 5), T=1`, `TimeConcatenate()([s, s])` → `S=(3, 5), T=2`. A selected sample, `T=1`, broadcasts along the window: `w - TimeSelect(-1)(w)` subtracts the newest sample from each [verified]. `SeqSelect(-1)` on a collected trajectory gives a stream that can close a `Loop` callback on an input without that sequence axis. There is no layer that reshapes or transposes. Write a custom layer (§16) if one is needed.

### 6.6 Calculus and recurrence
See §10 (`ModelyLoop` and `Input.close`, `Buffer`, `Loop`, `Roll`, `rollback`) and §11 (`Differentiate`, `DerivativeStep`, `Derivative`, `IntegrateStep`, `Integrate`, `Ode`, `OdeNet`).

---

## 7. Modely API

`Modely(name: str, inputs: list[Input], outputs: list[Output])`
- `inputs` lists the Inputs the model reads. Inputs used only as targets need not be listed; they are discovered at build. After build, `model.train_inputs` holds every Input that training data must provide, and `model.inference_inputs` the ones inference reads.
- `outputs` is a list of `Output` nodes.

| Method | Purpose |
|---|---|
| `minimize(name, source, target=None, loss="mse", gain=1.0, seq_weights=None)` | Registers an objective (§8). Call before build. |
| `remove_minimizer(name)` | Removes an objective by name. |
| `parameters` (property) | Every `Parameter` the model computes with (relations, layer weights, called models, Loop/Roll/OdeNet bodies), in graph order: a list of the Parameter objects, also indexed by name (`model.parameters["Fir1_kernel"]`, a missing name raises `KeyError`); `print(model.parameters)` shows a table of name, seq, time, dim, number of values and trainable, with totals [verified]. Before or after build. A Parameter's current value is `.param` (the Keras variable) or `.value_numpy`; `.value` is the value it was declared with, which training does not change. A loaded model composed into a new one keeps its trained weights when the new one is built; only the new layers are initialized [verified]. |
| `all_inputs` (property) | Every Input the model reads - declared or not, inside the models it calls, or only by an objective; in a `ModelyLoop`, every Input fed back in its step - as a list of the Input objects. `print(model.all_inputs)` shows a table of name, seq, sw, dim, declared, the stream it is closed with, what the data must hold (`required`, `optional`, `target`, or `-` when only fed back) and, fed back in a ModelyLoop, what it starts from [verified]. |
| `freeze()` / `unfreeze()` → self | Freezes or releases every `Parameter` the model computes with: its relations, the weights of its layers, the models it calls and the bodies of its Loops, Rolls and OdeNets. Before or after build. The Keras weights of `PReLU`, `BatchNorm` and `KerasFunction` are not Parameters and are not frozen: set `trainable` on their Keras layer. |
| `rollback({input: stream}, steps: int, name=None)` → self | k-step feedback unrolling (§10). Call before build. |
| `build()` → self | Creates the `keras.Model` (`model.model`) and its weights. Can be called again; weights are kept. |
| `model(inputs_dict)` | Inference. Takes `{input_name: array (batch, *seq, time, *dim)}`, the batch axis always included (§1.24). Reads `model.inference_inputs`, ignores other keys, and raises `ValueError` when one is missing. Returns `{output_name: backend tensor}` for the declared outputs; convert with `keras.ops.convert_to_numpy`. |
| `model([stream, ...])` | Composition: one stream per input, in `model.inputs` order; returns the outputs as streams of an outer graph (§12). |
| `train(train_data, val_data=None, epochs=10, batch_size=1, optimizer=None, early_stopping=None, lr=1e-3, shuffle=True, optimizer_kwargs=None, early_stopping_kwargs=None, printer="legacy", jit_compile="auto")` → `dict` | Keras history (§8). |
| `validate(val_data, out_dir=None, show=False, history=None, verbose=True)` → `ValidationResult` | Metrics plus optional PNG figures (§13). `verbose=False` skips the printed summary. |
| `flatten()` → Modely | Inlines every composed sub-model; the flat model keeps the rollback and the closed Inputs. |
| `summary()` | Keras summary. |
| `export_html(out_dir, filename=None, *, open_subgraph_in_new_tab=False, physics=True)` | Interactive vis-network graph, one page per sub-model. |
| `plot(out_dir, filename=None, *, flatten=False, include_minimizers=True, format="png")` | Graphviz images in `out_dir`, the root named `filename` (the model name by default); a path or filename with an image suffix (`.png`, `.svg`, `.pdf`) names the file and format. Nodes show only their name, coloured by role (inputs green, outputs red); arrows carry the shape `(*seq, time, *dim)` (`dyn` for a dynamic sequence axis). Without `flatten`, each block (ModelCall, Loop, Roll) is one SubModel node and its body is drawn as `<filename>_<body name>`, recursively; with `flatten`, one image, each called sub-model boxed. Loop recurrences are an orange `loop` arrow, Roll and rollback a teal `roll (steps=n)` arrow. Every image has the logo and a legend (Input, Output, Parameter, Constant, SubModel, Loss, loop, roll). Without the Graphviz `dot` program it warns and writes each image's DOT source as `.gv`. Returns None. |
| `save(path, weights=True)`, `Modely.load(path, weights=True)` | nnodely format: a folder with `model.json` and `model.weights.h5`, bodies of Loop, Roll and OdeNet in sub-folders (§1.14). |
| `export_keras(path, filename=None)`, `Modely.import_keras(filename, safe_mode=True)` → `keras.Model` | Writes `path/<filename or model name>.keras`: a stand-alone copy for inference. A weight read from several Keras graphs (a body in two Loops, a Parameter inside and outside a body, a block also in a Loop, the sub-graphs of `Differentiate`) is written once per graph with the same value, since Keras cannot rebuild a layer two graphs call: the imported model gives the same outputs, but further Keras training would untie the copies - use `save`/`load` for that [verified]. The imported object is a plain Keras model; call it with batched float32 dicts. Pass `safe_mode=False` if loading refuses nested or lambda configs. |
| `export_onnx(path, filename=None, *, input_signature=None, batch_size=None, opset_version=None, verbose=False)` | Writes `path/<filename or model name>.onnx`. Needs the `onnx` extra and the TF or Torch backend (§13). |
| `Modely.validate_onnx(filename, inputs, *, return_dict=False, providers=None)` | Runs an exported ONNX file with onnxruntime. |

Properties: `built`, `model` (the training keras.Model), `inference_model` (the keras.Model inference runs), `inputs`, `outputs`, `order` (topological node list), `minimizers` (list of dicts with keys `name`, `source`, `target`, `loss`, `gain`, `seq_weights`), `train_inputs`, `inference_inputs`, `train_outputs`, `optional_inputs` (the names of the Inputs the data may leave out, reading zeros: the closed Inputs a `ModelyLoop` lists without an `init`). `print(model)` lists the nodes.

---

## 8. Objectives and training

`minimize(name, source, target=None, loss="mse", gain=1.0, seq_weights=None)`
- `source`: an `Output` (preferred), any stream, or the name of a stream of the model.
- `target` follows §1.5. Common forms:
  - `TimeSelect(-1)(y)` with `y = Input("y", sw=(n, 1))`: one-step-ahead prediction of an input that is also read by the model (through `TimeRange(0, n)(y)`).
  - `Input("y_meas", sw=1)`: a separate measured column.
  - `Input("s_target", dim=2, seq=H, sw=1)`: a trajectory target for `Loop` (the shapes must match the source).
  - `Sin()(y)`: a target computed from the data.
  - `None`: residual.
- `loss`: any Keras loss (a name such as `"mse"`, `"mae"`, `"huber"`, `"log_cosh"`), a `keras.losses.Loss` instance, a serialized config, or a callable `loss(y_true, y_pred)`.
- `gain`: the weight of the objective in the total loss, `Σ gain_i · loss_i`. Its logged and validated loss stays unweighted.
- `seq_weights`: one weight per step of the source's **outer axis** (tensor axis 1): its outermost sequence axis, the rollout of a Loop, or the window of a stream without a sequence [verified], e.g. `np.exp(0.1 * np.arange(N))`, or a callable of the number of steps for a dynamic length. Normalized to mean one; the logged loss is the weighted one.
- Shapes: a constant target (a number, a Constant) broadcasts to the source; a target read from data must have exactly the source's shape, or `train()` and `validate()` raise (e.g. a window of 3 against a one-sample source) [verified].

`train(...)`
- `optimizer`: a name (`sgd rmsprop adam adamw adagrad adadelta adamax adafactor nadam ftrl lion`; default `"adam"`), a config dict, or an instance. `lr` and `optimizer_kwargs` (e.g. `{"weight_decay": 1e-4, "global_clipnorm": 1.0}`) apply only to names; configure an instance yourself.
- `early_stopping`: the name of a logged quantity (`"loss"`, `"<minimizer name>_loss"`, or their `val_` versions with `val_data`), completed by `early_stopping_kwargs` into a `keras.callbacks.EarlyStopping`; a dict of `EarlyStopping` arguments; or any callback that sets `self.model.stop_training = True`. Typical: `early_stopping="val_loss", early_stopping_kwargs={"patience": 10, "restore_best_weights": True}`. Monitoring a quantity that is not logged raises `ValueError`; the mode defaults to `"min"` (every logged quantity is a loss).
- `printer`: `"legacy"` (default, a loss table), `"tiny"`, `"nnodely"` (animated), `None` (silent), or any `keras.callbacks.Callback`.
- Returns a history dict: `loss` (the total) and `<minimizer name>_loss` for each objective, plus their `val_` versions with `val_data` [verified: one objective named `one` gives `['loss', 'one_loss', 'val_loss', 'val_one_loss']`]. Every logged loss is the mean over the dataset's samples, batches weighed by their size (a short last batch counts for its samples only) [verified].
- Each `train` call continues from the current weights.
- Padded rollout steps (`seq_length="full"`) are masked automatically, along the outermost sequence axis (tensor axis 1).
- To freeze a composed block, call `block.freeze()` before training the outer model; `block.unfreeze()` releases it. `p.freeze()` freezes one Parameter.

Reproducibility: call `set_seed(int)` before creating layers and data. It seeds Python, NumPy and the backend, and sets `NNODELY_SEED` so child processes inherit it; two runs with the same seed give identical results on each backend. `get_seed()` returns the current seed.

---

## 9. DataLoader

`DataLoader(model, source, format=None, csv_glob="*.csv", delimiter=",", header="infer", dtype=np.float32, seq_length=None, step=1, on_short="error")`. The model must be built.

**Sources**
- A dict `{input_name: array (T,) or (T, *dim)}`, or flat `(T, prod(dim))` in row-major order; another layout of as many values raises `ValueError` rather than being reordered (axes of size 1 aside). Keys must be the model's input names, including target-only inputs. `format` is ignored for dicts.
- A `pd.DataFrame`, a `.csv` path, or a folder of CSVs matching `csv_glob`. CSV files are read with `pd.read_csv(sep=delimiter, header=header)`. `format={"input": "col" | int_index | ["c1","c2",...]}` maps inputs to columns; a list is used for `dim>1`. Unmapped columns are ignored, and an input not listed in `format` is looked up by its own name. n/a cells are kept as NaN, with a warning.
- A list of dicts or DataFrames, or a folder with several files: several **simulations**. Windows never cross simulations, and lengths may differ.
- An input of `model.optional_inputs` (a closed Input a `ModelyLoop` lists without `init`) may be left out of a dict or a DataFrame: it reads zeros [verified]. The Inputs of a `ModelyLoop` carry a dynamic horizon: pass `seq_length=H` or `"full"`.

**Windowing and alignment** [verified]
- Every input is aligned on a common current sample `t`. Within one simulation of length `T`:
  - `first = max_past - 1`, where `max_past` is the largest past over all inputs.
  - The number of samples is `T - max_future - first - max_seq_span`, where `max_seq_span = Σ(seq_len - 1)`. Sample i has current time `t_i = first + max_seq_span + i`.
  - `step=k` keeps every k-th sample.
- A sequence input, `Input("x", seq=L, sw=n)`, gives at sample i the L temporal windows ending at `t_i-L+1 … t_i`, shape `(L, n, *dim)`. **The last sequence element is aligned with the current sample of non-sequence inputs.** For example, `a` = 3 while `b` = [1,2,3] for `b = Input("b", seq=3, sw=1)`.
- An input with `sw=1` contributes its current sample, a window of one: `(N, 1, *dim)`; an input without a window the same sample without the time axis: `(N, *dim)`.
- A too-short simulation raises. `on_short="skip"` drops it.
- Dynamic `seq=-1` needs `seq_length=int`, or `seq_length="full"` to get one sample per simulation over the whole length; `"full"` resolves only the **outermost** sequence axis (`seq=(3, -1)` raises). With `"full"`, simulations are padded along it (axis 1) by repeating the last step; `loader.mask` has shape `(n_sims, max_len)` and training masks the padded steps. `step>1` is not allowed with `"full"`.

**Access**: `len(loader)`, `loader[i]` → `{name: (*seq, time, *dim)}` (no batch axis: add one with `v[None]` before calling the model), `iter(loader)`, `loader.as_dict()` → `{name: (N, *seq, time, *dim)}` (e.g. `Input("c", dim=2, seq=3, sw=5)` gives `(N, 3, 5, 2)`, a `dim=2` input with `sw=1` `(N, 1, 2)`, without a window `(N, 2)` [verified]), `loader.get_samples(n, start=0, simulation=0)` → n consecutive samples of one simulation, batched like `as_dict()`, `loader.get_input(name)`, `loader.inputs`, `print(loader)`.

**Normalization** (explicit and per loader): `loader.normalize(method="minmax" | "standard", names=None, feature_range=(-1.0, 1.0))` changes the loader in place, with statistics computed per feature, reducing over the samples, the sequence and the window. `loader.denormalize()` restores the loader. `loader.denormalize({output_name: array})` inverts predictions, matching each output to the input its minimizer target reads. `loader.normalization_stats` holds the statistics. A validation loader must be normalized with the same statistics. Normalizing each loader separately gives different statistics, so for consistent scaling prefer normalizing the raw arrays yourself, or fold the scaling into the model with `Constant`s.

---

## 10. Recurrence: rollback, Loop, Roll

| Need | Use |
|---|---|
| A model declared one step at a time: states as Inputs closed with their next value, coupled states, blocks with states of their own, every output a trajectory | `ModelyLoop` with `x.close(...)` (§10.4) |
| Train to predict k steps ahead with the model's own feedback in its input window; only the final prediction counts | `model.rollback({x: x_pred}, steps=k)` |
| Simulation-error training over whole trajectories, returning every step; exogenous inputs change per step | `Loop` (recommended) |
| Unroll a one-step model over the time window of one of its inputs, as a block inside a graph; other inputs held fixed | `Roll` |
| Continuous-time learned field integrated over reported times (neural ODE) | `OdeNet` (§11) |
| Several states integrated physically at each step | `Ode` step as the body of `Loop` |

### 10.1 `Modely.rollback(rollback: dict, steps: int, name=None)`
- Mapping `{input (obj or name): stream (obj or name)}`. The stream can be any stream of the graph, including an Output, and must have one sample of the input: its dim and sequence, `T=1` [verified]. The input needs a window (`ValueError` otherwise).
- It evaluates the model `steps` times. After each evaluation, the stream's one-sample value is appended to the input's window and the oldest sample is dropped. Model outputs are those of the **final** evaluation.
- Inputs that are not fed back, and targets, are held at their window values for every step. Align the target yourself: `data = {"x": x[:-k], "F": F[:-k], "x_ahead": x[k:]}` with target `Input("x_ahead", sw=1)` (recipe R8).
- Several feedbacks at once: `rollback({"x": "x_next", "v": "v_next"}, steps=10)`.
- Supports `save` / `load`; `flatten()` keeps it.
- A model with a rollback cannot be called as a block (`ValueError`): inlined, it would be evaluated once. For its one-step model call `Modely(name, inputs=m.inputs, outputs=m.outputs)`, declared without rollback, which shares the weights; to unroll inside a graph use `Roll` or `Loop`.

### 10.2 `Loop(f, callback, init=None, name=None, length=None, collect=True)(inputs=None)`
- `f`: the body `Modely` (built automatically if needed). Its own objectives are ignored (§1.11).
- `callback`: `{body_input: body_output}`, as objects or names. The output dim must equal the input dim, and the output must have the input's sequence (an error mentions "the sequence must be the same"). Its window must be 1: with an input window of 1 the state is replaced, with a longer one the window is shifted, and after n steps it holds only predictions [verified]. The input window must read no future samples (`future == 0`: no `sw=(p, f)` with `f > 0`) [verified]. A fed-back input without a window takes an output without one, which replaces it.
- The seeds are configured with the loop, and the loop is bound by calling it on its exogenous inputs: `Loop(f, callback, init={...})({...})`. `Loop(...)()` when there are none.
- `init`: `{body_input: stream | number}` seeding the fed-back inputs. A fed-back input missing from the dict starts at zero; a non-dict raises `TypeError`.
  - A stream with **one more sequence axis** than the body input, the rollout as its outermost sequence axis (tensor axis 1), e.g. `Input("s0", dim=2, seq=H, sw=1)`, `S=(H,), T=1`, for body input `state = Input("state", dim=2, sw=1)`, `S=(), T=1`: step 0 reads its first step [verified].
  - A stream of the shape of the body input: used directly.
  - A number, or a value without a window: broadcast. A stream without a window read from data cannot seed a body input that has one.
  - For a feedback input with a window, seed it with that window along the rollout: `{x: x0}` with `x = Input("x", sw=3)` and `x0 = Input("x0", seq=H, sw=3)` [verified].
- `inputs`: `{body_input: outer_stream}` for exogenous signals, given in the call (`Loop(...)()` when there are none). The outer stream must match the body input's dim, and either have its sequence and window (held constant) or **one more sequence axis**, outermost (tensor axis 1, consumed one step per rollout step), e.g. `Input("F", seq=H, sw=1)` for body input `force = Input("force", sw=1)`. A stream of another sequence or window raises [verified].
- Body inputs that are neither fed back nor bound are held constant, and become inputs of the outer model under their own names. A body input its outputs never read, such as the target of the body's own objective, is not an input of the loop: neither fed nor required, and binding it raises ("is not an input the body model reads") [verified].
- Horizon: the outermost-sequence length of the exogenous rollout streams (of the `init` ones only when no input carries a rollout; an `init` is read at its first step only). Pass `length=` when the exogenous lengths differ, or when no stream carries a rollout [verified]. With `seq=-1`, the rollout follows the length of the tensor given at call time (also in a TensorFlow ONNX export), so `length` is optional: without it the streams are declared with `DYNAMIC_BUILD_LENGTH` (1) steps.
- At least one source must derive from an `Input`, so that there is a batch axis.
- Timing: at step k (0-based) the body sees state `s_k` (with `s_0` from `init`) and exogenous `u_k`. Every body output at step k is computed from `s_k`, and the callback output becomes `s_{k+1}`. With `collect=True` the result gets steps `0..H-1` as a new **outermost sequence axis**, prepended: a body output `S=(), T=1, D=(2,)` with horizon H gives `S=(H,), T=1, D=(2,)`, arrays `(B, H, 1, 2)` [verified]. So the fed-back trajectory is `s_1..s_H`: with `initial` = states window `[t..t+H-1]`, the target must be states `[t+1..t+H]`, i.e. data `{"s_seq": S[:-1], "s_target": S[1:]}` (recipe R3). A non-fed-back output (for example a measurement `y(s_k)`) aligns with the unshifted window. `collect=False` returns only the last step.
- Multiple outputs: `a, b = Loop(...)` yields one stream per body output, in `f.outputs` order.
- Loops can be nested; the body of a Loop can itself contain a Loop. Each rollout is prepended: a body output carrying an inner rollout `S=(3,), T=1` gives `S=(H, 3), T=1` [verified]; close a callback on the inner trajectory's last step with `SeqSelect(-1)`.
- Supports `save` / `load`, `export_keras` and ONNX (on TensorFlow the rollout is an ONNX `Loop` node).

### 10.3 `Roll(f, callback: dict, steps=None, name=None)`
- `f` must already be built (it raises otherwise). `callback` holds exactly one pair `{input: output}`; the input needs a window, and the output must be one sample of it: the input's dim and sequence, `T=1`.
- Returns a stream shaped like the callback input, its sequence kept (`Input("x", seq=3, sw=5)` gives `S=(3,), T=5`) [verified]. Every step feeds back along the window. `steps` defaults to the window length, so every sample is a prediction. The outer model must list the body's own Input objects as its inputs, because the other inputs are fed unchanged at every step. A body input its outputs never read is ignored [verified].
- Supports `save` / `load`, `export_keras` and ONNX.

### 10.4 `ModelyLoop(name, inputs, outputs, length=None)` and `Input.close(stream, init=None)`
- A model declared **one step at a time**. A state is an Input closed with its next value, `x_next = x.close(stream)`: inside the step `x` holds the value at the end of the previous step, and `stream` the value at the end of this one. Every closed Input the outputs read - directly, through the stream another closed Input is closed with, or inside the models the step calls, one per call - is fed back by **one** rollout, a single `Loop` over `model.body`, the one-step `Modely` [verified]. Coupled dynamics are written as they are:
```python
x, v = Input("x", sw=1), Input("v", sw=1)
a = (F - k * x - c * v) / m
v_next = v.close(IntegrateStep(dt=dt)(v, a))
x_next = x.close(IntegrateStep(dt=dt)(x, v_next))
msd = ModelyLoop("msd", inputs=[F, x], outputs=[Output("x", x_next)])
```
- `inputs`: every Input the step reads that nothing closes must be listed, else `ValueError` ("which nothing closes and are not among its inputs"); a closed Input may be listed too [verified]. A model without a closed Input raises ("reads no closed Input").
- Horizon: every listed Input carries it as its **outermost sequence axis**, `Input("u", sw=1)` fed `(batch, H, 1, 1)`: `length=None` follows the data (`S=(dyn,)`), `length=N` fixes it (`S=(N,)`) [verified]. Every output is a trajectory, element k computed at step k: an output `S=(), T=1` of the step gives `S=(H,), T=1`.
- Where a closed Input starts: from its `init` (a number, a `Parameter` / `Constant`, or an `Input`, of which the first step is read; an init computed from Inputs raises); else, when **listed**, from the first window of its own data, zeros where the data leaves it out (it is in `optional_inputs`); else from zero. `total_next = total.close(total + rate)` gives `[11, 12, 13]` from data 10, and `[1, 2, 3]` left out or unlisted [verified]. Listed **and** given an `init`, it raises.
- `minimize(name, source, target, ...)` takes streams of the step - an output of the model or an Input nothing closes, objects or names - and compares their trajectories [verified]; a stream computed inside the step that is not an output raises: declare it as one.
- Data: `DataLoader(model, source, seq_length=H)` or `"full"`, the target one step ahead of the states (`x_next` at step k is the state at k+1): `{"F": F[:-1], "x": X[:-1], "x_meas": X[1:]}` with `x` listed (recipe R10).
- Blocks: a model called in the step keeps the feedback of its own closed Inputs, one state per call. A closed Input it **declares**, given a stream by the call, keeps its feedback and starts from that stream, read at the first step. A closed Input that a called model reads without declaring it, and the calling graph reads too, raises ("both in the graph that calls ... and inside it"): make it an output of the block to read the call's state, or declare an Input for it in the block and pass the caller's state [verified].
- Built as a `Modely` instead, the same graph is the one-step model: the closed Inputs are read from data at every step, and a `UserWarning` names those it does not declare [verified]. `print(model.all_inputs)` shows how every Input is fed.
- Supports `save` / `load` (§1.14), `export_keras` and ONNX on TensorFlow, a dynamic horizon included [verified]; ONNX on Torch does not keep a dynamic horizon.

### 10.5 `Buffer(sw: int, init=None, name=None)(signal)`
- The last `sw` values of a signal computed one sample per step, the current one included: `Buffer(sw=3)(u)` is `[u[k-2], u[k-1], u[k]]`, `S=(), T=3`; over `u` = 1, 2, 3, 4 the windows are `[0, 0, 1]`, `[0, 1, 2]`, `[1, 2, 3]`, `[2, 3, 4]` [verified]. Its newest sample is the value of this step, within the step; an Input closed with the signal ends at the previous step instead.
- The past `sw - 1` values are an Input closed with the signal, so it needs a `ModelyLoop`. `init`, the values before the first step: a number, a `Parameter` / `Constant` or an `Input`, one sample repeated or `sw - 1` of them, oldest first; `None` for zeros. The signal is one sample, `T=1`, or has no window.
- Typical use: the window of forces a plant reads, computed step by step by a controller: `plant([x_init, Buffer(sw=10)(F_pid)])` (examples/open_callbacks/03).

---

## 11. Physics layers

### 11.1 `Differentiate(order=1, respect_to=<Input>, name=None)(stream)`, `DerivativeStep(order=1, dt=<float>, window=None, poly_order=None, name=None)(init, stream)` (or `(stream)`) and `Derivative(order=1, dt=<float>, init=None, window=None, poly_order=None, name=None)(stream)`
Orders 1 and 2 are supported by all three. Passing a number to `Differentiate` raises `TypeError` pointing to `DerivativeStep` and `Derivative`; both require a positive `dt`.
- **`Differentiate`, with respect to an Input** (autodiff, exact): the result is `d(Σ stream)/d(input window)` (a VJP), shaped like the window of the Input the stream reads. It includes the trainable weights, can be nested, and can be trained through. It takes no `init`, `window` or `poly_order`. The stream must depend on that Input. Typical use is a PINN residual: `minimize("physics", Differentiate(respect_to=t)(u) + u)`. For `u` with dim>1 the result is the gradient of the sum of the components; take `Select` first to get one component. ONNX export: TensorFlow only, with batch fixed to 1. Torch refuses, and JAX cannot export.
- **`DerivativeStep`, with respect to time** (`dt: float`, as in `IntegrateStep`): a causal finite difference along the stream's window that **keeps the window length**; the sequence is kept. A one-sample window is one step, `(x - init)/dt`.
  - `init`, **the first argument of the call**, `DerivativeStep(dt=dt)(init, x)`, is the sample(s) just before the window: a stream of 1 sample (reused for each) or `window-1` samples, its dim a scalar or the stream's, its sequence none or the stream's [verified]; or a number. Called on `x` alone, it is zero.
  - `window` defaults to `order+1`, the plain backward difference. Larger values fit a polynomial by least squares (Savitzky–Golay style): they smooth noise at the cost of about `(window-1)/2` samples of delay.
  - `poly_order` ranges over `order..window-1` and defaults to `order`.
  - With the default `window` and `order=1` it is the exact inverse of `IntegrateStep(solver="euler")` with the same `init`: `IntegrateStep(dt=dt)(x0, DerivativeStep(dt=dt)(x0, x))` is `x`; a longer, smoothing `window` is not inverted.
- **`Derivative`, with respect to time, one step at a time in a `ModelyLoop`** (§10.4): `DerivativeStep`'s stencil across the steps. The signal is one sample of a step, `T=1` and no sequence (`x = Input("x", sw=1)`); the `window - 1` samples before it are an Input closed with the signal, which the `ModelyLoop` feeds back with its other closed Inputs - no `Loop` of its own. The result, `S=(), T=1`, is the derivative at this step [verified]. A signal with a sequence ("carries a sequence") or a longer window raises [verified]. The same samples give the same values as `DerivativeStep` on a window. Built as a `Modely`, the past samples are read from data, with a warning.
  - `init` is the value just before the first step, as `Input.close` takes it: one sample (reused for every sample of history a longer stencil reads) or the `window - 1` samples it reads, oldest first, along its window; its dim a scalar or the signal's; a number, a `Parameter` / `Constant`, an `Input` (its first step), or `None` (zero).
  - In a `ModelyLoop`, `Integrate(dt=dt, init=x0)(Derivative(dt=dt, init=x0)(x))` is `x` (the default first-order stencil) [verified]. `DerivativeStep` on one sample per step is not this derivative: each step is a one-sample window, `(x_k - init)/dt`.

### 11.2 `IntegrateStep(solver="euler", dt=<required float>, name=None)(init, rate)` (or `(rate)`)
- `solver` is `"euler"` / `"rectangular"` (y[i] = y[i-1] + dt·r[i]) or `"trapezoidal"`. It integrates along the window and preserves its length. `init`, **the first argument of the call**, is the value just before the window: a stream of one sample (dim a scalar or the rate's, sequence none or the rate's) [verified]; or a number or another value without a window. Called on the rate alone it starts from 0. A stream without a window read from data raises.
- On a one-sample window it is exactly one state update: `IntegrateStep(dt=dt)(v, a)` = `v + dt*a`.
- On an n-sample window it returns the whole running integral in one pass, with no rollout needed: `x = IntegrateStep(dt=dt)(IntegrateStep(dt=dt)(a))` with `a = Input("a", sw=n)`.
- The rate needs a window; a stream without one raises.
- There is no rk4 or heun here; use `Ode` for those.

### 11.2b `Integrate(solver="euler", dt=<required float>, init=None, name=None)(rate)`: one step at a time, in a `ModelyLoop`
- `IntegrateStep`'s rules across the steps of a `ModelyLoop` (§10.4): the rate is one sample of a step, `T=1` and no sequence (`acc = Input("acc", sw=1)`); the integral is an Input closed with its update, fed back by the `ModelyLoop` with its other closed Inputs - no `Loop` of its own. The result, `S=(), T=1`, is the value at the end of the step: `y[k] = y[k-1] + dt*r[k]` (`"euler"`, `"rectangular"`) or `y[k-1] + dt/2*(r[k-1] + r[k])` (`"trapezoidal"`, the first step a rectangle) [verified].
- A rate with a sequence raises ("this rate carries a sequence"), and one with a longer window raises and points to `IntegrateStep` [verified]. Built as a `Modely`, the integral is read from data, with a warning.
- `init`, the value just before the first step, as `Input.close` takes it: a number, a `Parameter` / `Constant`, or an `Input` of one sample (the rate's dim, or a scalar for a one-axis dim), whose first step is read; `None` is 0.
- Chains: `position = Integrate(dt=dt, init=0.0)(Integrate(dt=dt, init=v0)(acc))` in `ModelyLoop(..., inputs=[acc, v0], ...)`, `v0` read at its first step.

### 11.3 `Ode(f, states, dt, method="rk4", args=())` (a function, not a class)
- Advances the states by **one** explicit step. `method` is one of `euler`, `midpoint`, `heun`, `rk4`.
- `f(*state_streams, *args)` must return one derivative stream per state, in order (a tuple or list, or a single stream when there is one state).
- `states` is a list of streams (typically `x`) or a single stream. `dt` is a number or a stream (it can be a `Constant`, a `Parameter`, or read from data).
- Returns a list of next-state streams (a single stream when a single state was given).
- `f` is evaluated once per RK stage, building new nodes each time. Keep `Parameter`s and weight-bearing layers **outside** `f` (§1.10).
- For a trajectory, wrap the `Ode` step in a body `Modely` and roll it out with `Loop`, or use `rollback`.

### 11.4 `OdeNet(f, states, t, initial=None, method="rk4", steps=1, event=None, reset=None, rtol=1e-6, atol=1e-8, max_steps=1000, name=None)`
- `f` is a built `Modely` giving the vector field `dx/dt`. It must be **autonomous**: every body input the field reads is a state, and exogenous drivers are rejected; an input it never reads, such as the target of its own objective, is ignored [verified].
- `states`: `{state_input: derivative_output}`. The derivative must match the state's dim and time.
- `t`: a stream of report times whose **outermost sequence axis** has a static count N ≥ 2, e.g. `Input("t", seq=N, sw=1)` with data `(batch, N, 1, 1)` [verified]. The times are **shared across the batch** (the first row is used), and `t[0]` is the initial condition.
- `initial`: `{state: outer_stream}` with exactly the state's dim, sequence and window (no extra sequence axis) [verified]. By default the body's own state Inputs are used, so the outer model lists them as inputs.
- Output: the points as a new outermost sequence axis, `(N, *state_seq, state_time, *dim)` per state: state `z` with `dim=2` gives `S=(N,), T=1, D=(2,)`, arrays `(B, N, 1, 2)` [verified], with index 0 equal to the initial state. `a, b = OdeNet(...)` unpacks several states.
- `method`: a fixed tableau (`euler`, `midpoint`, `heun`, `rk4`) applies `steps` substeps per reported interval, unrolled; use it for training and export. `"dopri5"` is adaptive (controlled by `rtol`, `atol`, `max_steps`): not ONNX-exportable, not reverse-differentiable on JAX, meant for inference. To switch a built model: `model.model.get_layer(<OdeNet name>).set_method("dopri5")`, which is why `name=` should be set.
- Hybrid events: `event` names an output of `f` with dim `(1,)` and one sample or no window, positive before the event and negative after. `reset` gives `{state: output}` with the post-event values of every state. Events need a fixed tableau, and the event time is differentiable.
- **Training with a DataLoader** [verified, recipe R5]: the state inputs have no sequence, so the loader aligns them with the **last** element of the `t` and target sequences (§9). Shift the seed column by N-1 so that it equals the first point of each window: `seed = np.vstack([np.repeat(ref[:1], N-1, axis=0), ref[:1-N]])`. See recipe R5.
- Supports `save` / `load` (the method chosen with `set_method` included) and `export_keras`.

---

## 12. Model composition

- `block = Modely("b", inputs=[a, c], outputs=[o1, o2]).build()`. Then `o1s, o2s = block([stream_for_a, stream_for_c])`. The streams go in the order of `block.inputs`, exactly one per input (another count raises `ValueError`), and a single output returns a single stream.
- A block keeps its shapes: each stream must have the shape of the block's input - the same dim, sequence and window; a dynamic axis matches any length - else `ValueError` ("takes a stream of that shape"). Nothing is reshaped, so a block with input `Input("x", sw=1)` refuses `Input("a", seq=3, sw=1)` or `Input("a", sw=3)` [verified]. An `Input` of that shape is passed as it is, `block([a])` [verified].
- A model loaded with `Modely.load` can be used as a block: building the larger model keeps its trained weights and initializes only the new layers [verified].
- The block brings its weights along (trained weights are reused while input shapes match), and training the outer model also trains the block. To freeze it: `block.freeze()`.
- Calling the same block twice shares its weights (§1.22 covers naming).
- `EquationLearner` uses composition internally.
- `Differentiate(respect_to=Input)` inside a block does not work once the block's inputs are rebound. Put the derivative in the outer model instead, where the Input is declared.
- An Input the block reads **without declaring it** is the outer model's Input too, one node read from the same data, whatever the order of the operands [verified].
- A model with a `rollback` cannot be a block (§10.1). A closed Input the block declares reads the stream given for it at every step in a `Modely`; in a `ModelyLoop` it keeps its feedback, and the stream is its initial value (§10.4).
- `model.flatten()` returns an equivalent model with everything inlined, its rollback and closed Inputs kept.

---

## 13. Inference, validation and export

**Inference arrays**: float32 NumPy, shape `(batch, *seq, time, *dim)` as each Input is declared (`x.shape` is the per-sample shape); `(batch, *dim)` for an Input without a window. Results come back laid out the same. A single sample keeps a batch axis of 1.

**`validate(val_data, out_dir=None, show=False, history=None, verbose=True)`**
- Scores every objective as training does: same forward pass (`training=False`), same target alignment, same padding mask. `metrics["loss"]` equals `train()`'s `val_<name>_loss` on the same data (`seq_weights` and masking included, `gain` not applied) [verified]. Prints a summary (`verbose=False` skips it).
- Metrics per objective (`result[name].metrics`, or `result.metrics()` for everything): `loss rmse mae max_error bias std_error nrmse_pct fit_pct r2 vaf_pct correlation non_finite`.
  - With e = ŷ − y: `fit_pct` = 100(1 − ‖e‖/‖y−ȳ‖), `r2` = 1 − Σe²/Σ(y−ȳ)², `vaf_pct` = 100(1 − var(e)/var(y)), `nrmse_pct` = 100·RMSE/range(y); `rmse² = bias² + std_error²`.
  - Per channel (each dim element of the signal): `result[name].channel_metrics[key]` is an array, one value per channel. The whole-signal absolute metrics pool every channel; the relative ones are the mean over channels [verified].
  - Relative metrics are NaN for a channel whose target is constant, and that channel is left out of the mean. Correlation is 0 for a constant prediction. Any NaN/inf error → `non_finite` count and NaN metrics.
  - Padded rollout steps (`seq_length="full"`, simulations of different lengths) are left out of every metric: the padded set scores the same as each simulation alone [verified]. `score.points` / `score.padded` count the scored and the left-out points. The loss keeps training's definition (padded steps add 0 but count in the mean).
  - `score.step_rmse`: `(steps, channels)` RMSE along the outermost sequence axis for a signal with a sequence (a rollout), else `None`.
- `out_dir` writes PNGs (prediction vs target over the scored points, error, parity, histogram, RMSE per rollout step when there is one, plus loss curves when `history` is given), and `result.figures` lists them.

**Export support**

| Model contains | `save`/`load` | `export_keras` | ONNX TF | ONNX Torch | ONNX JAX |
|---|---|---|---|---|---|
| Plain layers, `Parameter`, `Constant`, `Fir`, `Linear`, `LocalModel`, `EquationLearner`, `IntegrateStep`, `DerivativeStep`, `rollback` | yes | yes | yes | yes | no |
| `ModelyLoop` (closed Inputs, `Integrate`, `Derivative`, `Buffer`) | yes [verified] | yes [verified] | yes, dynamic horizon [verified] | not with a dynamic horizon | no |
| `Differentiate(respect_to=Input)` | yes | yes | yes (batch 1) | no | no |
| `Loop`, `Roll` | yes | yes [verified] | yes [verified for Loop] | yes (per docs) | no |
| `OdeNet` (fixed tableau) | yes | yes | yes (per docs) | yes (per docs) | no |
| `OdeNet(method="dopri5")` | yes (`set_method` included) | yes | no | no | no |
| A weight read from several graphs (body in two Loops, Parameter inside and outside a body, block also in a Loop, several `Differentiate`) | yes, shared | yes, as equal copies | yes | yes | no |

JAX cannot export ONNX at all: jax2tf emits an unconvertible XlaCallModule. `export_onnx(batch_size=n)` fixes the batch axis.

---

## 14. Common errors and fixes

| Error (substring) | Cause | Fix |
|---|---|---|
| `Model X is not built` / `Model is not built` | `DataLoader` / `train` / call before `build()` | Call `model.build()` first. |
| `nnodely runs on Keras, whose backend needs '...'` | The backend chosen by `KERAS_BACKEND` is not installed | `pip install "nnodely[<backend>]"`. |
| `reads the inputs [...], which are missing from the data it was called with` | An input of `model.inference_inputs` is not in the dict (§1.12) | Add it. |
| `expected shape=(None, ...), found shape=(...)` at inference | An array laid out in the old `(batch, *dim, time)` order, or without its batch axis, e.g. `loader[i]` (§1.24; a missing batch axis may also fail inside a layer, or be misread) | Lay it out `(batch, *seq, time, *dim)`; add a batch axis: `v[None]`. |
| `used as a block it takes one stream for each, in that order` | A block called with fewer or more streams than its inputs | Pass one stream per `block.inputs`. |
| `so used as a block it takes a stream of that shape` | A block called on a stream of another window, sequence or dim | Pass a stream of the shape the block reads (or the Input whole), or declare the block on the shape it is used with. |
| `... is missing required inputs: [...]` (DataLoader) | Data dict lacks an input, often a target-only one | Add the column, or map it with `format`. |
| `has only N samples, but the model requires at least M` | Simulation shorter than windows plus sequences | Shorter windows or seq, more data, or `on_short="skip"`. |
| `Some inputs have undefined sequence length` | `seq=-1` without `seq_length` | `DataLoader(..., seq_length=H)` or `"full"`. |
| `seq_length='full' only resolves the outermost one` | A dynamic length that is not the outermost sequence axis, e.g. `seq=(3, -1)` | Make the dynamic axis outermost, or pass `seq_length=H`. |
| `roll stream ... must produce one temporal sample with shape ...` | Rollback stream has a window > 1, a wrong dim, or another sequence | Feed back a one-sample stream with the input's dim and sequence. |
| `Loop cannot determine the rollout length` | No initial value or input has one more sequence axis than its body input | Pass `length=H`, or feed a rollout (`Input("x", seq=H, sw=1)`). |
| `Loop rollout inputs declare different lengths` | Exogenous rollout streams with different rollout lengths | Pass `length=` or make the lengths equal. |
| `Loop input '...' expects the sequence and window ..., or one more sequence axis for the rollout` | An exogenous stream whose shape is neither the body input's nor that with one sequence axis more | Declare it like the body input, with `seq=H` for a rollout. |
| `Model '...' reads no Input` at build | The model reads no Input, e.g. a Loop seeded and driven only by numbers | Seed or drive it from an `Input`. |
| `Loop init must be a dict` | Scalar or stream passed as `init` | `Loop(..., init={inp1: s1, inp2: s2})(...)`. |
| `the output must be a single sample, the next one of the input` | Loop callback output with a window longer than 1 | Feed back one sample, e.g. `TimeSelect(-1)(...)` of the window. |
| `the input window reads ... future samples` | Loop callback input declared with future samples (`sw=(p, f)`, `f > 0`) | Declare the fed-back input only up to its current sample. |
| `the sequence must be the same` | Loop callback output with another sequence than its input, e.g. a collected inner trajectory | Close the callback on `SeqSelect(-1)(...)`, or make the sequences equal. |
| `has two different outputs named ...` | A block called twice exposes same-named outputs | Wrap each call's output in a distinctly named `Output`. |
| `the differentiated relation does not depend on input ...` | `Differentiate` w.r.t. an Input inside a composed block, or no real dependency | Move the derivative to the model that declares the Input. |
| `Minimizer '...' compares '...' of shape ... with '...' of shape ...` in `train` / `validate` | The source and target of an objective have different shapes, e.g. a window target against a one-sample source | Make them match: use a window of the source's length as target. |
| `streams with windows of ... and ... cannot be combined` | Arithmetic between different windows, e.g. `Input("a", sw=3) + Input("b", sw=2)` | Select to the same window (`TimeRange`), or use a single sample, `T=1`, which broadcasts. |
| `a stream with no time ... cannot be combined with a stream with a window` | An Input declared without `sw` (or a relation of it) combined with a windowed stream | Declare it with `sw=1`; only values (numbers, Parameters, Constants) without a window combine with anything. |
| `streams with a different sequence cannot be combined` | Different sequences, e.g. `Input("s", seq=4, sw=1)` with `Input("q", seq=3, sw=1)` | Make the sequences equal, or select a single step. |
| `... acts on the window of its stream, and ... has none` | `Fir`, a `Time*` layer, `DerivativeStep`, `IntegrateStep` on a stream without a window | Declare the Input with `sw`. |
| `a sequence needs a window` | `Input(..., seq=N)` without `sw` | Add `sw=1` (one sample per step) or the window. |
| `sw is the window, ...` | `sw=0`, a negative count, or a pair that is not `(past, future)` | `sw=n` with `n >= 1`, or `sw=(past, future)` with at least one sample. |
| `dim ... and dim ... cannot be combined` | Dims neither equal nor one a scalar, e.g. `(3,)` with `(2, 3)` | Make them equal, or reduce one to a scalar. |
| `cannot be combined: each would be broadcast` | A one-sample non-scalar (e.g. a `(2, 3)` Constant) with the window of a scalar, or a sequence of one sample with a window without the sequence | Give the one-sample operand the window's time, or the window the operand's dim. |
| `idx ... out of bounds for dim axis` (`the window`, `seq axis`) | `Select` (`TimeSelect`, `SeqSelect`) index out of range | Use an index within the axis size. |
| `TypeError: bad operand type for unary -` | `-stream` | `-1.0 * stream` or `Negative()(stream)`. |
| `OdeNet integrates an autonomous field, but [...] are body inputs that are not states` | Exogenous input in the field | Use `Ode` + `Loop` for driven systems. |
| `which nothing closes and are not among its inputs` | A `ModelyLoop` step reads an Input that is neither listed nor closed | List it in `inputs`, or close it with `x.close(stream)`. |
| `reads no closed Input, so it has no feedback` | A `ModelyLoop` without a closed Input | Close an Input, or build a `Modely`. |
| `has an initial value ... and is among the inputs` | A closed Input listed in a `ModelyLoop` and given an `init` (or a stream by a call) | Leave it out of `inputs`, or give it no `init`. |
| `is closed with the value of one step` | `x.close(stream)` with a stream that is not one sample of `x` | Close it with one sample: `T=1`, `x`'s dim and sequence (`TimeSelect(-1)(...)` of a window). |
| `closed, it holds values already computed` / `is already closed with` | Closing an Input that reads future samples, or closing it twice | Declare it up to its current sample; close it once. |
| `reads the closed Input ... both in the graph that calls ... and inside it` | A closed Input read by a block that does not declare it, and by the model calling the block | Make it an output of the block, or declare an Input for it in the block and pass the state in the call. |
| `reads the closed Inputs [...], which only a ModelyLoop feeds back` (UserWarning at build) | A `Modely` reads closed Inputs it does not declare: they are read from data | Build it with `ModelyLoop`, or declare them. |
| `of one step of a ModelyLoop, and this rate carries a sequence` (or `this signal`) | `Integrate` / `Derivative` on a stream with a sequence: the old horizon form | Declare the rate `Input("r", sw=1)`, no `seq`, and build with `ModelyLoop`; for a window use `IntegrateStep` / `DerivativeStep`. |
| `has a rollback, which a block cannot carry` | A model with `rollback` called as a block | Call a `Modely` declared on its inputs and outputs without rollback (same weights), or use `Roll` / `Loop`. |
| `was never called on a stream` | A layer object used as a stream: `b + Exp()`, `Output("y", Sin())` | Call it on its streams: `Exp()(b)`. |
| `reads one stream, got 2` | `Fir` / `Linear` given several streams | Apply one to each, or join them with `Concatenate` / `TimeConcatenate`. |
| `is laid out (...) per timestep, but its dim=(...)` (DataLoader) | An array whose features are laid out neither as `dim` nor flat | Lay it out `(T, *dim)`, or flat `(T, prod(dim))` in row-major order. |
| `is not an input the body model reads` | A `Loop` bound on a body input its outputs never read | Leave it out of the call. |
| `is saved without the weights of [...]` (UserWarning at save) | A trained weight read only by the stream a closed Input is closed with, which a `Modely` does not compute | Save the `ModelyLoop` that computes it, or declare that stream among the outputs. |
| `calls models that declare the closed Inputs [...]` (UserWarning at save) | A call gives a stream for a closed Input its model declares: saved, the feedback is gone | Save the `ModelyLoop` that closes them. |
| Softmax output sums to one over the whole sample, not per feature vector | The default normalizes the whole sample (§1.15) | `Softmax(axis=-1)`. |

---

## 15. Recipes (all run as written; TensorFlow backend)

### R1. One-step ARX / FIR identification with train/val split
```python
import numpy as np
from nnodely import DataLoader, Fir, Input, Modely, Output, TimeRange, TimeSelect, set_seed
set_seed(0)
rng = np.random.default_rng(0)
N = 2000
u_data = rng.uniform(-1, 1, N); y_data = np.zeros(N)
for t in range(N - 1):
    y_data[t + 1] = 0.9 * y_data[t] - 0.2 * y_data[t - 1] + 0.5 * u_data[t]

y, u = Input("y", sw=(2, 1)), Input("u", sw=2)        # y: two past samples and the next one
y_next = Output("y_next", Fir(features=1)([TimeRange(0, 2)(y)]) + Fir(features=1)([u]))
model = Modely("arx", inputs=[y, u], outputs=[y_next])
model.minimize("one_step", y_next, TimeSelect(-1)(y), loss="mse")
model.build()

signals = {"y": y_data, "u": u_data}
split = int(0.8 * N)
train = DataLoader(model, source={k: v[:split] for k, v in signals.items()})
val = DataLoader(model, source={k: v[split:] for k, v in signals.items()})
history = model.train(train, val_data=val, epochs=30, batch_size=64, lr=1e-2, printer=None)
result = model.validate(val, history=history)        # add out_dir="plots" for PNGs
print(result["one_step"].metrics["fit_pct"])
pred = model({k: v[None] for k, v in val[0].items()})["y_next"]   # shape (1, 1, 1)
```

### R2. Gray-box parameter identification with an RK4 step (one-step objective)
```python
from nnodely import Input, Modely, Ode, Output, Parameter, Sin, DataLoader, TimeSelect
dt = 0.02
# the current sample and the next one of each state, read as one window
theta, omega, torque = Input("theta", sw=(1, 1)), Input("omega", sw=(1, 1)), Input("torque", sw=1)
g_over_l = Parameter("g_over_l", value=10.0)     # initial guesses; defined OUTSIDE f
damping = Parameter("damping", value=0.0)

def pendulum(th_, om_, tq_):                     # returns d(theta)/dt, d(omega)/dt
    return om_, tq_ - g_over_l * Sin()(th_) - damping * om_

theta_next, omega_next = Ode(pendulum, [TimeSelect(0)(theta), TimeSelect(0)(omega)], dt=dt,
                             method="rk4", args=(torque,))
theta_out, omega_out = Output("theta_next", theta_next), Output("omega_next", omega_next)
pend = Modely("pendulum", inputs=[theta, omega, torque], outputs=[theta_out, omega_out])
pend.minimize("theta_err", theta_out, TimeSelect(-1)(theta))
pend.minimize("omega_err", omega_out, TimeSelect(-1)(omega))
pend.build()
data = DataLoader(pend, source={"theta": th, "omega": om, "torque": tq})   # 1-D numpy arrays
pend.train(data, epochs=40, batch_size=128, lr=5e-2, printer=None)
print(g_over_l.value_numpy, damping.value_numpy)
```

### R3. Multi-step (simulation-error) training with Loop
```python
from nnodely import Input, Linear, Loop, Modely, Output, DataLoader
H = 20
# 1) body: one step, no objectives
state, force = Input("state", dim=2, sw=1), Input("force", sw=1)
step_out = Output("state_next",
    Linear(features=2, bias=False)([state])
    + Linear(features=2, bias=False)([force]))
body = Modely("step", inputs=[state, force], outputs=[step_out]).build()
# 2) outer: sequences of H steps
s_seq, f_seq = Input("s_seq", dim=2, seq=H, sw=1), Input("f_seq", seq=H, sw=1)
traj = Output("traj", Loop(f=body, callback={state: step_out}, init={state: s_seq})(
                           {force: f_seq}))      # traj: S=(H,), T=1, D=(2,)
sim = Modely("simulator", inputs=[s_seq, f_seq], outputs=[traj])
sim.minimize("sim_error", traj, Input("s_target", dim=2, seq=H, sw=1))
sim.build()
# 3) data: target is the state shifted one step ahead of the seed window
states = np.stack([th, om], axis=-1)             # (T, 2)
data = DataLoader(sim, source={"s_seq": states[:-1], "f_seq": tq[:-1], "s_target": states[1:]}, step=5)
sim.train(data, epochs=5, batch_size=32, lr=1e-2, printer=None)
# body weights are trained too; body(...) can now be used for one-step prediction
```
To use a physical step inside the body, build the body from `Ode(...)` as in R2, feeding back both states (`callback={"theta": "theta_next", "omega": "omega_next"}`, `init={"theta": th_seq, "omega": om_seq}` with `th_seq = Input("th_seq", seq=H, sw=1)`). For whole simulations of different lengths use Inputs with `seq=-1, sw=1`, `Loop(...)` and `DataLoader(..., seq_length="full")`.

### R4. Physics-informed network (Differentiate w.r.t. an Input + residual)
```python
from nnodely import Differentiate, Input, Linear, Modely, Output, Tanh, DataLoader
t_in = Input("t")                                  # no window: a static coordinate
u_hat = Linear(features=1)([Tanh()([Linear(features=16)([t_in])])])
du_dt = Differentiate(order=1, respect_to=t_in)(u_hat)
u_out = Output("u", u_hat)
pinn = Modely("decay", inputs=[t_in], outputs=[u_out])
pinn.minimize("data", u_out, Input("u_meas"))
pinn.minimize("physics", du_dt + u_hat)             # enforce du/dt = -u
pinn.build()
times = np.linspace(0, 2, 200)
data = DataLoader(pinn, source={"t": times, "u_meas": np.exp(-times)})
pinn.train(data, epochs=20, batch_size=32, lr=1e-2, printer=None)
pinn.validate(data)                                # scores the residual too
```

### R5. Neural ODE with OdeNet (including DataLoader training alignment)
```python
from nnodely import Input, Linear, Modely, OdeNet, Output, Tanh, DataLoader
N = 25
z = Input("z", dim=2, sw=1)
field = Modely("field", inputs=[z], outputs=[Output("dz",
    Linear(features=2)([Tanh()([Linear(features=32)([z])])]))]).build()
t_rep = Input("t_rep", seq=N, sw=1)
z_traj = Output("z_traj", OdeNet(f=field, states={z: "dz"}, t=t_rep, method="rk4", steps=2, name="node"))
node = Modely("neural_ode", inputs=[z, t_rep], outputs=[z_traj])           # z_traj: S=(N,), T=1, D=(2,)
node.minimize("traj", z_traj, Input("z_ref", dim=2, seq=N, sw=1))
node.build()
# data: ref (T, 2) sampled at times (T,). The seed must be the FIRST point of each window:
seed = np.vstack([np.repeat(ref[:1], N - 1, axis=0), ref[:1 - N]])
data = DataLoader(node, source={"z": seed, "t_rep": times, "z_ref": ref})
node.train(data, epochs=100, batch_size=64, lr=1e-2, printer=None)
# inference over any horizon with the same N points; the target is not needed
out = node({"z": np.ones((1, 1, 2)), "t_rep": np.linspace(0, 1, N).reshape(1, N, 1, 1)})["z_traj"]  # (1, N, 1, 2)
node.model.get_layer("node").set_method("dopri5")               # adaptive for inference
```

### R6. Gain scheduling with Fuzzify + LocalModel
```python
from nnodely import Fir, Fuzzify, Input, LocalModel, Modely, Output, Triangular
trq, gear = Input("trq", sw=10), Input("gear", sw=1)
mu = Fuzzify(centers=[1.0, 2.0, 3.0, 4.0], function="Triangular")([gear])   # S=(), T=1, D=(4,)
engine = LocalModel(lambda i: Fir(features=1))([trq], [mu])                     # S=(), T=1, D=(1,)
m = Modely("sched", inputs=[trq, gear], outputs=[Output("force", engine)]).build()
m({"trq": np.ones((1, 10, 1)), "gear": np.full((1, 1, 1), 2.5)})                 # (batch, time, dim)
# one 10-tap FIR per gear cell, each with weights of its own
```

### R7. Composition (reuse a trained block)
```python
from nnodely import Fir, Input, Linear, Modely, Output, ReLU
sig = Input("sig", sw=10)
smoother = Modely("smoother", inputs=[sig], outputs=[Output("smooth", Fir(features=1)([sig]))]).build()
raw = Input("raw", sw=10)
cmd = Output("cmd", ReLU()([Linear(features=1)([smoother([raw])])]))
controller = Modely("controller", inputs=[raw], outputs=[cmd]).build()
controller({"raw": np.ones((1, 10, 1))})
```

### R8. k-step-ahead training with rollback
```python
from nnodely import Fir, Input, Modely, Output, DataLoader
K = 5
x, F = Input("x", sw=5), Input("F", sw=1)
x_pred = Output("x_pred", Fir(features=1)([x]) + Fir(features=1)([F]))
rb = Modely("rb", inputs=[x, F], outputs=[x_pred])
rb.rollback({x: x_pred}, steps=K)
rb.minimize("ahead", x_pred, Input("x_ahead", sw=1))
rb.build()
data = DataLoader(rb, source={"x": xs[:-K], "F": Fs[:-K], "x_ahead": xs[K:]})
rb.train(data, epochs=20, batch_size=64, lr=1e-2, printer=None)
```

### R9. Mechanical model: acceleration → velocity → position in one pass (no rollout)
```python
from nnodely import Input, IntegrateStep, Linear, Modely, Output
dt, n = 0.01, 50
F, v0, x0 = Input("F", sw=n), Input("v0", sw=1), Input("x0", sw=1)
acc = Linear(features=1)([F])                         # S=(), T=n, D=(1,): per-sample force -> acceleration
vel = IntegrateStep(dt=dt)(v0, acc)                    # S=(), T=n, D=(1,)
pos = IntegrateStep(dt=dt)(x0, vel)                    # S=(), T=n, D=(1,)
m = Modely("mech", inputs=[F, v0, x0], outputs=[Output("pos", pos)])
m.minimize("pos_err", m.outputs[0], Input("x_meas", sw=n))
m.build()
```
Caution: `v0` and `x0` are their current samples, at the **end** of the window of `F`. Provide their data columns shifted by `n` so that they hold the state just before the window.

### R10. Mass-spring-damper declared one step at a time (ModelyLoop, identification on trajectories)
```python
import numpy as np
from nnodely import DataLoader, Input, IntegrateStep, ModelyLoop, Output, Parameter
dt, m = 0.05, 1.0
rng = np.random.default_rng(0)
F_data = np.repeat(rng.uniform(-1.0, 1.0, 60), 50)        # 3000 samples
X, V = np.zeros(3000), np.zeros(3000)
for i in range(2999):                                      # true k = 2, c = 0.5
    V[i + 1] = V[i] + dt * (F_data[i] - 2.0 * X[i] - 0.5 * V[i]) / m
    X[i + 1] = X[i] + dt * V[i + 1]

F, x, v = Input("F", sw=1), Input("x", sw=1), Input("v", sw=1)
k, c = Parameter("k", value=1.0), Parameter("c", value=1.0)
v_next = v.close(IntegrateStep(dt=dt)(v, (F - k * x - c * v) / m))
x_next = x.close(IntegrateStep(dt=dt)(x, v_next))
msd = ModelyLoop("msd", inputs=[F, x, v], outputs=[Output("x_next", x_next)])   # x, v listed: start from data
msd.minimize("x", "x_next", Input("x_meas", sw=1))        # x_next at step k is the state at k+1
msd.build()
data = {"F": F_data[:-1], "x": X[:-1], "v": V[:-1], "x_meas": X[1:]}
msd.train(DataLoader(msd, source=data, seq_length=50, step=10), epochs=30, batch_size=32, lr=5e-2, printer=None)
print(k.value_numpy, c.value_numpy)                         # about [2.] [0.5] [verified]
```
The same graph built as `Modely("msd_step", inputs=[F, x, v], outputs=[...])` is the one-step model, reading `x` and `v` from data.

---

## 16. Custom layers

Pattern: a symbolic `Layer` subclass returns a serializable Keras layer from `build_layer()`. The output shape is inferred automatically by running the Keras layer on zeros laid out `(1, *seq, time, *dim)` (`(1, *seq, *dim)` for an input without a window): the result keeps the inputs' number of sequence axes (at most), then a window if an input has one, and the remaining axes are its dim; a dynamic sequence axis (`seq=-1`) is probed with one step and declared dynamic again when the layer keeps the number of sequence axes. Override `output_shape(self, *inputs) -> Shape` (`nnodely.core.shape.Shape(dim=..., time=..., seq=...)`) only when that cannot work (a layer that adds or removes sequence axes, value-dependent shapes). The saved config is derived from the arguments passed to `super().__init__`, so `get_config` needs no override unless an argument is not JSON-serializable as given (`Fir` and `Linear` save their weights as the Parameter nodes they are, so their config only says whether there is a bias).
```python
import keras
from nnodely.core.layer import Layer

@keras.saving.register_keras_serializable(package="my_project")
class SquashImpl(keras.layers.Layer):
    def __init__(self, gain=1.0, **kwargs):
        super().__init__(**kwargs); self.gain = float(gain)
    def call(self, x):                      # x: [batch, *seq, time, *dim]; list of tensors if multi-input
        return keras.ops.tanh(self.gain * x)
    def get_config(self):
        return {**super().get_config(), "gain": self.gain}

class Squash(Layer):
    def __init__(self, gain=1.0, name=None):
        self.gain = float(gain)
        super().__init__(name=name, gain=self.gain)   # kwargs -> self._properties (re-used when called, and saved)
    def build_layer(self):
        return SquashImpl(gain=self.gain, name=self.name)

x = Input("x", sw=3)
y = Squash(gain=2.0)([x])
```
Rules:
- Every constructor argument must be passed to `super().__init__(name=name, **kwargs)`, because the node is re-instantiated as `cls(name=..., **properties)` when it is called, and saved as `{"name": ..., **properties}`.
- Use `keras.ops` only, never backend-specific ops.
- Register the Impl for `export_keras`.
- Weights go in the Impl's `build()` through `add_weight`. To expose them as Parameters that can be given, shared and frozen, follow `Fir` / `Linear`: subclass `_ParameterWeights` in `nnodely/layers/parameter.py`.
- A multi-input layer receives a list of tensors in `call`.
- Classes register themselves for `save` / `load` by class name, so class names must be unique.
- Example of a layer with its own `output_shape`: `Fir` in `nnodely/layers/fir.py`.

---

## 17. Checklist before returning code

- [ ] `KERAS_BACKEND` is set before the imports, if the user needs a specific backend.
- [ ] Every Input name is unique; every Output name is unique; data dict keys are exactly these names, including target-only Inputs.
- [ ] Every Input is declared once with the window it needs (`sw=n` or `sw=(past, future)`, `sw=1` for one sample, none for a value without time); parts of it are taken with `TimeSelect` / `TimeRange`. Sequences are declared with `Input(..., seq=, sw=)`.
- [ ] The shape of every stream is written down (§3) as `(*seq, time, *dim)`, `T=None` without a window. `Fir` returns one sample (`T=1`); `Linear` keeps the sequence and window and maps the whole dim; combined streams have the same sequence and window (or one is a single sample) and equal or scalar dims.
- [ ] Objectives are declared before `build()`, each with its own name. Targets are Inputs or parts of their windows, streams computed from data, numbers or `None`.
- [ ] Physical constants are `Parameter(value=guess)` or `Constant`, created outside `Ode` functions.
- [ ] No `-stream`; `-1.0 * stream` instead.
- [ ] Objectives go on the outer model, not on `Loop` / `Roll` / `OdeNet` bodies. Loop targets are shifted one step from the seed window. OdeNet seeds are shifted by N-1.
- [ ] A model with closed Inputs (`x.close`, `Integrate`, `Derivative`, `Buffer`) is built with `ModelyLoop`, which lists every Input the step reads that nothing closes; its data carries the horizon (`seq_length=`), and its targets are one step ahead of the states they are compared with.
- [ ] Every layer is called on its streams (`Exp()(x)`, not `Exp()`); `Fir` and `Linear` on one stream each; `Fuzzify` is given its `function`.
- [ ] `train(...)` has explicit `epochs`, `batch_size` and `lr`; `printer=None` for scripts and tests.
- [ ] Inference and data arrays are laid out `(batch, *seq, time, *dim)`, features last, with a batch axis (`loader[i]` needs `v[None]`).
- [ ] Export format matches the model and backend (§13): no ONNX on JAX or for `dopri5`.
- [ ] Results are converted with `keras.ops.convert_to_numpy(...)` before NumPy or matplotlib use.

---

## 18. Internals and repository map (for editing the library itself)

```
src/nnodely/
  __init__.py            public exports (see §4), __version__, the missing-backend check
  core/shape.py          Shape(dim, time, seq), the only module that knows the layout (batch, *seq, time, *dim): axes helpers,
                         Shape.broadcast and align (arithmetic), BATCH_AXIS, OUTER_AXIS = 1, tensor helpers
  core/stream.py         Node, Stream (operator overloads, literal→Constant cache), NODE_REGISTRY, NODE_NAMES
  core/layer.py          Layer base (symbolic call → new node sharing its _source; tensor call → build_layer();
                         get_config from the constructor arguments), BinaryOp Add/Subtract/Multiply/Divide/Power, Identity
  core/modely.py         Modely (build, minimize, train, validate, rollback, all_inputs, save/load, export wrappers), ModelCall
  core/modelyloop.py     ModelyLoop: the step inlined with its closed Inputs (one copy per call), one Loop over them, the rolled Inputs
  core/minimizer.py      objectives: checks, resolution against the built graph, MinimizerModel (the training model)
  core/export.py         export_keras, import_keras, export_onnx, validate_onnx
  core/dag.py            name registry (next_name), flatten (inline ModelCalls per call scope: Scope, flatten_closed), toposort, stream helpers
  core/dataloader.py     DataLoader (read → per-simulation arrays → sliding windows → alignment, mask, normalize)
  core/registry.py       ModelSerializer (format 7): model.json (+ model.weights.h5) via get_config/from_config; bodies in sub-folders;
                         each closed Input with the stream it is closed with and its init
  core/validation.py     validate, score_signal, SignalScore, ValidationResult
  layers/input.py        Input: declared once with dim, seq and its window sw (none: no time axis); its keras.Input;
                         close() (update, init); Inputs, the all_inputs table
  layers/partitions.py   Select, Range, Concatenate (dim axes); TimeSelect, TimeRange, TimeConcatenate, TimeReverse (the window);
                         SeqSelect, SeqRange, SeqConcatenate, SeqStack, SeqReverse (sequence axes)
  layers/{linear,fir,localmodel,fuzzify,function,interpolation,equationlearner,batchnorm,activations,arithmetic,trigonometric}.py   function.py: KerasFunction
  layers/{parameter,output}.py   parameter.py: _Value, Parameter, Constant
  layers/derivative.py   Differentiate (autodiff dispatch per backend), DerivativeStep (Savitzky–Golay banded operator),
                         Derivative (the same stencil one step at a time: the past samples an Input closed with the signal)
  layers/integrate.py    IntegrateStep (cumulative quadrature operator matrix), Integrate (its rules one step at a time:
                         the integral an Input closed with its update)
  layers/buffer.py       Buffer (the past samples an Input closed with the signal, the current one appended)
  layers/ode.py          Ode (Butcher tableaux inline), OdeNet (+dopri5 while_loop, events)
  layers/loop.py         Loop, LoopImpl (tf.while_loop on TensorFlow, keras.ops.scan elsewhere), LoopOutput
  layers/roll.py         Roll/RollImpl, ModelRollImpl (used by Modely.rollback)
  utils/                 printers, graphviz_plot, html_export (+ templates/graph.html, logo.png), validation_plot,
                         random (set_seed), utils (loss/optimizer resolution, MaskedLoss)
tests/                   pytest; fast suite `-m "not slow"`; all backends with scripts/test_all_backends.sh; KERAS_BACKEND defaults to tensorflow
docs/                    Sphinx user guide (getting_started, guide/*.rst, api/*.rst)
```
Mechanics worth knowing:
- Symbolic call: `Layer.__call__(streams)` infers the output `Shape` (by probing `build_layer()` on zeros unless `output_shape` is overridden) and returns a *new* node of the same class with the same name and properties, and the same `_source`: the layer object it applies.
- `Modely.build()` flattens composed models, topologically sorts them, and calls each node's `call()` on Keras tensors. Concrete Keras layers are keyed **by `_source`**, so the nodes of one layer object share its weights and separate objects never do, whatever their names; colliding Keras names get a suffix (`twin_1`).
- Closed Inputs: `Input.close` only records `update` and `init`. `ModelyLoop` inlines the step with `Scope(pending=[], loop=True)`, then what closes every closed Input reached (`flatten_closed`), one copy per call (repeated names get a suffix), and rolls them all out with one `Loop` over `body`; `Modely.build` reads them from data. A call scope starts from its bindings alone; Parameters, Constants and Inputs read from data are shared through `Scope.values`.
- Parameters and Constants have no predecessors. They are called with an anchor tensor of the graph and broadcast their value to its batch, so arithmetic treats them as any stream. They and every relation of them alone are values (`Stream._is_value`), which combine with streams with or without a window; `Layer._needs_window` refuses a stream without one in the temporal layers.
- `build()` makes two Keras models over the same layers: `model.model` for training, whose outputs include the sources and targets of every minimizer (targets computed from data by the graph itself), and `model.inference_model`, with the declared outputs and only the inputs they read. History and printers use the minimizer names.
