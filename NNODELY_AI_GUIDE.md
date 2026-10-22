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

1. **The lifecycle order is fixed:** declare streams → `Modely(name, inputs, outputs)` → `minimize(...)` / `rollback(...)` → `build()` → `DataLoader(model, ...)` → `train` / `validate` / `model(...)` / export. `DataLoader`, `train`, `validate`, inference and export all need a built model. Objectives and rollback must be declared **before** `build()`, and `rollback` after `build` raises. `build()` returns the model, so `Modely(...).build()` chains.
2. **`Input(name, *, dim=None)`**: `dim` is keyword-only, so `Input("x", 3)` is an error. `Input` takes no `seq`: the sequence (outer time) is declared on the Input's **first window**, `x.sw(n, seq=s)`, `x.last(seq=s)`, `x.next(seq=s)` (§3).
3. **Feed layers with windows, not raw Inputs:** use `x.last()`, `x.sw(n)`, `x.sw([p, f])`, `x.next()`, and `x.last(seq=H)` for a sequence. A raw `Input` used as a stream is **timeless**, shape `(dim,)`, data `(batch, *dim)` [verified]; one Input read both raw and through windows is refused by `build()` (`ValueError`) [verified]. As `minimize` targets and as `Loop` / `OdeNet` sequences use windows (`y.next()`, `s.last(seq=H)`): a raw timeless target against a timed source is refused by `train()` [verified].
4. **Stream operators:** `+ - * / **` work between streams and Python numbers; numbers become timeless `Constant`s, applied to every sample. Operands must have the same time axes or be timeless (§3). Unary minus is not supported: `-x` raises `TypeError` [verified], so write `-1.0 * x` or `Negative()(x)`. Also unsupported: `2 ** x`, `@`, indexing `x[0]`, comparisons. For indexing use `Select`, `Range`, `TimeSelect` or `TimeRange`.
5. **Objective targets** (`minimize(name, source, target, loss)`). `target` is one of:
   (a) a window of an `Input` (`y.next()`, `y.sw(n)`, `y.last(seq=H)`), or a raw `Input` for a timeless source;
   (b) any stream computed from the data, such as `Sin()(y.last())` or `y.last() * 2` [verified: the loss is computed against `sin(y)` and `2y`];
   (c) a number (`0` and `0.0` alike), which becomes a `Constant`;
   (d) `None`, which drives `source` to 0.
6. **Objectives are identified by name.** Several objectives may read the same source, and each is logged under its own name [verified]. Registering a name again replaces the objective it names, with a warning.
7. **Per-objective weights:** `minimize(..., gain=0.5)` weighs an objective in the total loss; its logged loss stays unweighted [verified]. `seq_weights=` weighs the steps of a rollout or window (§8).
8. **Numbers as targets:** `target=0` and `target=0.0` are the same; a number becomes a timeless `Constant`, broadcast to the source's shape [verified]. A target read from data must have exactly the source's shape.
9. **Names are keys, not identities.** Input and Output names must be unique within a model: they are the keys of data dicts, results and history. Layer names do not decide weight sharing: two layer objects are two layers with their own weights, even when created with the **same explicit `name=`**; Keras then receives them as `twin` and `twin_1` [verified]. Pass `name=` to look a layer up later (`model.model.get_layer(name)`). Auto names are `Linear1`, `Fir2`, ... from a global counter that skips names in use.
10. **Weight sharing means reusing the same layer object:** `lin = Linear(features=4)`, then `lin([a])` and `lin([b])` share weights. A `Linear(...)` created *inside* a function that runs several times (for example the `f` of `Ode`, which runs once per RK stage) creates new weights on every call. Create weight-bearing layers and `Parameter`s outside `f` and call them inside.
11. **Body models** passed to `Loop`, `Roll` or `OdeNet` keep their own objectives to themselves: the outer model neither trains them nor needs their target inputs [verified]. Put the objectives on the outer model.
12. **Inference reads only what the outputs need:** `model(data)` takes the Inputs in `model.inference_inputs`. Target-only Inputs can be left out and extra keys are ignored [verified]. A missing needed input raises `ValueError` naming it.
13. **`validate()` scores every objective as training does**, including residual (`target=None`) and number targets: its loss is `train()`'s `val_<name>_loss`, its metrics are per channel and leave padded rollout steps out (§13) [verified].
14. **`save()` / `load()`** (nnodely format) **supports every model**, including `Loop`, `Roll` and `OdeNet`, whose bodies are saved in sub-folders. Weights are saved and loaded by default; `save(path, weights=False)` stores the architecture alone and `load(path, weights=False)` ignores saved weights, and either way the model is initialized as `build()` does. A loaded model's layers are its own: they never share weights with the model it was saved from or with another load of the same file. The method an `OdeNet.set_method()` switched to is saved with it. A `Parameter` or `Constant` read by several models of the tree (Loop bodies, blocks, the outer model) is one object: saved under one id in each model's file, it is loaded as one object, listed once in `parameters`, and frozen everywhere at once [verified]. Objectives are not saved: a loaded model has no minimizers; call `minimize(...)` and `build()` again before `train()` [verified].
15. **`Softmax()` normalizes each sample as a whole**: its values over every time and dim axis sum to one [verified]. For a softmax over the features use `Softmax(axis=-1)` [verified]: `axis` is a raw tensor axis (batch 0, time axes next, dims last), so `-1` is the last dim axis. `BatchNorm()` already normalizes per feature (its default axis is the first dim axis).
16. **`Differentiate(respect_to=x)`**: `x` must be the `Input` object, not a window of it. The result has the shape of the window of x that the differentiated relation reads (the union of its windows of x), whatever wider window other relations read [verified]. It does not work inside a model that is used as a composed block (§12).
17. **`train()` defaults to `batch_size=1` and `epochs=10`.** Always pass both (typical batch sizes are 32–256).
18. **Set `KERAS_BACKEND` before the first import** of `keras` or `nnodely`. With no value set, the backend is TensorFlow. A backend that is not installed raises `ImportError` naming the extra to install.
19. **`Parameter` initializes to `random_normal` when no value is given.** For physical constants pass `value=<initial guess>`.
20. **`dt` is always explicit** for `IntegrateStep` and `Derivative`; nothing is inferred from data. Windows are counted in samples, not seconds.
21. **Using a `Loop` node directly** gives the trajectory of its **first callback output**. Unpack it (`a, b = Loop(...)`) to get every body output, in `f.outputs` order [verified].
22. **A model called twice as a block** returns outputs with the same names both times. Combine them, or wrap each in an `Output` with a distinct name; otherwise `build` raises `two different outputs named ...`.
23. **`DataLoader(..., delimiter=, header=)` are passed to `pd.read_csv`** as its `sep` and `header`. Missing (n/a) cells are kept as NaN, with a warning naming the input and the count [verified].
24. **Inference arrays are always batched:** `(batch, *time, *dim)`, a batch of 1 for one sample; `(batch, *dim)` for a never-windowed Input. Flat lists raise; an unbatched array raises, or is silently misread with its first axis taken as the batch [verified]. `loader[i]` has no batch axis: pass `{k: v[None] for k, v in loader[i].items()}`, or `loader.as_dict()` for every sample.

---

## 2. Mapping equations to blocks

| Physical or modeling concept | nnodely construct |
|---|---|
| Measured signal `x(t)` | `x = Input("x")`; vector signal: `Input("x", dim=3)` |
| Current sample `x[t]` | `x.last()` |
| Past n samples `x[t-n+1..t]` | `x.sw(n)` |
| Next sample `x[t+1]` (typical one-step target) | `x.next()` |
| Trajectory of H steps (rollout data, seed, target) | `x.last(seq=H)`: H consecutive samples as an outer time; `seq=-1` follows the data |
| Linear combination of past samples (ARX / FIR term) | `Fir(features=1)([x.sw(n)])` |
| Static affine map / MLP layer on features | `Linear(features=k)([s])` plus an activation |
| Unknown physical constant (mass, stiffness, g/l) | `Parameter("m", value=guess)` |
| Known constant | `Constant("g", value=9.81)`, or just a number in arithmetic |
| Nonlinearity `sin`, `exp`, `abs`, saturation | `Sin()(s)`, `Exp()(s)`, `Abs()(s)`, `Clamp(min, max)(s)` |
| Lookup table / characteristic curve (fixed) | `Interpolation(x_points, y_points)(s)` |
| Gain scheduling / operating regimes | `Fuzzify(centers)` → `LocalModel(lambda i: Fir(features=1))([x.sw(n)], [membership])` |
| Symbolic regression of an unknown law | `EquationLearner(["sin", "multiply", ...], linear_out=Linear(1))` |
| `du/dx` exact (PINN residual, Sobolev) | `Differentiate(order=1, respect_to=x_input)(u)` |
| `dx/dt` from a sampled window | `Derivative(dt=dt, init=..., window=...)(x.sw(n))` |
| `∫ rate dt` over a window / one Euler state update | `IntegrateStep(solver="euler", dt=dt, init=state.last())(rate)` |
| `∫ rate dt` along a horizon (one rate sample per step) | `Integrate(solver="euler", dt=dt, init=x0)(rate.last(seq=-1))`: the horizon is the rate's outer time |
| One explicit RK step of `dx/dt = f(x, u, θ)` | `Ode(f, [states], dt, method="rk4", args=(...))` |
| Learned vector field integrated over reported times | `OdeNet(f=field_model, states={...}, t=times_stream)` |
| Multi-step / simulation-error training | `Loop` (sequence rollout) or `Modely.rollback` (k-step ahead) |
| Physical law as a soft constraint | `model.minimize("law", residual_stream)` (no target) |
| Reuse a trained sub-model | `sub_model([stream_a, stream_b])` (composition) |

---

## 3. Shapes and windows

**Semantic shape** (no batch axis): `(*time, *dim)`.
- `time`: a tuple of time axes, **outermost first**, `(T_k, ..., T_1)`.
  - The **inner time** `T_1` (the last) is the window of consecutive samples. Windows, `Fir`, `Derivative`, `IntegrateStep`, `Roll` and `rollback` act on it.
  - The **outer time** `T_k` (the first) is where a `Loop` rolls out; a collected rollout is prepended as a new outer time. The axes before the inner one, `time[:-1]`, are the stream's **sequence**, declared with `seq=` on a window.
  - The `Time*` layers (§6.5) act on any time axis, counted outermost first: `axis=-1`, their default, is the inner time, `axis=0` the outer time. With one time axis the two are the same.
  - `()` is a **timeless** stream: a number, a `Parameter` / `Constant` declared without time, an Input never windowed. It broadcasts along any time (see below).
  - A dynamic axis (`seq=-1`) is `None` internally, `dyn` when printed.
- `dim`: feature axes. Default `(1,)`; `dim=3` gives `(3,)`; `dim=(3,2)` gives a matrix per sample.
- There is no `seq` attribute or argument besides the windows' `seq=`: no `Shape.seq`, `Stream.seq`, `Input(seq=)`, `Parameter(seq=)` or `Constant(seq=)`.

**Runtime tensor**: `[batch, *time, *dim]`. The time axes are tensor axes `1 … time_rank` (outer time = axis 1, inner time = axis `time_rank`) and the dims are the last axes. Every data array passed to a model or produced by the `DataLoader` follows this layout.

Stream attributes: `s.shape` (a `Shape(dim, time)`, tuple-like over `(*time, *dim)`; prints its named axes, e.g. `Shape(T=(5, 10), D=(3,))`, a dynamic axis as `dyn`), `s.dim`, `s.time` (a tuple), `s.name`, `s.shape.tuple` (= `(*time, *dim)`), `s.shape.dimensions` (= `(time, dim)`), `s.shape.time_rank`, `s.shape.dim_rank`, `s.shape.inner_time`, `s.shape.outer_time` (the window when there is one time axis; a timeless stream raises), and the tensor axes `s.shape.inner_time_axis`, `s.shape.time_axes`, `s.shape.dim_axes` [verified: `Input("w", dim=3).sw(10, seq=5)` has time `(5, 10)`, `inner_time` 10, `outer_time` 5, `inner_time_axis` 2, `time_axes` `(1, 2)`, `dim_axes` `(3,)`].

**Windows on an `Input`:**

| Call | Samples | `time` |
|---|---|---|
| `x.sw(n)` | `x[t-n+1] … x[t]` | `(n,)` |
| `x.sw([p, f])` | `x[t-p+1] … x[t], x[t+1] … x[t+f]` | `(p+f,)` |
| `x.last()` | `x[t]` (= `sw(1)`) | `(1,)` |
| `x.next()` | `x[t+1]` (= `sw([0, 1])`) | `(1,)` |
| `x.sw(n, seq=s)`, `x.last(seq=s)`, `x.next(seq=s)` | `s` consecutive windows, the last ending at `t` (§9) | `(*s, n)` |

`seq=` (an int or a tuple, outermost first; `-1` dynamic) declares the Input's sequence on its **first** window: `x.last(seq=(4, -1))` has time `(4, dyn, 1)`. Every window of the Input has that sequence. A later window omits `seq` and inherits it (after `w.sw(10, seq=5)`, `w.last()` has time `(5, 1)`); giving `seq` again, even the same value, or after a first window without one, raises `ValueError` ("set only once") [verified]. For another sequence, declare another Input.

Requesting a window leaves the Input unchanged. `build()` gives every Input the union of the windows **that model's graph** reads (outputs, minimizer sides, rollback streams): `past = max p`, `future = max f`, `Input.shape.time = (*seq, past + future)`; a window declared but unused widens nothing, and each model reads its own windows (`model.train_inputs` hold them; the Input object shows the last model built) [verified: `u.sw(10)` and `u.sw([2, 3])` both used give `Shape(T=(13,), D=(3,))` after build]. Before build an Input is timeless, `Shape(T=(), D=(1,))`. A loaded model keeps at least the window the saved one read. Windows are ordered oldest first; index `-1` along the inner time is the newest sample.

**Shape rules of common layers** [verified]. Input `v = Input("v", dim=3)`, window `v.sw(4)` of shape `T=(4,), D=(3,)`:

| Expression | Output shape |
|---|---|
| `Fir(features=2)([v.sw(4)])` | `T=(1,), D=(3, 2)`: each of the 3 features filtered alone over its 4 samples onto 2 channels, a new last dim axis; one output sample; outer time axes are kept (`v.sw(4, seq=5)` gives `T=(5, 1)`). `Fir(features=1)` keeps the dim, `D=(3,)`; on a scalar the channels replace it, `Fir(features=2)([x.sw(4)])` is `T=(1,), D=(2,)` |
| `Linear(features=5)([v.sw(4)])` | `T=(4,), D=(5,)`: per-sample map of the whole dim, every time axis kept; a `D=(2, 3)` input gives `D=(5,)` too, and `axis=k` maps only dim axis `k` (`Linear(5, axis=1)` on `D=(2, 3)` gives `D=(2, 5)`), with a matrix per row unless `shared_kernel=True` |
| `Sum()([v.sw(4)])` | `T=(4,), D=(1,)`: sums dim axes (keepdims) |
| `Select(1)([v.sw(4)])` | `T=(4,), D=(1,)` |
| `Range(0, 2)([v.sw(4)])` | `T=(4,), D=(2,)` |
| `TimeSelect(-1)([v.sw(4)])` | `T=(1,), D=(3,)`; `keep_dim=False` drops the time axis, `T=()` |
| `TimeRange(0, 2)([v.sw(4)])` | `T=(2,), D=(3,)` |
| `Concatenate()([v.last(), w_dim2.last()])` | `T=(1,), D=(5,)`: along dim axis 0 |
| `TimeConcatenate()([v.sw(2), v.sw(3)])` | `T=(5,), D=(3,)` |
| `Fuzzify(centers=[0,1,2])([g.last()])` | `T=(1,), D=(3,)`: one feature per center |
| `Fuzzify(centers=[0,1,2])([g.sw(4)])` | `T=(4,), D=(3,)` |
| `Fuzzify(centers=Parameter("c", value=[0,1,2]))([g.last()])` | `T=(1,), D=(3,)`: trainable centers |
| `LocalModel(lambda i: Fir(features=1))([t.sw(10)], [Fuzzify(...)([g.last()])])` | `T=(1,), D=(1,)` |
| `Derivative(dt=dt)(x.sw(n))` | `T=(n,), D=(1,)`: window length preserved |
| `Differentiate(respect_to=t_input)(u)` | the window of `t_input` that `u` reads |
| `IntegrateStep(dt=dt)(a.sw(n))` | `T=(n,), D=(1,)` |
| Elementwise layers (activations, trig, `Exp`, `Clamp`, `Interpolation`, `BatchNorm`) | unchanged |

**Arithmetic broadcasting** (`+ - * / **`, `Shape.broadcast`) [verified]:
- **Time**: streams with time must have exactly the same time axes; different time is never blended. `x.sw(3) + x.last()` (`(3,)` with `(1,)`) and `s.last(seq=4) + x.last()` (`(4, 1)` with `(1,)`) raise `ValueError`. A dynamic axis matches any length and stays dynamic. A **timeless** stream (a number, a Parameter or Constant without time, a never-windowed Input) applies to every sample of any time.
- **Dim**: equal, or one of them a scalar (every axis 1), which broadcasts numpy-style: `(1, 1)` with `(3,)` gives `(1, 3)`. A vector never broadcasts to a matrix: `(3,)` with `(2, 3)` raises.
- **One operand must already have the result's shape**; there is no two-way broadcast. A timeless `(2, 3)` Constant plus `x.sw(4)` raises; plus `x.last()` gives `T=(1,), D=(2, 3)` (a time of length 1 is not a broadcast).
- Examples: `v.sw(4) * x.sw(4)` gives `T=(4,), D=(3,)`; `v.last() * 2`, `v.last() * Parameter("g", value=2.0)` and `v.last() + Constant("c", value=[1., 2., 3.])` give `T=(1,), D=(3,)`.
- At run time the missing time axes of an operand are inserted right after the batch, and the missing leading dims of a scalar before its dims. Parameters and Constants carry the batch axis like any stream.

**Parameter / Constant shapes** [verified]: by default every axis of a `value` is a dim axis and the value is **timeless**: `value=2.0` gives `T=(), D=(1,)`; `value=[1,2,3]` gives `D=(3,)`; `value=[[1,2]]` gives `D=(1, 2)`; an array of shape `(2,3,2)` gives `D=(2, 3, 2)`. Given with a `value`, `dim` / `time` are **axes of the value**, not sizes. `time` names one or several axes, outermost first, and the axes it leaves are the dim unless `dim` names them: `value=[[1,2]], time=1` gives `T=(2,), D=(1,)`; `value=ones((2,3,4)), dim=0, time=(2, 1)` gives `T=(4, 3), D=(2,)`; `dim=(2, 0), time=1` gives `T=(3,), D=(4, 2)`; `dim=(1, 0)` alone transposes a `(2, 3)` value into `D=(3, 2)`. When both are given they must take every axis once (`ValueError` otherwise). Without `value` they are sizes, `time` an int or a tuple (outermost first): `dim=3` gives `T=(), D=(3,)`, `dim=(2,3), time=4` gives `T=(4,), D=(2, 3)`, `dim=2, time=(3, 4)` gives `T=(3, 4), D=(2,)`. `.value` and the variable (`.param`) are laid out `(*time, *dim)`: a timeless `(3, 2)` value is a `(3, 2)` variable; `value=ones((2, 3)), time=1` is `T=(3,), D=(2,)` and a `(3, 2)` variable.

---

## 4. Public API and imports

```python
import os; os.environ["KERAS_BACKEND"] = "torch"   # or "tensorflow" / "jax"; BEFORE imports
from nnodely import (
    Modely, DataLoader, Input, Output, Parameter, Constant, set_seed, get_seed,
    # layers with weights
    Linear, Fir, LocalModel, EquationLearner, BatchNorm,
    # fuzzy / lookup
    Fuzzify, Triangular, Rectangular, Gaussian, Interpolation, KerasFunction,
    # dim axes / time axes
    Select, Range, Concatenate, Sum,
    TimeSelect, TimeRange, TimeConcatenate, TimeStack, TimeReverse,
    # calculus / recurrence
    Differentiate, Derivative, Integrate, IntegrateStep, Ode, OdeNet, Loop, Roll,
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
`Input(name: str, *, dim: int | tuple | None = None)`
- Methods: `.sw(n | [p, f], seq=None)`, `.last(seq=None)`, `.next(seq=None)`. Each returns a new `SampleWindow` stream and widens the Input's window.
- Attributes: `.past`, `.future`, `.shape`, `.dim`, `.time`, `.name`.
- `seq=N` on the first window gives a fixed-length sequence (trajectory): `x.last(seq=N)` has time `(N, 1)`. A tuple gives several outer axes, outermost first. `seq=-1` gives a dynamic length, resolved by `DataLoader(seq_length=...)` and followed by a `Loop` at call time; only one dynamic axis per input is allowed. Later windows inherit the sequence and may not change it (§3).
- Never windowed, an Input is timeless: `(dim,)`, data `(batch, *dim)` (§1.3).
- An Input used only as a target (for example `Input("y_meas").last()`) is read by training and validation (it is in `model.train_inputs`), not by inference (§1.12).

### Output
`Output(name: str, stream: Stream)`: gives a stream a public name. It is the key in results, history and `rollback` / `Loop` mappings.

### Parameter (trainable)
`Parameter(name=None, *, value=None, initializer="random_normal", time=None, dim=None, trainable=True)`
- With `value`: the shape comes from the value, all its axes in the dim (timeless) unless `dim` / `time` name which axes of the value play each part (see §3).
- Without `value`: `dim` / `time` are the sizes (`time` an int or a tuple, outermost first; omitted, the Parameter is timeless), and `initializer` draws the value.
- `initializer` is any Keras initializer name or object.
- After build: `.param` (a Keras variable) and `.value_numpy`, laid out `(*time, *dim)`.
- `p.freeze()` / `p.unfreeze()` (each returns `p`), or `p.trainable = False`, keep the value fixed through training or release it; `trainable=False` creates it frozen. Before or after build: the next `train()` follows. The flag is saved with the model. A Parameter shared by several models is frozen for all of them.
- At runtime it has a batch axis like any stream, the same value for every sample; a timeless one applies to every sample of any stream's time.

### Constant (fixed)
`Constant(name=None, *, value, time=None, dim=None)`: the shape comes from `value`, its axes named by `dim` / `time` as for a Parameter (§3). It is a Parameter that training does not change. After build: `.constant` and `.value_numpy`. Numbers in arithmetic become cached anonymous Constants.

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
- `Fir` kernel `(*dim, window, features)`, bias `(*dim, features)`; without the `*dim` for a scalar input or `shared_kernel=True`. Window rows are the samples of the inner time, oldest first: `Fir(features=2)([x.sw(4)])` has kernel `(4, 2)`, on `D=(2, 3)` `(2, 3, 4, 2)` [verified].
- `Linear` kernel `(*dim, features)`, bias `(features,)`. With `axis=k`: kernel `(*other_dim_axes, dim[k], features)`, bias `(*other_dim_axes, features)`, the other axes dropped with `shared_kernel=True`. On `D=(2, 3)`: `axis=1` gives `(2, 3, 5)` / `(2, 5)`, `axis=0` gives `(3, 2, 5)` / `(3, 5)` [verified].
- `LocalModel` cells with known weights: a factory giving each cell its Parameters, `LocalModel(lambda i: Fir(features=1, kernel=Parameter(value=K[i])))`; the returned stream has no `kernel`.

---

## 6. Layer reference

Legend: **in → out** gives semantic shapes `(*time, *dim)`. `d` = dim, `T` = all the time axes, `To` = the outer time axes (the sequence, `time[:-1]`), `Ti` = the inner time (the window).

### 6.1 With weights
| Signature | in → out | Notes |
|---|---|---|
| `Linear(features=1, kernel="glorot_uniform", bias=True, axis=None, shared_kernel=False, name=None)` | `(*T, *d)` → `(*T, features)`; with `axis=k`, dim axis `k` becomes `features` | Per sample of every time axis. Without `axis` the whole dim is mapped (every feature feeds every output). With `axis=k` only that axis is mapped, with a matrix and bias per position of the other dim axes, or one for all with `shared_kernel=True` (which requires `axis`). `bias=True` draws the bias with glorot, not zeros. Weights: §5. |
| `Fir(features, kernel="glorot_uniform", bias=True, shared_kernel=False, name=None)` | `(*To, Ti, *d)` → `(*To, 1, *d, features)` | Each dim element is filtered alone over its inner-time window (elements never mixed), with a kernel of its own or one for all with `shared_kernel=True`; applied at every step of the outer time. The channels are a new last dim axis. `features=1` keeps the dim (`(*To, 1, *d)`); on a scalar the channels replace it (`(*To, 1, features)`). `bias=True` is zeros. `features` is required. Weights: §5. |
| `LocalModel(input_function=None, output_function=None, vectorized=False, name=None)` called as `(inputs, activations: list)` | `a_k: (T, n_k)`: dim `(n_k,)` and at most one time axis, T the cells' inner time or 1 → shape of one output cell | The activations are multiplied into `N = Π n_k` joint memberships `mu`, row-major (cell `(i1, i2)` is `i1*n2 + i2`), and a `CellWeight` layer (`<name>_weighted`) weighs the cells with them: a 1-sample membership weighs every sample of a window. Activations with outer time raise. Computes `Σ_i output_function_i(mu_i * input_function_i([x...]))`. Each function is: `None` (no function: one input weighted as it is / the weighted cells only summed); a weightless elementwise layer instance (`Sin()`, `Tanh()`, `ReLU()`, ...), applied once; any other `Layer` instance, applied to every cell with the same weights; or a callable, which is a factory `f((i1, i2, ...)) -> layer` called once per cell. With `vectorized=True` a callable or a layer (other than a weightless elementwise one) is instead called once on all cells and must return `(*cell, N)` (`(N,)` for scalar cells), e.g. `Fir(features=N)`: its N channels are the N cells. Functions receive a **list** of streams; all cells must have the same shape. Lists of functions are not accepted. |
| `EquationLearner(functions: list, *, linear_in: Linear \| None = None, linear_out: Linear \| None = None, name=None)` called on `stream` or `[streams]` | `(*T, d)` → `(*T, n_functions)`, or `linear_out`'s output | `functions` items can be a name (`identity sin cos tan asin acos atan relu leaky_relu elu prelu sigmoid tanh swish gelu softplus add subtract multiply divide power`), a Layer class or instance, a callable over streams (arity inferred), or `(callable, arity)`. `linear_in` projects to Σarity arguments; its `features` must equal that sum. Inputs must have dim rank 1 (several inputs are concatenated). Internally it is a composed `Modely`. |
| `BatchNorm(axis=None, momentum=0.99, epsilon=1e-3, center=True, scale=True, name=None)` | unchanged | Per-feature statistics, along the stream's first dim axis by default [verified]. An explicit `axis` is a raw tensor axis (batch 0, time axes next, dims last). Uses the inference branch in `validate` and inference. |
| `Parameter(...)` | – | See §5. |

### 6.2 Fuzzy and lookup (no weights)
| Signature | in → out | Notes |
|---|---|---|
| `Fuzzify(centers, function=Triangular, name=None)` called on a scalar stream | `(*T, 1)` → `(*T, n)` | `centers` are numbers (fixed, sorted), or move with the model: one stream of dim `(n,)` with one sample of inner time or none - a `Parameter` vector (trained) or a computed stream such as `Fir(features=n)` (every sample its own centers) - or a list of scalar streams `(1,)` and numbers (the numbers become Constants, kept fixed). Moving centers are not sorted, used in the order given (give them ascending); the memberships are placed on them and sized by their neighbors as computed, so the gradient reaches them and the streams they come from (`Rectangular` has none); with outer time axes they have the signal's. A matrix, a time window or a non-scalar list item raises `ValueError`. `function` is one function for every center or a list of one per center (paired before the centers are sorted). A membership class `Triangular` / `Rectangular` / `Gaussian` (or its name as a string) is placed on each center and sized by its neighbors, as before: triangles reach the neighboring centers (lopsided for uneven centers) and sum to 1 inside the center range, rectangles reach halfway to the neighboring centers, half-open, so exactly one holds every value between the centers, Gaussians have sigma = (left gap + right gap) / 6; the outer sets reuse the one gap they have. Any other nnodely layer class (made anew per center) or instance (shared) is applied to `x - center`: `function=Sin` gives `sin(x - c)`. So is a custom function, a `keras.layers.Layer` subclass (made anew per center) or instance (shared), wrapped in a `KerasFunction` so the model saves, loads and exports; its class is registered with Keras for this session - in another, define it (with `@keras.saving.register_keras_serializable()`) before loading. Its `call` reads one argument, `x - c`: a `call` needing more (those with a default, such as `training=None`, aside) raises `TypeError`, and so does calling a Fuzzify on more than one stream. Not a Layer itself: it builds one layer per center, named after its class (`<name>_<Class>_<index>`, e.g. `Fuzzify1_Triangular_0`; a lone center takes the Fuzzify's name, a shared Keras instance `<name>_<Class>`), and a `Concatenate` named after it; a repeated call reuses its layers. |
| `KerasFunction(function, name=None)` called on `stream` or `[streams]` | shape read off the Keras layer's result | A custom function: `function` is a `keras.layers.Layer` subclass (made here, no arguments) or instance (its weights shared wherever it is used). The streams, laid out `(batch, *time, *dim)` (features on the last axis), are passed to `call` as separate arguments: `call(self, x, y)` takes two. Saves, loads and exports (the class is registered with Keras for this session; in another, define it with `@keras.saving.register_keras_serializable()` before loading). Its weights are Keras weights, trained with the model but not `Parameter`s: `model.freeze()` / `model.parameters` do not reach them - set the layer's `trainable`. |
| `Triangular(center=0.0, radius=1.0, name=None)` | unchanged | 1 at `center`, linear down to 0 at `radius` (a number or `(left, right)`), 0 beyond. Elementwise on any stream. |
| `Rectangular(center=0.0, radius=1.0, name=None)` | unchanged | 1 on `[center - left, center + right)`, 0 elsewhere; `radius` a number or `(left, right)`. |
| `Gaussian(center=0.0, sigma=1.0, name=None)` | unchanged | `exp(-((x - center) / sigma)^2 / 2)`. |
| `Interpolation(x_points, y_points, mode="linear", name=None)` | elementwise | `mode` is `"linear"` (piecewise) or `"polynomial"` (the unique global polynomial). Queries are clamped to `[x_min, x_max]`. Not trainable. |

### 6.3 Activations (elementwise, no weights except PReLU)
`ReLU(max_value=None, negative_slope=0.0, threshold=0.0)`, `LeakyReLU(negative_slope=0.3)`, `ELU(alpha=1.0)`, `PReLU(shared_axes=None)` (trainable: one slope per feature, shared along every time axis, a dynamic one included [verified]; `shared_axes` counts raw tensor axes: batch 0, time axes `1 … time_rank`, dims last), `Sigmoid()`, `Tanh()`, `Swish()`, `GELU(approximate=True)`, `Softplus()`, `Softmax(axis=None)` (the whole sample; `axis=-1` for the features, a raw tensor axis, §1.15). All accept `name=`. Called on a list of streams, an activation (or a math or trigonometric layer, `Clamp`, a membership function) is applied to each and returns a list, `Sin([a, b])` → `[sin(a), sin(b)]`; one with settings applied before it is configured, `ReLU(x)`, raises `TypeError`.

### 6.4 Math (elementwise)
`Sin Cos Tan Asin Acos Atan Exp Log Log10 Sqrt Abs Floor Ceil Deg2Rad Sign Negative` take no arguments except `name=`. Neither does `Atan2`, called on `[y, x]`: `atan2(y, x)`, the angle of the point `(x, y)` in `[-pi, pi]`, with its quadrant; `y` and `x` broadcast as the arithmetic operators do (§3), e.g. a `Constant` against a sequence. `Clamp(min=None, max=None)`: a `None` bound is unbounded. `Sum(axis=None)` sums the dim axes and keeps them with length 1 (`axis=None` sums all dim axes, an int picks one). `Floor`, `Ceil` and `Sign` have zero gradient.

### 6.5 Axis manipulation
| Signature | Effect |
|---|---|
| `Select(idx, axis=0, keep_dim=False)` | Picks one index on a dim axis, **the axis dropped**: `D=(4, 3)` → `D=(3,)`; `keep_dim=True` keeps it with length 1, `D=(1, 3)`. A stream always has a dim, so a vector gives `D=(1,)` either way. Negative idx allowed; an index out of range raises. |
| `Range(start, end, axis=0)` | Slice `[start, end)` on a dim axis. |
| `Concatenate(axis=0)([a, b, ...])` | Concatenates along a dim axis. Timed inputs must have equal time; a timeless input (a Parameter, a number) is repeated along the time of the others [verified]. |
| `TimeSelect(idx, axis=-1, keep_dim=True)` | One index on a time axis, **kept with length 1** (`-1` = newest sample of the window); `keep_dim=False` drops the axis: `T=(4,)` → `T=()`, and with `axis=0` `T=(dyn, 1)` → `T=(1,)`. Negative idx counts from the end of the data, also on a dynamic axis, so `TimeSelect(-1, axis=0)` is the last step of the outermost rollout, of any length. |
| `TimeRange(start, end, axis=-1)` | Slice `[start, end)` on a time axis, kept; `None` leaves a bound open. On a dynamic axis a negative bound or an open end gives a dynamic axis. |
| `TimeConcatenate(axis=-1)([a, b, ...])` | Joins along an existing time axis; dim and the other time axes must match. A dynamic input makes the axis dynamic. |
| `TimeStack(axis=-1)([a, b, ...])` | Stacks equal-shape streams into a new time axis: 3 × `T=()` → `T=(3,)`; with `axis=0`, 4 × `T=(1,)` → `T=(4, 1)`. `axis` is the new axis's position among the result's time axes: `-1` the new inner time, `0` the new outermost. |
| `TimeReverse(axis=-1)` | Reverses the elements of a time axis. |

The `Time*` layers mirror the dim ones on the time axes. Their `axis` counts the time axes **outermost first**: `-1`, the default, is the inner time, the window; `0` is the outer time, the rollout of the outermost (last applied) `Loop`, or the window when there is no rollout. On `T=(3, 5, 1)` [verified]: `TimeSelect(-1, axis=0, keep_dim=False)` → `T=(5, 1)`, `TimeSelect(0, axis=1)` → `T=(3, 1, 1)`, `TimeSelect(0, axis=1, keep_dim=False)` → `T=(3, 1)`, `TimeRange(1, 3, axis=0)` → `T=(2, 5, 1)`, `TimeRange(1, 3, axis=1)` → `T=(3, 2, 1)`, `TimeStack(axis=0)([s, s])` → `T=(2, 3, 5, 1)`, `TimeStack()([s, s])` → `T=(3, 5, 1, 2)`. A timeless stream raises (`axis -1 out of bounds for time rank 0`). Dropped from the only time axis, a sample is a timeless stream, which broadcasts along any window: `w - TimeSelect(-1, keep_dim=False)(w)` subtracts the newest sample from each [verified]. `TimeSelect(-1, axis=0, keep_dim=False)` on a collected trajectory gives a stream that can close a `Loop` callback on an input without that outer time. There is no layer that reshapes or transposes. Write a custom layer (§16) if one is needed.

### 6.6 Calculus and recurrence
See §10 (`Loop`, `Roll`, `rollback`) and §11 (`Differentiate`, `Derivative`, `IntegrateStep`, `Integrate`, `Ode`, `OdeNet`).

---

## 7. Modely API

`Modely(name: str, inputs: list[Input], outputs: list[Output])`
- `inputs` lists the Inputs the model reads. Inputs used only as targets need not be listed; they are discovered at build. After build, `model.train_inputs` holds every Input that training data must provide, and `model.inference_inputs` the ones inference reads.
- `outputs` is a list of `Output` nodes.

| Method | Purpose |
|---|---|
| `minimize(name, source, target=None, loss="mse", gain=1.0, seq_weights=None)` | Registers an objective (§8). Call before build. |
| `remove_minimizer(name)` | Removes an objective by name. |
| `parameters` (property) | Every `Parameter` the model computes with (relations, layer weights, called models, Loop/Roll/OdeNet bodies), in graph order: a list of the Parameter objects; `print(model.parameters)` shows a table of name, time (`-` when timeless), dim, number of values and trainable, with totals [verified]. Before or after build. |
| `freeze()` / `unfreeze()` → self | Freezes or releases every `Parameter` the model computes with: its relations, the weights of its layers, the models it calls and the bodies of its Loops, Rolls and OdeNets. Before or after build. The Keras weights of `PReLU`, `BatchNorm` and `KerasFunction` are not Parameters and are not frozen: set `trainable` on their Keras layer. |
| `rollback({input: stream}, steps: int, name=None)` → self | k-step feedback unrolling (§10). Call before build. |
| `build()` → self | Creates the `keras.Model` (`model.model`) and its weights. Can be called again; weights are kept. |
| `model(inputs_dict)` | Inference. Takes `{input_name: array (batch, *time, *dim)}`, the batch axis always included (§1.24). Reads `model.inference_inputs`, ignores other keys, and raises `ValueError` when one is missing. Returns `{output_name: backend tensor}` for the declared outputs; convert with `keras.ops.convert_to_numpy`. |
| `model([stream, ...])` | Composition: one stream per input, in `model.inputs` order; returns the outputs as streams of an outer graph (§12). |
| `train(train_data, val_data=None, epochs=10, batch_size=1, optimizer=None, early_stopping=None, lr=1e-3, shuffle=True, optimizer_kwargs=None, early_stopping_kwargs=None, printer="legacy", jit_compile="auto")` → `dict` | Keras history (§8). |
| `validate(val_data, out_dir=None, show=False, history=None, verbose=True)` → `ValidationResult` | Metrics plus optional PNG figures (§13). `verbose=False` skips the printed summary. |
| `flatten()` → Modely | Inlines every composed sub-model. |
| `summary()` | Keras summary. |
| `export_html(out_dir, filename=None, *, open_subgraph_in_new_tab=False, physics=True)` | Interactive vis-network graph, one page per sub-model. |
| `plot(out_dir, filename=None, *, flatten=False, include_minimizers=True, format="png")` | Graphviz images in `out_dir`, the root named `filename` (the model name by default); a path or filename with an image suffix (`.png`, `.svg`, `.pdf`) names the file and format. Nodes show only their name, coloured by role (inputs green, outputs red); arrows carry the shape `(*time, *dim)` (`dyn` for a dynamic time axis). Without `flatten`, each block (ModelCall, Loop, Roll) is one SubModel node and its body is drawn as `<filename>_<body name>`, recursively; with `flatten`, one image, each called sub-model boxed. Loop recurrences are an orange `loop` arrow, Roll and rollback a teal `roll (steps=n)` arrow. Every image has the logo and a legend (Input, Output, Parameter, Constant, SubModel, Loss, loop, roll). Without the Graphviz `dot` program it warns and writes each image's DOT source as `.gv`. Returns None. |
| `save(path, weights=True)`, `Modely.load(path, weights=True)` | nnodely format: a folder with `model.json` and `model.weights.h5`, bodies of Loop, Roll and OdeNet in sub-folders (§1.14). |
| `export_keras(path, filename=None)`, `Modely.import_keras(filename, safe_mode=True)` → `keras.Model` | Writes `path/<filename or model name>.keras`: a stand-alone copy for inference. A weight read from several Keras graphs (a body in two Loops, a Parameter inside and outside a body, a block also in a Loop, the sub-graphs of `Differentiate`) is written once per graph with the same value, since Keras cannot rebuild a layer two graphs call: the imported model gives the same outputs, but further Keras training would untie the copies - use `save`/`load` for that [verified]. The imported object is a plain Keras model; call it with batched float32 dicts. Pass `safe_mode=False` if loading refuses nested or lambda configs. |
| `export_onnx(path, filename=None, *, input_signature=None, batch_size=None, opset_version=None, verbose=False)` | Writes `path/<filename or model name>.onnx`. Needs the `onnx` extra and the TF or Torch backend (§13). |
| `Modely.validate_onnx(filename, inputs, *, return_dict=False, providers=None)` | Runs an exported ONNX file with onnxruntime. |

Properties: `built`, `model` (the training keras.Model), `inference_model` (the keras.Model inference runs), `inputs`, `outputs`, `order` (topological node list), `minimizers` (list of dicts with keys `name`, `source`, `target`, `loss`, `gain`, `seq_weights`), `train_inputs`, `inference_inputs`, `train_outputs`. `print(model)` lists the nodes.

---

## 8. Objectives and training

`minimize(name, source, target=None, loss="mse", gain=1.0, seq_weights=None)`
- `source`: an `Output` (preferred), any stream, or the name of a stream of the model.
- `target` follows §1.5. Common forms:
  - `y.next()`: one-step-ahead prediction of an input that is also read by the model.
  - `Input("y_meas").last()`: a separate measured column.
  - `Input("s_target", dim=2).last(seq=H)`: a trajectory target for `Loop` (the shapes must match the source).
  - `Sin()(y.last())`: a target computed from the data.
  - `None`: residual.
- `loss`: any Keras loss (a name such as `"mse"`, `"mae"`, `"huber"`, `"log_cosh"`), a `keras.losses.Loss` instance, a serialized config, or a callable `loss(y_true, y_pred)`.
- `gain`: the weight of the objective in the total loss, `Σ gain_i · loss_i`. Its logged and validated loss stays unweighted.
- `seq_weights`: one weight per step of the source's **outer time** (tensor axis 1): the rollout of a Loop, or the window of a stream without one [verified], e.g. `np.exp(0.1 * np.arange(N))`, or a callable of the number of steps for a dynamic length. Normalized to mean one; the logged loss is the weighted one.
- Shapes: a constant target (a number, a Constant) broadcasts to the source; a target read from data must have exactly the source's shape, or `train()` and `validate()` raise (e.g. a raw timeless Input against a `last()` source) [verified].

`train(...)`
- `optimizer`: a name (`sgd rmsprop adam adamw adagrad adadelta adamax adafactor nadam ftrl lion`; default `"adam"`), a config dict, or an instance. `lr` and `optimizer_kwargs` (e.g. `{"weight_decay": 1e-4, "global_clipnorm": 1.0}`) apply only to names; configure an instance yourself.
- `early_stopping`: the name of a logged quantity (`"loss"`, `"<minimizer name>_loss"`, or their `val_` versions with `val_data`), completed by `early_stopping_kwargs` into a `keras.callbacks.EarlyStopping`; a dict of `EarlyStopping` arguments; or any callback that sets `self.model.stop_training = True`. Typical: `early_stopping="val_loss", early_stopping_kwargs={"patience": 10, "restore_best_weights": True}`. Monitoring a quantity that is not logged raises `ValueError`; the mode defaults to `"min"` (every logged quantity is a loss).
- `printer`: `"legacy"` (default, a loss table), `"tiny"`, `"nnodely"` (animated), `None` (silent), or any `keras.callbacks.Callback`.
- Returns a history dict: `loss` (the total) and `<minimizer name>_loss` for each objective, plus their `val_` versions with `val_data` [verified: one objective named `one` gives `['loss', 'one_loss', 'val_loss', 'val_one_loss']`]. Every logged loss is the mean over the dataset's samples, batches weighed by their size (a short last batch counts for its samples only) [verified].
- Each `train` call continues from the current weights.
- Padded rollout steps (`seq_length="full"`) are masked automatically, along the outer time (tensor axis 1).
- To freeze a composed block, call `block.freeze()` before training the outer model; `block.unfreeze()` releases it. `p.freeze()` freezes one Parameter.

Reproducibility: call `set_seed(int)` before creating layers and data. It seeds Python, NumPy and the backend, and sets `NNODELY_SEED` so child processes inherit it; two runs with the same seed give identical results on each backend. `get_seed()` returns the current seed.

---

## 9. DataLoader

`DataLoader(model, source, format=None, csv_glob="*.csv", delimiter=",", header="infer", dtype=np.float32, seq_length=None, step=1, on_short="error")`. The model must be built.

**Sources**
- A dict `{input_name: array (T,) or (T, *dim)}`. Keys must be the model's input names, including target-only inputs. `format` is ignored for dicts.
- A `pd.DataFrame`, a `.csv` path, or a folder of CSVs matching `csv_glob`. CSV files are read with `pd.read_csv(sep=delimiter, header=header)`. `format={"input": "col" | int_index | ["c1","c2",...]}` maps inputs to columns; a list is used for `dim>1`. Unmapped columns are ignored, and an input not listed in `format` is looked up by its own name. n/a cells are kept as NaN, with a warning.
- A list of dicts or DataFrames, or a folder with several files: several **simulations**. Windows never cross simulations, and lengths may differ.

**Windowing and alignment** [verified]
- Every input is aligned on a common current sample `t`. Within one simulation of length `T`:
  - `first = max_past - 1`, where `max_past` is the largest past over all inputs.
  - The number of samples is `T - max_future - first - max_seq_span`, where `max_seq_span = Σ(seq_len - 1)`. Sample i has current time `t_i = first + max_seq_span + i`.
  - `step=k` keeps every k-th sample.
- A sequence input `x.sw(n, seq=L)` gives, at sample i, the L temporal windows ending at `t_i-L+1 … t_i`, shape `(L, n, *dim)`. **The last sequence element is aligned with the current sample of non-sequence inputs.** For example, `a.last()` = 3 while `b.last(seq=3)` = [1,2,3].
- A never-windowed input contributes its current sample, with no time axis: `(N, *dim)`.
- A too-short simulation raises. `on_short="skip"` drops it.
- Dynamic `seq=-1` needs `seq_length=int`, or `seq_length="full"` to get one sample per simulation over the whole length; `"full"` resolves only the **outermost** time axis (`seq=(3, -1)` raises). With `"full"`, simulations are padded along the outer time (axis 1) by repeating the last step; `loader.mask` has shape `(n_sims, max_len)` and training masks the padded steps. `step>1` is not allowed with `"full"`.

**Access**: `len(loader)`, `loader[i]` → `{name: (*time, *dim)}` (no batch axis: add one with `v[None]` before calling the model), `iter(loader)`, `loader.as_dict()` → `{name: (N, *time, *dim)}` (e.g. `c.sw(5, seq=3)` of a `dim=2` input gives `(N, 3, 5, 2)`, a never-windowed `dim=2` input `(N, 2)` [verified]), `loader.get_samples(n, start=0, simulation=0)` → n consecutive samples of one simulation, batched like `as_dict()`, `loader.get_input(name)`, `loader.inputs`, `print(loader)`.

**Normalization** (explicit and per loader): `loader.normalize(method="minmax" | "standard", names=None, feature_range=(-1.0, 1.0))` changes the loader in place, with statistics computed per feature, reducing over the samples and every time axis. `loader.denormalize()` restores the loader. `loader.denormalize({output_name: array})` inverts predictions, matching each output to the input its minimizer target reads. `loader.normalization_stats` holds the statistics. A validation loader must be normalized with the same statistics. Normalizing each loader separately gives different statistics, so for consistent scaling prefer normalizing the raw arrays yourself, or fold the scaling into the model with `Constant`s.

---

## 10. Recurrence: rollback, Loop, Roll

| Need | Use |
|---|---|
| Train to predict k steps ahead with the model's own feedback in its input window; only the final prediction counts | `model.rollback({x: x_pred}, steps=k)` |
| Simulation-error training over whole trajectories, returning every step; exogenous inputs change per step | `Loop` (recommended) |
| Unroll a one-step model over the time window of one of its inputs, as a block inside a graph; other inputs held fixed | `Roll` |
| Continuous-time learned field integrated over reported times (neural ODE) | `OdeNet` (§11) |
| Several states integrated physically at each step | `Ode` step as the body of `Loop` |

### 10.1 `Modely.rollback(rollback: dict, steps: int, name=None)`
- Mapping `{input (obj or name): stream (obj or name)}`. The stream can be any stream of the graph, including an Output, and must have the input's dim and one sample of inner time, its outer time kept: time `(*input.time[:-1], 1)` [verified].
- It evaluates the model `steps` times. After each evaluation, the stream's one-sample value is appended to the input's inner-time window and the oldest sample is dropped. Model outputs are those of the **final** evaluation.
- Inputs that are not fed back, and targets, are held at their window values for every step. Align the target yourself: `data = {"x": x[:-k], "F": F[:-k], "x_ahead": x[k:]}` with target `Input("x_ahead").last()` (recipe R8).
- Several feedbacks at once: `rollback({"x": "x_next", "v": "v_next"}, steps=10)`.
- Supports `save` / `load`.

### 10.2 `Loop(f, callback, init=None, name=None, length=None, collect=True)(inputs=None)`
- `f`: the body `Modely` (built automatically if needed). Its own objectives are ignored (§1.11).
- `callback`: `{body_input: body_output}`, as objects or names. The output dim must equal the input dim, and the output must have the input's outer time axes (its sequence; an error mentions "the outer time axes, the sequence"). Its inner time must be 1: with an input window of 1 the state is replaced, with a longer one the window is shifted along the inner time, and after n steps it holds only predictions [verified]. The input window must read no future samples (`future == 0`: no `x.sw([p, f])` with `f > 0`, no `x.next()`) [verified].
- The seeds are configured with the loop, and the loop is bound by calling it on its exogenous inputs: `Loop(f, callback, init={...})({...})`. `Loop(...)()` when there are none.
- `init`: `{body_input: stream | number}` seeding the fed-back inputs. A fed-back input missing from the dict starts at zero; a non-dict raises `TypeError`.
  - A stream with **one more time axis** than the body input, the rollout as its outer time (tensor axis 1), e.g. `Input("s0", dim=2).last(seq=H)`, `T=(H, 1)`, for body input `state.last()`, `T=(1,)`: step 0 reads its first step [verified].
  - A stream of the same shape: used directly.
  - A number: broadcast.
  - For a windowed feedback input use a window with the rollout: `{x: x0.sw(3, seq=H)}` [verified].
- `inputs`: `{body_input: outer_stream}` for exogenous signals, given in the call (`Loop(...)()` when there are none). The outer stream must match the body input's dim, and either have its time (held constant) or **one more outer time axis** (tensor axis 1, consumed one step per rollout step), e.g. `F.last(seq=H)` for body input `force.last()`. A timeless stream for a timed body input raises [verified].
- Body inputs that are neither fed back nor bound are held constant, and become inputs of the outer model under their own names.
- Horizon: the outer-time length of the exogenous rollout streams (of the `init` ones only when no input carries a rollout; an `init` is read at its first step only). Pass `length=` when the exogenous lengths differ, or when no stream carries a rollout [verified]. With `seq=-1`, the rollout follows the length of the tensor given at call time (also in a TensorFlow ONNX export), so `length` is optional: without it the streams are declared with `DYNAMIC_BUILD_LENGTH` (1) steps.
- At least one source must derive from an `Input`, so that there is a batch axis.
- Timing: at step k (0-based) the body sees state `s_k` (with `s_0` from `init`) and exogenous `u_k`. Every body output at step k is computed from `s_k`, and the callback output becomes `s_{k+1}`. With `collect=True` the result gets steps `0..H-1` as a new **outer time**, prepended: a body output `T=(1,), D=(2,)` with horizon H gives `T=(H, 1), D=(2,)`, arrays `(B, H, 1, 2)` [verified]. So the fed-back trajectory is `s_1..s_H`: with `initial` = states window `[t..t+H-1]`, the target must be states `[t+1..t+H]`, i.e. data `{"s_seq": S[:-1], "s_target": S[1:]}` (recipe R3). A non-fed-back output (for example a measurement `y(s_k)`) aligns with the unshifted window. `collect=False` returns only the last step.
- Multiple outputs: `a, b = Loop(...)` yields one stream per body output, in `f.outputs` order.
- Loops can be nested; the body of a Loop can itself contain a Loop. Each rollout is prepended: a body output carrying an inner rollout `T=(3, 1)` gives `T=(H, 3, 1)` [verified]; close a callback on the inner trajectory's last step with `TimeSelect(-1, axis=0, keep_dim=False)`.
- Supports `save` / `load`, `export_keras` and ONNX (on TensorFlow the rollout is an ONNX `Loop` node).

### 10.3 `Roll(f, callback: dict, steps=None, name=None)`
- `f` must already be built (it raises otherwise). `callback` holds exactly one pair `{input: output}`, and the output must have one sample of inner time and the input's dim and outer time, time `(*input.time[:-1], 1)`.
- Returns a stream shaped like the callback input's union window, its outer time kept (`x.sw(5, seq=3)` gives `T=(3, 5)`) [verified]. Every step feeds back along the inner time. `steps` defaults to the window length, so every sample is a prediction. The outer model must list the body's own Input objects as its inputs, because the other inputs are fed unchanged at every step.
- Supports `save` / `load`, `export_keras` and ONNX.

---

## 11. Physics layers

### 11.1 `Differentiate(order=1, respect_to=<Input>, name=None)(stream)` and `Derivative(order=1, dt=<float>, init=None, window=None, poly_order=None, name=None)(stream)`
Orders 1 and 2 are supported by both. Passing a number to `Differentiate` raises `TypeError` pointing to `Derivative`; `Derivative` requires a positive `dt`.
- **`Differentiate`, with respect to an Input** (autodiff, exact): the result is `d(Σ stream)/d(input window)` (a VJP), shaped like the window of the Input the stream reads. It includes the trainable weights, can be nested, and can be trained through. It takes no `init`, `window` or `poly_order`. The stream must depend on that Input. Typical use is a PINN residual: `minimize("physics", Differentiate(respect_to=t)(u) + u)`. For `u` with dim>1 the result is the gradient of the sum of the components; take `Select` first to get one component. ONNX export: TensorFlow only, with batch fixed to 1. Torch refuses, and JAX cannot export.
- **`Derivative`, with respect to time** (`dt: float`, as in `IntegrateStep`): a causal finite difference along the stream's inner time that **keeps the window length**; outer time axes are kept.
  - `init` is the sample(s) just before the window: a stream of 1 sample of inner time (reused for each) or `window-1` samples, or a timeless one, its dim a scalar or the stream's, its outer time none or the stream's [verified]; a number; or `None` (zero).
  - `window` defaults to `order+1`, the plain backward difference. Larger values fit a polynomial by least squares (Savitzky–Golay style): they smooth noise at the cost of about `(window-1)/2` samples of delay.
  - `poly_order` ranges over `order..window-1` and defaults to `order`.
  - The time derivative is the exact inverse of `IntegrateStep(solver="euler")` with the same `init`.

### 11.2 `IntegrateStep(solver="euler", dt=<required float>, init=None, name=None)(rate)`
- `solver` is `"euler"` / `"rectangular"` (y[i] = y[i-1] + dt·r[i]) or `"trapezoidal"`. It integrates along the inner time and preserves the window length. `init` is the value just before the window: a stream of one sample of inner time, or a timeless one (dim a scalar or the rate's, outer time none or the rate's) [verified]; a number; or `None` = 0.
- On a one-sample window it is exactly one state update: `IntegrateStep(dt=dt, init=v.last())(a.last())` = `v + dt*a`.
- On an n-sample window it returns the whole running integral in one pass, with no rollout needed: `x = IntegrateStep(dt=dt)(IntegrateStep(dt=dt)(a.sw(n)))`.
- There is no rk4 or heun here; use `Ode` for those.

### 11.2b `Integrate(solver="euler", dt=<required float>, init=None, name=None)(rate)`: along a horizon
- A block: an `IntegrateStep` rolled out by a `Loop`, one rate sample per step, the integrated value fed back as the next step's state. Calling it returns that `Loop` node, bound to the rate and `init`: use it directly in further relations, `Output` or `minimize`. `Integrate.body` is the one-step `Modely`.
- `rate` must carry the horizon as its **outer time** (its outermost time axis: `acc.last(seq=N)`, or `acc.last(seq=-1)` to follow the data length at each call) and have one sample of inner time per step: `Integrate(...)(acc.last(seq=-1))`. A rate without outer time, or with more than one inner sample, raises and points to `IntegrateStep` [verified]. Further outer axes (`seq=(H, M)`: horizon H) are integrated apart. The result is shaped like the rate: element k of the outer time is the value at the end of step k.
- Same rules and first-step convention as `IntegrateStep`: the same samples give the same values along a horizon or along a time window. `"trapezoidal"` feeds the previous rate back too.
- `init`: a stream of one sample of inner time, or a timeless one (the rate's dim, or a scalar for a one-axis dim), whose outer time is the rate's without the horizon (`rate.time[1:-1]`, so none for `seq=N`) [verified]; a number; or `None` = 0.
- Chains: `pos = Integrate(dt=dt, init=x0)(Integrate(dt=dt, init=v0)(acc.last(seq=-1)))`. A dynamic Loop result keeps a dynamic outer time, so a second rollout follows it.
- A dynamic horizon stays dynamic through further relations (`x * 2.0`, `x * Parameter(...)`, `Sin()(x)`, `Linear`, another `Integrate`), and trains on simulations of different lengths with `DataLoader(..., seq_length="full")`, the padded steps masked.

### 11.3 `Ode(f, states, dt, method="rk4", args=())` (a function, not a class)
- Advances the states by **one** explicit step. `method` is one of `euler`, `midpoint`, `heun`, `rk4`.
- `f(*state_streams, *args)` must return one derivative stream per state, in order (a tuple or list, or a single stream when there is one state).
- `states` is a list of streams (typically `x.last()`) or a single stream. `dt` is a number or a stream (it can be a `Constant`, a `Parameter`, or read from data).
- Returns a list of next-state streams (a single stream when a single state was given).
- `f` is evaluated once per RK stage, building new nodes each time. Keep `Parameter`s and weight-bearing layers **outside** `f` (§1.10).
- For a trajectory, wrap the `Ode` step in a body `Modely` and roll it out with `Loop`, or use `rollback`.

### 11.4 `OdeNet(f, states, t, initial=None, method="rk4", steps=1, event=None, reset=None, rtol=1e-6, atol=1e-8, max_steps=1000, name=None)`
- `f` is a built `Modely` giving the vector field `dx/dt`. It must be **autonomous**: every body input is a state, and exogenous drivers are rejected.
- `states`: `{state_input: derivative_output}`. The derivative must match the state's dim and time.
- `t`: a stream of report times whose **outer time** has a static count N ≥ 2, e.g. `Input("t").last(seq=N)` with data `(batch, N, 1, 1)` [verified]. The times are **shared across the batch** (the first row is used), and `t[0]` is the initial condition.
- `initial`: `{state: outer_stream}` with exactly the state's dim and time (no extra outer time; a timeless stream raises) [verified]. By default the body's own state Inputs are used, so the outer model lists them as inputs.
- Output: the points as a new outer time, `(N, *state_time, *dim)` per state: state `z.last()` with `dim=2` gives `T=(N, 1), D=(2,)`, arrays `(B, N, 1, 2)` [verified], with index 0 equal to the initial state. `a, b = OdeNet(...)` unpacks several states.
- `method`: a fixed tableau (`euler`, `midpoint`, `heun`, `rk4`) applies `steps` substeps per reported interval, unrolled; use it for training and export. `"dopri5"` is adaptive (controlled by `rtol`, `atol`, `max_steps`): not ONNX-exportable, not reverse-differentiable on JAX, meant for inference. To switch a built model: `model.model.get_layer(<OdeNet name>).set_method("dopri5")`, which is why `name=` should be set.
- Hybrid events: `event` names an output of `f` with dim `(1,)` and time `()` or `(1,)`, positive before the event and negative after. `reset` gives `{state: output}` with the post-event values of every state. Events need a fixed tableau, and the event time is differentiable.
- **Training with a DataLoader** [verified, recipe R5]: the state inputs have no sequence, so the loader aligns them with the **last** element of the `t` and target sequences (§9). Shift the seed column by N-1 so that it equals the first point of each window: `seed = np.vstack([np.repeat(ref[:1], N-1, axis=0), ref[:1-N]])`. See recipe R5.
- Supports `save` / `load` (the method chosen with `set_method` included) and `export_keras`.

---

## 12. Model composition

- `block = Modely("b", inputs=[a, c], outputs=[o1, o2]).build()`. Then `o1s, o2s = block([stream_for_a, stream_for_c])`. The streams go in the order of `block.inputs`, exactly one per input (another count raises `ValueError`), and a single output returns a single stream.
- A block keeps its shapes: each stream must have the shape the block's graph reads that input with (its window, with its sequence; a dynamic axis matches any length), else `ValueError` ("takes a stream of that shape") - nothing is reshaped, so a block reading `x.last()` refuses `a.last(seq=3)` or `a.sw(3)` [verified]. An `Input` can be passed whole, `block([a])`: the block's own windows read it, and it must have the block input's dim and sequence [verified].
- The block brings its weights along (trained weights are reused while input shapes match), and training the outer model also trains the block. To freeze it: `block.freeze()`.
- Calling the same block twice shares its weights (§1.22 covers naming).
- `EquationLearner` uses composition internally.
- `Differentiate(respect_to=Input)` inside a block does not work once the block's inputs are rebound. Put the derivative in the outer model instead, where the Input is declared.
- `model.flatten()` returns an equivalent model with everything inlined.

---

## 13. Inference, validation and export

**Inference arrays**: float32 NumPy, shape `(batch, *time, *dim)`, where `time` is the union of the windows the model reads, with its sequence (`x.shape.time` after build); `(batch, *dim)` for a never-windowed Input. Results come back laid out the same. A single sample keeps a batch axis of 1.

**`validate(val_data, out_dir=None, show=False, history=None, verbose=True)`**
- Scores every objective as training does: same forward pass (`training=False`), same target alignment, same padding mask. `metrics["loss"]` equals `train()`'s `val_<name>_loss` on the same data (`seq_weights` and masking included, `gain` not applied) [verified]. Prints a summary (`verbose=False` skips it).
- Metrics per objective (`result[name].metrics`, or `result.metrics()` for everything): `loss rmse mae max_error bias std_error nrmse_pct fit_pct r2 vaf_pct correlation non_finite`.
  - With e = ŷ − y: `fit_pct` = 100(1 − ‖e‖/‖y−ȳ‖), `r2` = 1 − Σe²/Σ(y−ȳ)², `vaf_pct` = 100(1 − var(e)/var(y)), `nrmse_pct` = 100·RMSE/range(y); `rmse² = bias² + std_error²`.
  - Per channel (each dim element of the signal): `result[name].channel_metrics[key]` is an array, one value per channel. The whole-signal absolute metrics pool every channel; the relative ones are the mean over channels [verified].
  - Relative metrics are NaN for a channel whose target is constant, and that channel is left out of the mean. Correlation is 0 for a constant prediction. Any NaN/inf error → `non_finite` count and NaN metrics.
  - Padded rollout steps (`seq_length="full"`, simulations of different lengths) are left out of every metric: the padded set scores the same as each simulation alone [verified]. `score.points` / `score.padded` count the scored and the left-out points. The loss keeps training's definition (padded steps add 0 but count in the mean).
  - `score.step_rmse`: `(steps, channels)` RMSE along the outer time for a signal with ≥ 2 time axes (a rollout), else `None`.
- `out_dir` writes PNGs (prediction vs target over the scored points, error, parity, histogram, RMSE per rollout step when there is one, plus loss curves when `history` is given), and `result.figures` lists them.

**Export support**

| Model contains | `save`/`load` | `export_keras` | ONNX TF | ONNX Torch | ONNX JAX |
|---|---|---|---|---|---|
| Plain layers, `Parameter`, `Constant`, `Fir`, `Linear`, `LocalModel`, `EquationLearner`, `IntegrateStep`, `Derivative`, `rollback` | yes | yes | yes | yes | no |
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
| `expected shape=(None, ...), found shape=(...)` at inference | An array laid out in the old `(batch, *dim, time)` order, or without its batch axis, e.g. `loader[i]` (§1.24; a missing batch axis may also fail inside a layer, or be misread) | Lay it out `(batch, *time, *dim)`; add a batch axis: `v[None]`. |
| `used as a block it takes one stream for each, in that order` | A block called with fewer or more streams than its inputs | Pass one stream per `block.inputs`. |
| `so used as a block it takes a stream of that shape` | A block called on a stream of another window, sequence or dim | Pass a stream of the shape the block reads (or the Input whole), or declare the block on the shape it is used with. |
| `... is missing required inputs: [...]` (DataLoader) | Data dict lacks an input, often a target-only one | Add the column, or map it with `format`. |
| `has only N samples, but the model requires at least M` | Simulation shorter than windows plus sequences | Shorter windows or seq, more data, or `on_short="skip"`. |
| `Some inputs have undefined sequence length` | `seq=-1` without `seq_length` | `DataLoader(..., seq_length=H)` or `"full"`. |
| `seq_length='full' only resolves the outermost one` | A dynamic length that is not the outermost time axis, e.g. `seq=(3, -1)` | Make the dynamic axis outermost, or pass `seq_length=H`. |
| `roll stream ... must produce one temporal sample with shape ...` | Rollback stream has an inner time > 1, a wrong dim, or other outer time | Feed back a one-sample stream with the input's dim and outer time. |
| `Loop cannot determine the rollout length` | No initial value or input has one more outer time than its body input | Pass `length=H`, or feed a rollout (`x.last(seq=H)`). |
| `Loop rollout inputs declare different lengths` | Exogenous rollout streams with different outer-time lengths | Pass `length=` or make the lengths equal. |
| `Loop input '...' expects time ... or one more outer time for the rollout` | An exogenous stream whose time is neither the body input's nor one outer axis more (e.g. a timeless one) | Use a window like the body's, with `seq=H` for a rollout. |
| `Model '...' reads no Input` at build | The model reads no Input, e.g. a Loop seeded and driven only by numbers | Seed or drive it from an `Input`. |
| `Loop init must be a dict` | Scalar or stream passed as `init` | `Loop(..., init={inp1: s1, inp2: s2})(...)`. |
| `the output must be a single sample, the next one of the input` | Loop callback output with an inner time longer than 1 | Feed back one sample, e.g. `TimeSelect(-1)(...)` of the window. |
| `the input window reads ... future samples` | Loop callback input read through a window with future samples (`sw([p, f])`, `next()`) somewhere in the body | Read the fed-back input only up to its current sample. |
| `the outer time axes, the sequence, must be the same` | Loop callback output with other outer time than its input, e.g. a collected inner trajectory, or a sample selected with `keep_dim=False` (time `()`) fed back into a window | Close the callback on `TimeSelect(-1, axis=0, keep_dim=False)(...)`; keep a selected sample's axis (the default) to feed it back; or make the sequences equal. |
| `has two different outputs named ...` | A block called twice exposes same-named outputs | Wrap each call's output in a distinctly named `Output`. |
| `the differentiated relation does not depend on input ...` | `Differentiate` w.r.t. an Input inside a composed block, or no real dependency | Move the derivative to the model that declares the Input. |
| `Minimizer '...' compares '...' of shape ... with '...' of shape ...` in `train` / `validate` | The source and target of an objective have different shapes, e.g. a raw timeless Input target against a `last()` source | Make them match: use a window as target. |
| `streams with different time axes cannot be combined` | Arithmetic between different times, e.g. `x.sw(3) + x.last()` or `x.last(seq=4) + y.last()` | Select or window to the same time axes, or use a timeless operand. |
| `dim ... and dim ... cannot be combined` | Dims neither equal nor one a scalar, e.g. `(3,)` with `(2, 3)` | Make them equal, or reduce one to a scalar. |
| `cannot be combined: each would be broadcast` | A timeless non-scalar (e.g. a `(2, 3)` Constant) with the window of a scalar | Give the timeless operand the window's time, or the window the operand's dim. |
| `the sequence of an Input is set only once` | A later window with `seq=` (any value) | Omit `seq` on later windows, or declare another Input. |
| `reads the Input ... whole, and the graph also reads it through windows` | One Input read raw and windowed | Read it through a window there too (`x.last()`). |
| `idx ... out of bounds for dim axis` | `Select` index out of range | Use an index within the axis size. |
| `TypeError: bad operand type for unary -` | `-stream` | `-1.0 * stream` or `Negative()(stream)`. |
| `OdeNet integrates an autonomous field, but [...] are body inputs that are not states` | Exogenous input in the field | Use `Ode` + `Loop` for driven systems. |
| Softmax output sums to one over the whole sample, not per feature vector | The default normalizes the whole sample (§1.15) | `Softmax(axis=-1)`. |

---

## 15. Recipes (all run as written; TensorFlow backend)

### R1. One-step ARX / FIR identification with train/val split
```python
import numpy as np
from nnodely import DataLoader, Fir, Input, Modely, Output, set_seed
set_seed(0)
rng = np.random.default_rng(0)
N = 2000
u_data = rng.uniform(-1, 1, N); y_data = np.zeros(N)
for t in range(N - 1):
    y_data[t + 1] = 0.9 * y_data[t] - 0.2 * y_data[t - 1] + 0.5 * u_data[t]

y, u = Input("y"), Input("u")
y_next = Output("y_next", Fir(features=1)([y.sw(2)]) + Fir(features=1)([u.sw(2)]))
model = Modely("arx", inputs=[y, u], outputs=[y_next])
model.minimize("one_step", y_next, y.next(), loss="mse")
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
from nnodely import Input, Modely, Ode, Output, Parameter, Sin, DataLoader
dt = 0.02
theta, omega, torque = Input("theta"), Input("omega"), Input("torque")
g_over_l = Parameter("g_over_l", value=10.0)     # initial guesses; defined OUTSIDE f
damping = Parameter("damping", value=0.0)

def pendulum(th_, om_, tq_):                     # returns d(theta)/dt, d(omega)/dt
    return om_, tq_ - g_over_l * Sin()(th_) - damping * om_

theta_next, omega_next = Ode(pendulum, [theta.last(), omega.last()], dt=dt,
                             method="rk4", args=(torque.last(),))
theta_out, omega_out = Output("theta_next", theta_next), Output("omega_next", omega_next)
pend = Modely("pendulum", inputs=[theta, omega, torque], outputs=[theta_out, omega_out])
pend.minimize("theta_err", theta_out, theta.next())
pend.minimize("omega_err", omega_out, omega.next())
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
state, force = Input("state", dim=2), Input("force")
step_out = Output("state_next",
    Linear(features=2, bias=False)([state.last()])
    + Linear(features=2, bias=False)([force.last()]))
body = Modely("step", inputs=[state, force], outputs=[step_out]).build()
# 2) outer: sequences of H steps
s_seq, f_seq = Input("s_seq", dim=2), Input("f_seq")
traj = Output("traj", Loop(f=body, callback={state: step_out}, init={state: s_seq.last(seq=H)})(
                           {force: f_seq.last(seq=H)}))      # traj: T=(H, 1), D=(2,)
sim = Modely("simulator", inputs=[s_seq, f_seq], outputs=[traj])
sim.minimize("sim_error", traj, Input("s_target", dim=2).last(seq=H))
sim.build()
# 3) data: target is the state shifted one step ahead of the seed window
states = np.stack([th, om], axis=-1)             # (T, 2)
data = DataLoader(sim, source={"s_seq": states[:-1], "f_seq": tq[:-1], "s_target": states[1:]}, step=5)
sim.train(data, epochs=5, batch_size=32, lr=1e-2, printer=None)
# body weights are trained too; body(...) can now be used for one-step prediction
```
To use a physical step inside the body, build the body from `Ode(...)` as in R2, feeding back both states (`callback={"theta": "theta_next", "omega": "omega_next"}`, `init={"theta": th_seq.last(seq=H), "omega": om_seq.last(seq=H)}`). For whole simulations of different lengths use windows with `seq=-1`, `Loop(...)` and `DataLoader(..., seq_length="full")`.

### R4. Physics-informed network (Differentiate w.r.t. an Input + residual)
```python
from nnodely import Differentiate, Input, Linear, Modely, Output, Tanh, DataLoader
t_in = Input("t")
u_hat = Linear(features=1)([Tanh()([Linear(features=16)([t_in.last()])])])
du_dt = Differentiate(order=1, respect_to=t_in)(u_hat)
u_out = Output("u", u_hat)
pinn = Modely("decay", inputs=[t_in], outputs=[u_out])
pinn.minimize("data", u_out, Input("u_meas").last())
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
z = Input("z", dim=2)
field = Modely("field", inputs=[z], outputs=[Output("dz",
    Linear(features=2)([Tanh()([Linear(features=32)([z.last()])])]))]).build()
t_rep = Input("t_rep")
z_traj = Output("z_traj", OdeNet(f=field, states={z: "dz"}, t=t_rep.last(seq=N), method="rk4", steps=2, name="node"))
node = Modely("neural_ode", inputs=[z, t_rep], outputs=[z_traj])           # z_traj: T=(N, 1), D=(2,)
node.minimize("traj", z_traj, Input("z_ref", dim=2).last(seq=N))
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
trq, gear = Input("trq"), Input("gear")
mu = Fuzzify(centers=[1.0, 2.0, 3.0, 4.0], function="Triangular")([gear.last()])   # T=(1,), D=(4,)
engine = LocalModel(lambda i: Fir(features=1))([trq.sw(10)], [mu])              # T=(1,), D=(1,)
m = Modely("sched", inputs=[trq, gear], outputs=[Output("force", engine)]).build()
m({"trq": np.ones((1, 10, 1)), "gear": np.full((1, 1, 1), 2.5)})                 # (batch, time, dim)
# one 10-tap FIR per gear cell, each with weights of its own
```

### R7. Composition (reuse a trained block)
```python
from nnodely import Fir, Input, Linear, Modely, Output, ReLU
sig = Input("sig")
smoother = Modely("smoother", inputs=[sig], outputs=[Output("smooth", Fir(features=1)([sig.sw(10)]))]).build()
raw = Input("raw")
cmd = Output("cmd", ReLU()([Linear(features=1)([smoother([raw.sw(10)])])]))
controller = Modely("controller", inputs=[raw], outputs=[cmd]).build()
controller({"raw": np.ones((1, 10, 1))})
```

### R8. k-step-ahead training with rollback
```python
from nnodely import Fir, Input, Modely, Output, DataLoader
K = 5
x, F = Input("x"), Input("F")
x_pred = Output("x_pred", Fir(features=1)([x.sw(5)]) + Fir(features=1)([F.last()]))
rb = Modely("rb", inputs=[x, F], outputs=[x_pred])
rb.rollback({x: x_pred}, steps=K)
rb.minimize("ahead", x_pred, Input("x_ahead").last())
rb.build()
data = DataLoader(rb, source={"x": xs[:-K], "F": Fs[:-K], "x_ahead": xs[K:]})
rb.train(data, epochs=20, batch_size=64, lr=1e-2, printer=None)
```

### R9. Mechanical model: acceleration → velocity → position in one pass (no rollout)
```python
from nnodely import Input, IntegrateStep, Linear, Modely, Output
dt, n = 0.01, 50
F, v0, x0 = Input("F"), Input("v0"), Input("x0")
acc = Linear(features=1)([F.sw(n)])                   # T=(n,), D=(1,): per-sample force -> acceleration
vel = IntegrateStep(dt=dt, init=v0.last())(acc)                # T=(n,), D=(1,)
pos = IntegrateStep(dt=dt, init=x0.last())(vel)                # T=(n,), D=(1,)
m = Modely("mech", inputs=[F, v0, x0], outputs=[Output("pos", pos)])
m.minimize("pos_err", m.outputs[0], Input("x_meas").sw(n))
m.build()
```
Caution: `v0` and `x0` are read at the current sample, which is the **end** of `F.sw(n)`. Provide their data columns shifted by `n` so that they hold the state just before the window.

---

## 16. Custom layers

Pattern: a symbolic `Layer` subclass returns a serializable Keras layer from `build_layer()`. The output shape is inferred automatically by running the Keras layer on zeros laid out `(1, *time, *dim)`: the result keeps the inputs' number of time axes (at most) and the remaining axes are its dim; a dynamic time axis (`seq=-1`) is probed with one step and declared dynamic again when the layer keeps the number of time axes. Override `output_shape(self, *inputs) -> Shape` (`nnodely.core.shape.Shape(dim=..., time=...)`) only when that cannot work (a layer that adds or removes time axes, value-dependent shapes). The saved config is derived from the arguments passed to `super().__init__`, so `get_config` needs no override unless an argument is not JSON-serializable as given (`Fir` and `Linear` save their weights as the Parameter nodes they are, so their config only says whether there is a bias).
```python
import keras
from nnodely.core.layer import Layer

@keras.saving.register_keras_serializable(package="my_project")
class SquashImpl(keras.layers.Layer):
    def __init__(self, gain=1.0, **kwargs):
        super().__init__(**kwargs); self.gain = float(gain)
    def call(self, x):                      # x: [batch, *time, *dim]; list of tensors if multi-input
        return keras.ops.tanh(self.gain * x)
    def get_config(self):
        return {**super().get_config(), "gain": self.gain}

class Squash(Layer):
    def __init__(self, gain=1.0, name=None):
        self.gain = float(gain)
        super().__init__(name=name, gain=self.gain)   # kwargs -> self._properties (re-used when called, and saved)
    def build_layer(self):
        return SquashImpl(gain=self.gain, name=self.name)

y = Squash(gain=2.0)([x.sw(3)])
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
- [ ] Layers receive windows (`.last()`, `.sw()`), not raw Inputs; sequences are declared with `seq=` on the first window, never on `Input`.
- [ ] The shape of every stream is written down (§3) as `(*time, *dim)`. `Fir` returns one sample (inner time 1); `Linear` keeps time and maps the whole dim; combined streams have the same time axes (or one is timeless) and equal or scalar dims.
- [ ] Objectives are declared before `build()`, each with its own name. Targets are Input windows, streams computed from data, numbers or `None`.
- [ ] Physical constants are `Parameter(value=guess)` or `Constant`, created outside `Ode` functions.
- [ ] No `-stream`; `-1.0 * stream` instead.
- [ ] Objectives go on the outer model, not on `Loop` / `Roll` / `OdeNet` bodies. Loop targets are shifted one step from the seed window. OdeNet seeds are shifted by N-1.
- [ ] `train(...)` has explicit `epochs`, `batch_size` and `lr`; `printer=None` for scripts and tests.
- [ ] Inference and data arrays are laid out `(batch, *time, *dim)`, features last, with a batch axis (`loader[i]` needs `v[None]`).
- [ ] Export format matches the model and backend (§13): no ONNX on JAX or for `dopri5`.
- [ ] Results are converted with `keras.ops.convert_to_numpy(...)` before NumPy or matplotlib use.

---

## 18. Internals and repository map (for editing the library itself)

```
src/nnodely/
  __init__.py            public exports (see §4), __version__, the missing-backend check
  core/shape.py          Shape(dim, time), the only module that knows the layout (batch, *time, *dim): axes helpers,
                         Shape.broadcast and align (arithmetic), BATCH_AXIS, OUTER_TIME_AXIS = 1, tensor helpers
  core/stream.py         Node, Stream (operator overloads, literal→Constant cache), NODE_REGISTRY, NODE_NAMES
  core/layer.py          Layer base (symbolic call → new node sharing its _source; tensor call → build_layer();
                         get_config from the constructor arguments), BinaryOp Add/Subtract/Multiply/Divide/Power, Identity
  core/modely.py         Modely (build, minimize, train, validate, rollback, save/load, export wrappers), ModelCall
  core/minimizer.py      objectives: checks, resolution against the built graph, MinimizerModel (the training model)
  core/export.py         export_keras, import_keras, export_onnx, validate_onnx
  core/dag.py            name registry (next_name), flatten (inline ModelCalls per call scope), toposort, stream helpers
  core/dataloader.py     DataLoader (read → per-simulation arrays → sliding windows → alignment, mask, normalize)
  core/registry.py       ModelSerializer: model.json (+ model.weights.h5) via get_config/from_config; bodies in sub-folders
  core/validation.py     validate, score_signal, SignalScore, ValidationResult
  layers/input.py        Input (+ sw/last/next → SampleWindow)
  layers/partitions.py   SampleWindow, Select, Range, Concatenate (dim axes); TimeSelect, TimeRange, TimeConcatenate, TimeStack, TimeReverse (time axes)
  layers/{linear,fir,localmodel,fuzzify,function,interpolation,equationlearner,batchnorm,activations,arithmetic,trigonometric}.py   function.py: KerasFunction
  layers/{parameter,output}.py   parameter.py: _Value, Parameter, Constant
  layers/derivative.py   Differentiate (autodiff dispatch per backend), Derivative (Savitzky–Golay banded operator)
  layers/integrate.py    IntegrateStep (cumulative quadrature operator matrix), Integrate (IntegrateStep + Loop block)
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
- Parameters and Constants have no predecessors. They are called with an anchor tensor of the graph and broadcast their value to its batch, so arithmetic treats them as any stream.
- `build()` makes two Keras models over the same layers: `model.model` for training, whose outputs include the sources and targets of every minimizer (targets computed from data by the graph itself), and `model.inference_model`, with the declared outputs and only the inputs they read. History and printers use the minimizer names.
