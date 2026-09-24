# nnodely — reference for AI code generation

Target reader: an LLM or coding agent that has to write, debug or explain Python code
using **nnodely 2.0.0** (Model-Structured Neural Networks on Keras 3; backends TensorFlow,
PyTorch, JAX; Python 3.10–3.13). The file is dense on purpose, and exact signatures are
given. Items marked **[verified]** were checked by running code against the source tree
(TensorFlow backend, Keras 3.12.1). All recipes in §15 run as written.

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
2. **`Input(name, *, dim=None, seq=None)`**: `dim` and `seq` are keyword-only, so `Input("x", 3)` is an error.
3. **Feed layers with windows, not raw Inputs:** use `x.last()`, `x.sw(n)`, `x.sw([p, f])`, `x.next()`. A raw `Input` used as a stream carries the *union* of all its windows (time = past+future) [verified]. Raw Inputs are fine as `minimize` targets and as sequence sources for `Loop` / `OdeNet`.
4. **Stream operators:** `+ - * / **` work between streams and Python numbers; numbers become `Constant`s. Unary minus is not supported: `-x` raises `TypeError` [verified], so write `-1.0 * x` or `Negative()(x)`. Also unsupported: `2 ** x`, `@`, indexing `x[0]`, comparisons. For indexing use `Select`, `Range`, `TimeSelect` or `TimeRange`.
5. **Objective targets** (`minimize(name, source, target, loss)`). `target` must be one of:
   (a) an `Input`, or a window of one, used **untransformed**;
   (b) a Python `float`;
   (c) `None`, which drives `source` to 0.
   A *transformed* target does not work. A single-predecessor chain such as `Sin()(y.last())` is **silently replaced by the raw data column** [verified: loss was computed against `y`, not `sin(y)`]. A multi-predecessor chain such as `y.last()*2` raises `Training target '...' must be present in the dataset or be a constant value` [verified]. For a computed target use the residual form `minimize(name, source - target_expr)`, which gives the same MSE [verified].
6. **One minimizer per source stream.** A second `minimize` with the same `source` silently replaces the first, because losses are keyed by source name [verified]. To put two losses on the same stream, wrap it in two `Output`s with different names.
7. **There are no per-loss weights.** Scale the residual instead: `minimize("w", (out - y.last()) * 0.3)`.
8. **Float targets must be `float`:** write `target=0.0`, not `target=0`. An int is not converted.
9. **Names are identities.** Input and Output names must be unique within a model: they are the keys of data dicts, results and history. Two layer objects created with the **same explicit `name=`** share one Keras layer, and therefore its weights [verified]. Pass `name=` only to look a layer up later (`model.model.get_layer(name)`) or to share weights on purpose. Auto names are `Linear1`, `Fir2`, ... from a global counter.
10. **Weight sharing means reusing the same layer object:** `lin = Linear(out_features=4)`, then `lin([a])` and `lin([b])` share weights. A `Linear(...)` created *inside* a function that runs several times (for example the `f` of `Ode`, which runs once per RK stage) creates new weights on every call. Create weight-bearing layers and `Parameter`s outside `f` and call them inside.
11. **Body models** passed to `Loop`, `Roll`, `OdeNet` or `Gradient` must not have `minimize` objectives with Input targets. The target input becomes a required body input, and the rollout fails at call time with `Missing data for input "..."` [verified]. Build a clean body model with no objectives and put the objectives on the outer model.
12. **Once a model has an objective with an Input target, inference needs that key too:** `model(...)` raises `KeyError` without it [verified]. Pass an array of the right shape (zeros are fine).
13. **`validate()` fails on objectives whose target is `None` or a float**, with `ValueError: ... compares ... with 'ConstantN' ... must match` [verified]. Training is unaffected. Before `validate`, call `model.remove_minimizer(name)` on those objectives.
14. **`save()` / `load()`** (nnodely JSON format) **does not support models containing `Loop`, `Roll`, `OdeNet` or `Gradient`**: `load` raises `TypeError` [verified]. Use `export_keras` / `import_keras` for those. `Modely.rollback` models, `Derivative`, `Integrate`, `LocalModel` and `EquationLearner` do round-trip [verified].
15. **`Softmax()` defaults to `axis=-1`, which is the time or seq axis** of the runtime tensor `[batch, *dim, time, *seq]`. For a softmax over features use `Softmax(axis=1)` [verified]. `BatchNorm(axis=1)` already normalizes per feature.
16. **`Derivative(respect_to=x)`**: `x` must be the `Input` object, not a window of it. The result has the shape of x's **whole** window. It does not work inside a model that is used as a composed block (§12).
17. **`train()` defaults to `batch_size=1` and `epochs=10`.** Always pass both (typical batch sizes are 32–256).
18. **Set `KERAS_BACKEND` before the first import** of `keras` or `nnodely`. With no value set, the backend is TensorFlow.
19. **`Parameter` initializes to `random_normal` when no value is given.** For physical constants pass `value=<initial guess>`.
20. **`dt` is always explicit** for `Integrate` and time-`Derivative`; nothing is inferred from data. Windows are counted in samples, not seconds.
21. **Using a `Loop` node directly** gives the trajectory of its **first callback output**. Unpack it (`a, b = Loop(...)`) to get every body output, in `f.outputs` order [verified].
22. **A model called twice as a block** returns outputs with the same names both times. Combine them, or wrap each in an `Output` with a distinct name; otherwise `build` raises `two different outputs named ...`.
23. **`DataLoader(..., delimiter=, header=)` are stored but not passed to `pd.read_csv`.** CSV files are read with pandas defaults (comma separator, header inferred).

---

## 2. Mapping equations to blocks

| Physical or modeling concept | nnodely construct |
|---|---|
| Measured signal `x(t)` | `x = Input("x")`; vector signal: `Input("x", dim=3)` |
| Current sample `x[t]` | `x.last()` |
| Past n samples `x[t-n+1..t]` | `x.sw(n)` |
| Next sample `x[t+1]` (typical one-step target) | `x.next()` |
| Linear combination of past samples (ARX / FIR term) | `Fir(out_features=1)([x.sw(n)])` |
| Static affine map / MLP layer on features | `Linear(out_features=k)([s])` plus an activation |
| Unknown physical constant (mass, stiffness, g/l) | `Parameter("m", value=guess)` |
| Known constant | `Constant("g", value=9.81)`, or just a number in arithmetic |
| Nonlinearity `sin`, `exp`, `abs`, saturation | `Sin()(s)`, `Exp()(s)`, `Abs()(s)`, `Clamp(min, max)(s)` |
| Lookup table / characteristic curve (fixed) | `Interpolation(x_points, y_points)(s)` |
| Gain scheduling / operating regimes | `Fuzzify(centers)` → `LocalModel()([x.sw(n)], [membership])` |
| Symbolic regression of an unknown law | `EquationLearner(["sin", "multiply", ...], linear_out=Linear(1))` |
| `du/dx` exact (PINN residual, Sobolev) | `Derivative(order=1, respect_to=x_input)(u)` |
| `dx/dt` from a sampled window | `Derivative(respect_to=dt, init=..., window=...)(x.sw(n))` |
| `∫ rate dt` over a window / one Euler state update | `Integrate(solver="euler", dt=dt, init=state.last())(rate)` |
| One explicit RK step of `dx/dt = f(x, u, θ)` | `Ode(f, [states], dt, method="rk4", args=(...))` |
| Learned vector field integrated over reported times | `OdeNet(f=field_model, states={...}, t=times_stream)` |
| Gradient of a learned scalar potential `∂H/∂x` | `Gradient(H_model, x_stream)` (not exported at top level) |
| Multi-step / simulation-error training | `Loop` (sequence rollout) or `Modely.rollback` (k-step ahead) |
| Physical law as a soft constraint | `model.minimize("law", residual_stream)` (no target) |
| Reuse a trained sub-model | `sub_model([stream_a, stream_b])` (composition) |

---

## 3. Shapes and windows

**Semantic shape** (no batch axis): `(*dim, time, *seq)`.
- `dim`: feature axes. Default `(1,)`; `dim=3` gives `(3,)`; `dim=(3,2)` gives a matrix per sample.
- `time`: number of consecutive samples in the stream's window.
- `seq`: optional sequence or rollout axes, used by `Loop`, `OdeNet` and the `DataLoader`. `seq=-1` means dynamic (`None` internally).

**Runtime tensor**: `[batch, *dim, time, *seq]`. The time axis index is `1 + len(dim)`. Every data array passed to a model or produced by the `DataLoader` follows this layout.

Stream attributes: `s.shape` (tuple-like, prints e.g. `(3, 10)`), `s.dim`, `s.time`, `s.seq`, `s.name`, `s.shape.tuple`, `s.shape.dim_rank`.

**Windows on an `Input`:**

| Call | Samples | `time` |
|---|---|---|
| `x.sw(n)` | `x[t-n+1] … x[t]` | n |
| `x.sw([p, f])` | `x[t-p+1] … x[t], x[t+1] … x[t+f]` | p+f |
| `x.last()` | `x[t]` (= `sw(1)`) | 1 |
| `x.next()` | `x[t+1]` (= `sw([0, 1])`) | 1 |

The Input's own window is the union of all requested windows: `past = max p`, `future = max f`, and `Input.shape.time = past + future` [verified: `u.sw(10); u.sw([2,3])` gives `u.shape == (3, 13)`]. Windows are ordered oldest first; index `-1` along time is the newest sample.

**Shape rules of common layers** [verified]. Input `v = Input("v", dim=3)`, window `v.sw(4)` of shape `(3, 4)`:

| Expression | Output shape |
|---|---|
| `Fir(out_features=2)([v.sw(4)])` | `(2, 1)`: flattens dim×time (and seq), one output sample |
| `Linear(out_features=5)([v.sw(4)])` | `(5, 4)`: per-sample map on the first dim axis, time kept |
| `Sum()([v.sw(4)])` | `(1, 4)`: sums dim axes (keepdims) |
| `Select(1)([v.sw(4)])` | `(1, 4)` |
| `Range(0, 2)([v.sw(4)])` | `(2, 4)` |
| `TimeSelect(-1)([v.sw(4)])` | `(3, 1)` |
| `TimeRange(0, 2)([v.sw(4)])` | `(3, 2)` |
| `Concatenate()([v.last(), w_dim2.last()])` | `(5, 1)`: along dim axis 0 |
| `TimeConcatenate()([v.sw(2), v.sw(3)])` | `(3, 5)` |
| `Fuzzify(centers=[0,1,2])([g.last()])` | `(3, 1)`: one feature per center |
| `Fuzzify(centers=[0,1,2])([g.sw(4)])` | `(3, 4)` |
| `LocalModel(out_features=1)([t.sw(10)], [Fuzzify(...)([g.last()])])` | `(1, 1)` |
| `Derivative(respect_to=dt)(x.sw(n))` | `(1, n)`: window length preserved |
| `Derivative(respect_to=t_input)(u)` | shape of `t_input`'s full window |
| `Integrate(dt=dt)(a.sw(n))` | `(1, n)` |
| Elementwise layers (activations, trig, `Exp`, `Clamp`, `Interpolation`, `BatchNorm`) | unchanged |

**Arithmetic broadcasting.** `+ - *` align ranks by appending trailing axes and give batch-free operands (Constants, Parameters) a batch axis. After that, NumPy broadcasting applies, so `(3,1) * (1,1)`, `(1,5) + (1,1)` and `(1,1,1) + (1,1,H)` all work. `/` and `**` use plain NumPy broadcasting, aligned from the right. Do not mix operands with different seq ranks in `/` or `**`; scalar constants and parameters are fine.

**Parameter / Constant shapes** [verified]: `value=2.0` gives `(1,1)`; `value=[1,2,3]` gives `(3,1)`; `value=[[1],[2],[3]]` gives `(3,1)`; `value=[[1,2]]` gives `(1,2)`; `dim=3` gives `(3,1)`; `dim=3, time=2` gives `(3,2)`; an array of shape `(d,t,s)` gives `(d,t,s)`.

---

## 4. Public API and imports

```python
import os; os.environ["KERAS_BACKEND"] = "torch"   # or "tensorflow" / "jax"; BEFORE imports
from nnodely import (
    Modely, DataLoader, Input, Output, Parameter, Constant, set_seed, get_seed,
    # layers with weights
    Linear, Fir, LocalModel, EquationLearner, BatchNorm,
    # fuzzy / lookup
    Fuzzify, Interpolation,
    # feature axis / time axis
    Select, Range, Concatenate, Sum, TimeSelect, TimeRange, TimeConcatenate,
    # calculus / recurrence
    Derivative, Integrate, Ode, OdeNet, Loop, Roll,
    # activations
    ReLU, LeakyReLU, ELU, PReLU, Sigmoid, Tanh, Softmax, Swish, GELU, Softplus,
    # math
    Sin, Cos, Tan, Asin, Acos, Atan, Exp, Log, Log10, Sqrt, Abs, Floor, Ceil,
    Deg2Rad, Sign, Negative, Clamp,
    # training printers (callbacks)
    TinyPrinter, LegacyPrinter, NNodelyPrinter,
)
# Not exported at top level:
from nnodely.layers.gradient import Gradient                  # d(scalar model)/d(stream), autodiff
from nnodely.core.layer import Layer, Identity                # base class for custom layers
from nnodely.core.validation import ValidationResult
# Experimental, untracked in git at time of writing (may change or be missing):
from nnodely.layers.porthamiltonian import PortHamiltonian
```

Install: `pip install "nnodely[torch]"` (or `[tensorflow]` / `[jax]`); add `onnx` for ONNX export (`"nnodely[torch,onnx]"`).

---

## 5. Core objects

### Input
`Input(name: str, *, dim: int | tuple | None = None, seq: int | tuple[int, ...] | None = None)`
- Methods: `.sw(n | [p, f])`, `.last()`, `.next()`. Each returns a new `SampleWindow` stream and widens the Input's window.
- Attributes: `.past`, `.future`, `.shape`, `.dim`, `.seq`, `.name`.
- `seq=N` gives a fixed-length sequence (trajectory) input. `seq=-1` gives a dynamic length, resolved by `DataLoader(seq_length=...)`; only one dynamic seq axis per input is allowed.
- An Input used only as a target (for example `Input("y_meas").last()`) is still a model input: data must provide it (§1.12).

### Output
`Output(name: str, stream: Stream)`: gives a stream a public name. It is the key in results, history and `rollback` / `Loop` mappings.

### Parameter (trainable)
`Parameter(name=None, *, value=None, initializer="random_normal", seq=None, time=None, dim=None)`
- The shape comes from `value` if given (see §3), otherwise from `dim` / `time` / `seq`.
- `initializer` is any Keras initializer name or object.
- After build: `.param` (a Keras variable) and `.value_numpy`.
- It is batch-free and broadcasts in arithmetic.

### Constant (fixed)
`Constant(name=None, *, value, dim=None)`. After build: `.constant` and `.value_numpy`. Numbers in arithmetic become cached anonymous Constants.

### Layer calling convention
Configure first, then call on a stream or a list of streams:
- `Linear(out_features=4)([s])` and `Linear(out_features=4)(s)` are both accepted.
- Multi-input layers take a list: `Concatenate()([a, b])`.
- `LocalModel` takes two lists: `LocalModel(...)(inputs, activations)`.

After build, weights are available on the *returned stream node*: `node.kernel`, `node.bias` (Keras variables; `.assign(np.array(...))` works).
- `Fir.kernel` has shape `(prod(dim)*time, out_features)`; rows follow C order over `(*dim, time)`, so for `dim=1` row k is window sample k, oldest first.
- `Linear.kernel` has shape `(in_features, out_features)`.
- `LocalModel.kernel` has shape `[cells, features, out_features]` per input.

---

## 6. Layer reference

Legend: **in → out** gives semantic shapes. `d` = dim, `T` = time, `S` = seq.

### 6.1 With weights
| Signature | in → out | Notes |
|---|---|---|
| `Linear(out_features=1, use_bias=True, name=None, initializer="glorot_uniform", bias_initializer="glorot_uniform")` | `(d, T, *S)` → `(out, T, *S)` | Dense on the first dim axis, applied per time/seq element. The bias initializer is glorot, not zeros. |
| `Fir(out_features, use_bias=True, name=None)` | anything → `(out, 1)` | Flattens **all** of dim×time×seq, then Dense. `out_features` is required. The seq axis is lost, so do not use it on seq streams. |
| `LocalModel(out_features=1, use_bias=True, initializer="glorot_uniform", bias_initializer="zeros", input_function=None, output_function=None, name=None)` called as `(inputs: list, activations: list)` | `x_j: (d, T)`, `a_j: (cells, 1)` → `(out, 1)` | Computes `Σ_j Σ_i a_j,i (W_j,i x_j + b_j,i)`. Each activation must be `(cells,1)`, e.g. `Fuzzify(...)([g.last()])`. Inputs must not carry seq. With `input_function` / `output_function` (callables receiving a **list** `[stream]`), one explicit cell is built per membership: `Σ_i output_function(input_function([x]) * a_i)`, and `out_features` / `use_bias` are then unused. Create layers inside those callables so that each cell gets its own weights. |
| `EquationLearner(functions: list, *, linear_in: Linear \| None = None, linear_out: Linear \| None = None, name=None)` called on `stream` or `[streams]` | `(d, T)` → `(n_functions, T)`, or `linear_out`'s output | `functions` items can be a name (`identity sin cos tan asin acos atan relu leaky_relu elu prelu sigmoid tanh swish gelu softplus add subtract multiply divide power`), a Layer class or instance, a callable over streams (arity inferred), or `(callable, arity)`. `linear_in` projects to Σarity arguments; its `out_features` must equal that sum. Inputs must have dim rank 1 (several inputs are concatenated). Internally it is a composed `Modely`. |
| `BatchNorm(axis=1, momentum=0.99, epsilon=1e-3, center=True, scale=True, name=None)` | unchanged | Per-feature statistics. Uses the inference branch in `validate` / `_predict`. |
| `Parameter(...)` | – | See §5. |

### 6.2 Fuzzy and lookup (no weights)
| Signature | in → out | Notes |
|---|---|---|
| `Fuzzify(centers: list[float], function="Triangular", name=None)` | `(1, T)` → `(len(centers), T)` | `function` is one of `"Triangular" \| "Rectangular" \| "Gaussian"`. Centers are sorted. The outer sets extend one spacing beyond the end centers. Triangular memberships sum to 1 inside the center range. |
| `Interpolation(x_points, y_points, mode="linear", name=None)` | elementwise | `mode` is `"linear"` (piecewise) or `"polynomial"` (the unique global polynomial). Queries are clamped to `[x_min, x_max]`. Not trainable. |

### 6.3 Activations (elementwise, no weights except PReLU)
`ReLU(max_value=None, negative_slope=0.0, threshold=0.0)`, `LeakyReLU(negative_slope=0.3)`, `ELU(alpha=1.0)`, `PReLU(shared_axes=None)` (trainable), `Sigmoid()`, `Tanh()`, `Swish()`, `GELU(approximate=True)`, `Softplus()`, `Softmax(axis=-1)` (use `axis=1` for features, §1.15). All accept `name=`.

### 6.4 Math (elementwise)
`Sin Cos Tan Asin Acos Atan Exp Log Log10 Sqrt Abs Floor Ceil Deg2Rad Sign Negative` take no arguments except `name=`. `Clamp(min=None, max=None)`: a `None` bound is unbounded. `Sum(axis=None)` sums the dim axes and keeps them with length 1 (`axis=None` sums all dim axes, an int picks one). `Floor`, `Ceil` and `Sign` have zero gradient.

### 6.5 Axis manipulation
| Signature | Effect |
|---|---|
| `Select(idx, axis=0)` | Picks one index on a dim axis and keeps it with length 1. Negative idx allowed. |
| `Range(start, end, axis=0)` | Slice `[start, end)` on a dim axis. |
| `Concatenate(axis=0)([a, b, ...])` | Concatenates along a dim axis; time and seq must match. |
| `TimeSelect(idx)` | One time sample, kept with length 1 (`-1` = newest). |
| `TimeRange(start, end)` | Slice `[start, end)` on the time axis. |
| `TimeConcatenate()([a, b, ...])` | Concatenates along time; dim and seq must match. |

There is **no layer that selects along a seq axis**, and none that reshapes or transposes. Write a custom layer (§16) if one is needed.

### 6.6 Calculus and recurrence
See §10 (`Loop`, `Roll`, `rollback`) and §11 (`Derivative`, `Integrate`, `Ode`, `OdeNet`, `Gradient`, `PortHamiltonian`).

---

## 7. Modely API

`Modely(name: str, inputs: list[Input], outputs: list[Output])`
- `inputs` lists the Inputs the model reads. Inputs used only as targets need not be listed; they are discovered at build. After build, `model.train_inputs` holds every Input that data must provide.
- `outputs` is a list of `Output` nodes.

| Method | Purpose |
|---|---|
| `minimize(name, source, target=None, loss="mse")` → self | Registers an objective (§8). Call before build. |
| `remove_minimizer(name)` | Removes an objective by name. |
| `rollback({input: stream}, steps: int, name=None)` → self | k-step feedback unrolling (§10). Call before build. |
| `build()` → self | Creates the `keras.Model` (`model.model`) and its weights. Can be called again; weights are kept. |
| `model(inputs_dict)` | Inference. Takes `{input_name: array (batch, *dim, time, *seq)}`; the batch axis may be omitted for one sample, and a flat list is reshaped when its size matches [verified: `model({"x": [1,2,3,4,5]})` for `x.sw(5)`]. Extra keys are ignored. Returns `{output_name: backend tensor}`; convert with `keras.ops.convert_to_numpy`. Outputs also include the streams of minimizer sources and targets. |
| `model([stream, ...])` | Composition: returns the outputs as streams of an outer graph (§12). |
| `train(train_data, val_data=None, epochs=10, batch_size=1, optimizer=None, lr=1e-3, shuffle=True, optimizer_kwargs=None, printer="legacy")` → `dict` | Keras history (§8). |
| `validate(val_data, out_dir=None, show=False, history=None)` → `ValidationResult` | Metrics plus optional PNG figures (§13). |
| `flatten()` → Modely | Inlines every composed sub-model. |
| `summary()` | Keras summary. |
| `export_html(out_dir, filename=None, *, open_subgraph_in_new_tab=False, physics=True)` → str | Interactive vis-network graph, one page per sub-model. |
| `plot(to_file, include_minimizers=True, flatten=False)` | Graphviz image (needs the system graphviz binaries). |
| `save(path)`, `Modely.load(path)` | nnodely format: a folder with `model.json` and `model.weights.h5`. Not for Loop, Roll, OdeNet or Gradient (§1.14). |
| `export_keras(filename.keras)`, `Modely.import_keras(filename, safe_mode=True)` → `keras.Model` | The imported object is a plain Keras model; call it with batched float32 dicts. Pass `safe_mode=False` if loading refuses nested or lambda configs. |
| `export_onnx(filename, *, input_signature=None, batch_size=None, opset_version=None, verbose=False)` → Path | Needs the `onnx` extra and the TF or Torch backend (§13). |
| `Modely.validate_onnx(filename, inputs, *, return_dict=False, providers=None)` | Runs an exported ONNX file with onnxruntime. |

Properties: `built`, `model` (keras.Model), `inputs`, `outputs`, `order` (topological node list), `minimizers` (list of dicts with keys `name`, `source`, `target`, `loss`), `train_inputs`, `train_outputs`. `print(model)` lists the nodes.

---

## 8. Objectives and training

`minimize(name, source, target=None, loss="mse")`
- `source`: an `Output` (preferred) or any stream. A non-Output source is added to the outputs under its auto name.
- `target` follows §1.5. Common forms:
  - `y.next()`: one-step-ahead prediction of an input that is also read by the model.
  - `Input("y_meas").last()`: a separate measured column.
  - `Input("s_target", dim=2, seq=H)`: a trajectory target for `Loop` (the shapes must match the source).
  - `None`: residual.
- `loss`: any Keras loss (a name such as `"mse"`, `"mae"`, `"huber"`, `"log_cosh"`), a `keras.losses.Loss` instance, a serialized config, or a callable `loss(y_true, y_pred)`.
- The total training loss is the sum of all objectives (equal weights, §1.7).

`train(...)`
- `optimizer`: a name (`sgd rmsprop adam adamw adagrad adadelta adamax adafactor nadam ftrl lion`; default `"adam"`), a config dict, or an instance. `lr` and `optimizer_kwargs` (e.g. `{"weight_decay": 1e-4, "global_clipnorm": 1.0}`) apply only to names; configure an instance yourself.
- `printer`: `"legacy"` (default, a loss table), `"tiny"`, `"nnodely"` (animated), `None` (silent), or any `keras.callbacks.Callback`.
- Returns a history dict. With one objective the keys are `loss` and `val_loss` (with `val_data`). With several objectives Keras adds `<source_output_name>_loss` per objective, e.g. `['loss', 'omega_next_loss', 'theta_next_loss']` [verified].
- Each `train` call continues from the current weights.
- Padded rollout steps (`seq_length="full"`) are masked automatically.
- To freeze a composed block, set `block.model.trainable = False` before training the outer model.

Reproducibility: call `set_seed(int)` before creating layers. It seeds Python, NumPy and the backend, and sets `NNODELY_SEED` so child processes inherit it. `get_seed()` returns the current seed.

---

## 9. DataLoader

`DataLoader(model, source, format=None, csv_glob="*.csv", delimiter=",", header="infer", dtype=np.float32, seq_length=None, step=1, on_short="error")`. The model must be built.

**Sources**
- A dict `{input_name: array (T,) or (T, *dim)}`. Keys must be the model's input names, including target-only inputs. `format` is ignored for dicts.
- A `pd.DataFrame`, a `.csv` path, or a folder of CSVs matching `csv_glob`. `format={"input": "col" | int_index | ["c1","c2",...]}` maps inputs to columns; a list is used for `dim>1`. Unmapped columns are ignored, and an input not listed in `format` is looked up by its own name.
- A list of dicts or DataFrames, or a folder with several files: several **simulations**. Windows never cross simulations, and lengths may differ.

**Windowing and alignment** [verified]
- Every input is aligned on a common current sample `t`. Within one simulation of length `T`:
  - `first = max_past - 1`, where `max_past` is the largest past over all inputs.
  - The number of samples is `T - max_future - first - max_seq_span`, where `max_seq_span = Σ(seq_len - 1)`. Sample i has current time `t_i = first + max_seq_span + i`.
  - `step=k` keeps every k-th sample.
- A sequence input `Input(seq=L)` gives, at sample i, the temporal windows ending at `t_i-L+1 … t_i`. **The last sequence element is aligned with the current sample of non-sequence inputs.** For example, `a.last()` = 3 while `b(seq=3)` = [1,2,3].
- An input with no window contributes the current sample only.
- A too-short simulation raises. `on_short="skip"` drops it.
- Dynamic `seq=-1` needs `seq_length=int`, or `seq_length="full"` to get one sample per simulation over the whole length. With `"full"`, simulations are padded by repeating the last step; `loader.mask` has shape `(n_sims, max_len)` and training masks the padded steps. `step>1` is not allowed with `"full"`.

**Access**: `len(loader)`, `loader[i]` → `{name: (*dim, time, *seq)}` (no batch axis; can be passed straight to `model(...)`), `iter(loader)`, `loader.as_dict()` → `{name: (N, *dim, time, *seq)}`, `loader.get_input(name)`, `loader.inputs`, `print(loader)`.

**Normalization** (explicit and per loader): `loader.normalize(method="minmax" | "standard", names=None, feature_range=(-1.0, 1.0))` changes the loader in place, with statistics computed per feature. `loader.denormalize()` restores the loader. `loader.denormalize({output_name: array})` inverts predictions, matching each output to the input its minimizer target reads. `loader.normalization_stats` holds the statistics. A validation loader must be normalized with the same statistics. Normalizing each loader separately gives different statistics, so for consistent scaling prefer normalizing the raw arrays yourself, or fold the scaling into the model with `Constant`s.

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
- Mapping `{input (obj or name): stream (obj or name)}`. The stream can be any stream of the graph, including an Output, and must have shape `(*input.dim, 1, *input.seq)`.
- It evaluates the model `steps` times. After each evaluation, the stream's one-sample value is appended to the input's time window and the oldest sample is dropped. Model outputs are those of the **final** evaluation.
- Inputs that are not fed back, and targets, are held at their window values for every step. Align the target yourself: `data = {"x": x[:-k], "F": F[:-k], "x_ahead": x[k:]}` with target `Input("x_ahead").last()` (recipe R9).
- Several feedbacks at once: `rollback({"x": "x_next", "v": "v_next"}, steps=10)`.
- Supports `save` / `load`.

### 10.2 `Loop(f, callback, initial=0.0, inputs=None, name=None, length=None, collect=True)`
- `f`: the body `Modely` (built automatically if needed). It must have no Input-target objectives (§1.11).
- `callback`: `{body_input: body_output}`, as objects or names. The output dim must equal the input dim, and the output time must either equal the input time (the state is replaced) or be 1 when the input window is longer than 1 (the window is shifted, and after n steps it holds only predictions).
- `initial`: seeds the fed-back inputs. It is a stream or number when there is one callback, and a dict `{body_input: stream | number}` when there are several (a dict is required then).
  - A stream with **one extra trailing seq axis** relative to the body input (e.g. `Input("s0", dim=2, seq=H)` for body input `dim=2`): step 0 reads element 0 of the sequence.
  - A stream of the same shape: used directly.
  - A number: broadcast.
  - For a windowed feedback input use a window of the seq Input: `initial={x: x0_seq.sw(3)}`.
- `inputs`: `{body_input: outer_stream}` for exogenous signals. The outer stream must match dim and time. Its seq is either the same as the body input (held constant) or one axis longer (consumed one element per step).
- Body inputs that are neither fed back nor bound are held constant, and become inputs of the outer model under their own names.
- Horizon: taken from the concrete seq length of the rollout streams. Pass `length=` when every rollout stream is dynamic (`seq=-1`) or when their lengths differ. With `seq=-1`, the rollout follows the length of the tensor given at call time, and `length` only fixes the shapes used at build.
- At least one source must derive from an `Input`, so that there is a batch axis.
- Timing: at step k (0-based) the body sees state `s_k` (with `s_0` from `initial`) and exogenous `u_k`. Every body output at step k is computed from `s_k`, and the callback output becomes `s_{k+1}`. With `collect=True` the result stacks steps `0..H-1` as a new **last seq axis**, giving shape `(*dim, time, *seq, H)`. So the fed-back trajectory is `s_1..s_H`: with `initial` = states window `[t..t+H-1]`, the target must be states `[t+1..t+H]`, i.e. data `{"s_seq": S[:-1], "s_target": S[1:]}` (recipe R3). A non-fed-back output (for example a measurement `y(s_k)`) aligns with the unshifted window. `collect=False` returns only the last step.
- Multiple outputs: `a, b = Loop(...)` yields one stream per body output, in `f.outputs` order.
- Loops can be nested; the body of a Loop can itself contain a Loop.
- Supports `export_keras` and ONNX. Does **not** support `save` / `load`.

### 10.3 `Roll(f, callback: dict, steps=None, name=None)`
- `f` must already be built (it raises otherwise). `callback` holds exactly one pair `{input: output}`, and the output must have time 1.
- Returns a stream shaped like the callback input's window. `steps` defaults to the window length, so every sample is a prediction. The outer model must list the body's own Input objects as its inputs, because the other inputs are fed unchanged at every step.
- Supports `export_keras` and ONNX. Does not support `save` / `load`.

---

## 11. Physics layers

### 11.1 `Derivative(order=1, respect_to=<Input | float>, init=None, window=None, poly_order=None, name=None)(stream)`
Orders 1 and 2 are supported.
- **With respect to an Input** (autodiff, exact): the result is `d(Σ stream)/d(input window)` (a VJP) with the Input's full shape. It includes the trainable weights, can be nested, and can be trained through. `init`, `window` and `poly_order` are not allowed in this mode. The stream must depend on that Input. Typical use is a PINN residual: `minimize("physics", Derivative(respect_to=t)(u) + u)`. For `u` with dim>1 the result is the gradient of the sum of the components; take `Select` first to get one component. ONNX export: TensorFlow only, with batch fixed to 1. Torch refuses, and JAX cannot export.
- **With respect to time** (`respect_to=dt: float`): a causal finite difference along the stream's own time axis that **keeps the window length**.
  - `init` is the sample(s) just before the window: a stream (time 1, or `window-1`), a number, or `None` (zero).
  - `window` defaults to `order+1`, the plain backward difference. Larger values fit a polynomial by least squares (Savitzky–Golay style): they smooth noise at the cost of about `(window-1)/2` samples of delay.
  - `poly_order` ranges over `order..window-1` and defaults to `order`.
  - The time derivative is the exact inverse of `Integrate(solver="euler")` with the same `init`.

### 11.2 `Integrate(solver="euler", dt=<required float>, init=None, name=None)(rate)`
- `solver` is `"euler"` / `"rectangular"` (y[i] = y[i-1] + dt·r[i]) or `"trapezoidal"`. The window length is preserved. `init` is the value just before the window (a stream of time 1, a number, or `None` = 0).
- On a one-sample window it is exactly one state update: `Integrate(dt=dt, init=v.last())(a.last())` = `v + dt*a`.
- On an n-sample window it returns the whole running integral in one pass, with no rollout needed: `x = Integrate(dt=dt)(Integrate(dt=dt)(a.sw(n)))`.
- There is no rk4 or heun here; use `Ode` for those.

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
- `t`: a stream of report times whose last axis has a static count N ≥ 2, e.g. `Input("t", seq=N)` with data `(batch, 1, 1, N)`. The times are **shared across the batch** (the first row is used), and `t[0]` is the initial condition.
- `initial`: `{state: outer_stream}` with no seq axis. By default the body's own state Inputs are used, so the outer model lists them as inputs.
- Output: `(*dim, time, N)` per state, with index 0 equal to the initial state. `a, b = OdeNet(...)` unpacks several states.
- `method`: a fixed tableau (`euler`, `midpoint`, `heun`, `rk4`) applies `steps` substeps per reported interval, unrolled; use it for training and export. `"dopri5"` is adaptive (controlled by `rtol`, `atol`, `max_steps`): not ONNX-exportable, not reverse-differentiable on JAX, meant for inference. To switch a built model: `model.model.get_layer(<OdeNet name>).set_method("dopri5")`, which is why `name=` should be set.
- Hybrid events: `event` names an output of `f` of shape `(1,1)` that is positive before the event and negative after. `reset` gives `{state: output}` with the post-event values of every state. Events need a fixed tableau, and the event time is differentiable.
- **Training with a DataLoader** [verified pattern from tests]: the state inputs have no seq, so the loader aligns them with the **last** element of the `t` and target windows (§9). Shift the seed column by N-1 so that it equals the first point of each window: `seed = np.vstack([np.repeat(ref[:1], N-1, axis=0), ref[:1-N]])`. See recipe R5.
- Supports `export_keras`. Not `save` / `load`.

### 11.5 `Gradient(f, x, name=None)` (`from nnodely.layers.gradient import Gradient`)
- `f` is a built `Modely` with exactly one input and one scalar output (dim `(1,)`, time 1). `x` is any stream shaped like f's input.
- Returns `∂f/∂x` with x's shape, using backend autodiff, differentiable with respect to the weights. Can be evaluated at several points with shared weights.
- Use it for Hamiltonian or potential-based models: e.g. H = Linear(1)(Tanh()(Linear(16)(q.last()))) in `f`, then `Gradient(H_model, x.last())`.
- No ONNX export, no `save` / `load`.

### 11.6 `PortHamiltonian(state_dim, input_dim, hamiltonian=(64,64), activation=Tanh, J="constant", R="constant", G="constant", name=None)`: EXPERIMENTAL
- Lives in `nnodely/layers/porthamiltonian.py`. At the time of writing the file is untracked in git and not exported at top level.
- `dx, y = PortHamiltonian(...)(x.last(), u.last())` returns the field `(J−R)∇H + G u` and the collocated output `Gᵀ∇H`. J is skew by construction, R = LLᵀ.
- `J`, `R` and `G` are `"constant"` (a Parameter) or a list of hidden widths (an MLP of the state). Call it inside `Ode(lambda x, u: field(x, u)[0], ...)`; all calls share weights. `state_dim` must be at least 2.

---

## 12. Model composition

- `block = Modely("b", inputs=[a, c], outputs=[o1, o2]).build()`. Then `o1s, o2s = block([stream_for_a, stream_for_c])`. The streams go in the order of `block.inputs`, and a single output returns a single stream.
- The block brings its weights along (trained weights are reused while input shapes match), and training the outer model also trains the block. To freeze it: `block.model.trainable = False`.
- Calling the same block twice shares its weights (§1.22 covers naming).
- `EquationLearner` uses composition internally.
- `Derivative(respect_to=Input)` inside a block does not work once the block's inputs are rebound. Put the derivative in the outer model instead, where the Input is declared.
- `model.flatten()` returns an equivalent model with everything inlined.

---

## 13. Inference, validation and export

**Inference arrays**: float32 NumPy, shape `(batch, *dim, time, *seq)`, where `time` is the Input's union window (`x.shape`). A single sample may drop the batch axis.

**`validate(val_data, out_dir=None, show=False, history=None)`**
- Scores every objective and prints a summary.
- Metrics per objective (`result[name].metrics`, or `result.metrics()` for everything): `loss rmse mae max_error bias std_error nrmse_pct fit_pct r2 correlation non_finite`.
  - `fit_pct` = 100(1 − ‖y−ŷ‖/‖y−ȳ‖).
  - Relative metrics are NaN when the target is constant.
- `out_dir` writes PNGs (prediction vs target, error, parity, histogram, plus loss curves when `history` is given), and `result.figures` lists them.
- Remove residual objectives first (§1.13).

**Export support**

| Model contains | `save`/`load` | `export_keras` | ONNX TF | ONNX Torch | ONNX JAX |
|---|---|---|---|---|---|
| Plain layers, `Parameter`, `Constant`, `Fir`, `Linear`, `LocalModel`, `EquationLearner`, `Integrate`, time-`Derivative`, `rollback` | yes | yes | yes | yes | no |
| `Derivative(respect_to=Input)` | yes | yes | yes (batch 1) | no | no |
| `Loop`, `Roll` | no [verified] | yes [verified] | yes [verified for Loop] | yes (per docs) | no |
| `OdeNet` (fixed tableau) | no [verified] | yes | yes (per docs) | yes (per docs) | no |
| `OdeNet(method="dopri5")` | no | yes | no | no | no |
| `Gradient` | no [verified] | yes, `import_keras(..., safe_mode=False)` [verified] | no | no | no |

JAX cannot export ONNX at all: jax2tf emits an unconvertible XlaCallModule. `export_onnx(batch_size=n)` fixes the batch axis.

---

## 14. Common errors and fixes

| Error (substring) | Cause | Fix |
|---|---|---|
| `Model X is not built` / `Model is not built` | `DataLoader` / `train` / call before `build()` | Call `model.build()` first. |
| `Training target '...' must be present in the dataset or be a constant value` | Computed target (§1.5) | Residual form: `minimize(name, source - expr)`. |
| `Missing data for input "..."` inside `LoopImpl` / `RollImpl` | Body model has an Input-target objective (§1.11) | Remove objectives from the body. |
| `KeyError: '<input>'` at inference | Target-only input not supplied (§1.12) | Add it to the dict (zeros are fine). |
| `... is missing required inputs: [...]` (DataLoader) | Data dict lacks an input, often a target-only one | Add the column, or map it with `format`. |
| `has only N samples, but the model requires at least M` | Simulation shorter than windows plus sequences | Shorter windows or seq, more data, or `on_short="skip"`. |
| `Some inputs have undefined sequence length` | `seq=-1` without `seq_length` | `DataLoader(..., seq_length=H)` or `"full"`. |
| `roll stream ... must produce one temporal sample with shape ...` | Rollback stream has time > 1 or a wrong dim | Feed back a one-sample stream matching the input's dim. |
| `Loop cannot determine the rollout length` | All rollout streams are dynamic | Pass `length=H`. |
| `Loop rollout inputs declare different lengths` | Mixed seq lengths | Pass `length=` or make the lengths equal. |
| `Loop needs at least one input with a batch axis` | Only constants seed and drive the Loop | Bind a stream derived from an `Input`. |
| `Multiple Loop callbacks require an initial value dict` | Several callbacks, scalar `initial` | `initial={inp1: s1, inp2: s2}`. |
| `the output must either cover the whole time window of the input ... or a single step` | Callback output time is incompatible | The output time must be 1 or equal to the input time. |
| `has two different outputs named ...` | A block called twice exposes same-named outputs | Wrap each call's output in a distinctly named `Output`. |
| `the differentiated relation does not depend on input ...` | `Derivative` w.r.t. an Input inside a composed block, or no real dependency | Move the derivative to the model that declares the Input. |
| `compares '...' of shape (N, ...) with 'ConstantK' of shape (...)` in `validate` | Residual objective (§1.13) | `remove_minimizer` before `validate`. |
| `Loop.__init__() got an unexpected keyword argument 'horizon'` (also `Roll`, `OdeNet`, `Gradient` missing args) on `Modely.load` | Unsupported save format (§1.14) | Use `export_keras` / `import_keras`. |
| `TypeError: bad operand type for unary -` | `-stream` | `-1.0 * stream` or `Negative()(stream)`. |
| `OdeNet integrates an autonomous field, but [...] are body inputs that are not states` | Exogenous input in the field | Use `Ode` + `Loop` for driven systems. |
| Softmax output is all ones | Default axis is time | `Softmax(axis=1)`. |

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
y_next = Output("y_next", Fir(out_features=1)([y.sw(2)]) + Fir(out_features=1)([u.sw(2)]))
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
pred = model(val[0])["y_next"]                        # shape (1, 1, 1)
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
    Linear(out_features=2, use_bias=False)([state.last()])
    + Linear(out_features=2, use_bias=False)([force.last()]))
body = Modely("step", inputs=[state, force], outputs=[step_out]).build()
# 2) outer: sequences of H steps
s_seq, f_seq = Input("s_seq", dim=2, seq=H), Input("f_seq", seq=H)
traj = Output("traj", Loop(f=body, callback={state: step_out},
                           initial={state: s_seq}, inputs={force: f_seq}))
sim = Modely("simulator", inputs=[s_seq, f_seq], outputs=[traj])
sim.minimize("sim_error", traj, Input("s_target", dim=2, seq=H))
sim.build()
# 3) data: target is the state shifted one step ahead of the seed window
states = np.stack([th, om], axis=-1)             # (T, 2)
data = DataLoader(sim, source={"s_seq": states[:-1], "f_seq": tq[:-1], "s_target": states[1:]}, step=5)
sim.train(data, epochs=5, batch_size=32, lr=1e-2, printer=None)
# body weights are trained too; body(...) can now be used for one-step prediction
```
To use a physical step inside the body, build the body from `Ode(...)` as in R2, feeding back both states (`callback={"theta": "theta_next", "omega": "omega_next"}`, `initial={"theta": th_seq, "omega": om_seq}`). For whole simulations of different lengths use `Input(seq=-1)`, `Loop(..., length=H_build)` and `DataLoader(..., seq_length="full")`.

### R4. Physics-informed network (Derivative w.r.t. an Input + residual)
```python
from nnodely import Derivative, Input, Linear, Modely, Output, Tanh, DataLoader
t_in = Input("t")
u_hat = Linear(out_features=1)([Tanh()([Linear(out_features=16)([t_in.last()])])])
du_dt = Derivative(order=1, respect_to=t_in)(u_hat)
u_out = Output("u", u_hat)
pinn = Modely("decay", inputs=[t_in], outputs=[u_out])
pinn.minimize("data", u_out, Input("u_meas").last())
pinn.minimize("physics", du_dt + u_hat)             # enforce du/dt = -u
pinn.build()
times = np.linspace(0, 2, 200)
data = DataLoader(pinn, source={"t": times, "u_meas": np.exp(-times)})
pinn.train(data, epochs=20, batch_size=32, lr=1e-2, printer=None)
pinn.remove_minimizer("physics")                   # validate() cannot score residuals
pinn.validate(data)
```

### R5. Neural ODE with OdeNet (including DataLoader training alignment)
```python
from nnodely import Input, Linear, Modely, OdeNet, Output, Tanh, DataLoader
N = 25
z = Input("z", dim=2)
field = Modely("field", inputs=[z], outputs=[Output("dz",
    Linear(out_features=2)([Tanh()([Linear(out_features=32)([z.last()])])]))]).build()
t_rep = Input("t_rep", seq=N)
z_traj = Output("z_traj", OdeNet(f=field, states={z: "dz"}, t=t_rep, method="rk4", steps=2, name="node"))
node = Modely("neural_ode", inputs=[z, t_rep], outputs=[z_traj])
node.minimize("traj", z_traj, Input("z_ref", dim=2, seq=N))
node.build()
# data: ref (T, 2) sampled at times (T,). The seed must be the FIRST point of each window:
seed = np.vstack([np.repeat(ref[:1], N - 1, axis=0), ref[:1 - N]])
data = DataLoader(node, source={"z": seed, "t_rep": times, "z_ref": ref})
node.train(data, epochs=100, batch_size=64, lr=1e-2, printer=None)
# inference over any horizon with the same N points; targets must still be supplied (zeros)
out = node({"z": np.ones((1, 2, 1)), "t_rep": np.linspace(0, 1, N).reshape(1, 1, 1, N),
            "z_ref": np.zeros((1, 2, 1, N))})["z_traj"]          # (1, 2, 1, N)
node.model.get_layer("node").set_method("dopri5")               # adaptive for inference
```

### R6. Gain scheduling with Fuzzify + LocalModel
```python
from nnodely import Fir, Fuzzify, Input, LocalModel, Modely, Output
trq, gear = Input("trq"), Input("gear")
mu = Fuzzify(centers=[1.0, 2.0, 3.0, 4.0], function="Triangular")([gear.last()])   # (4, 1)
engine = LocalModel(out_features=1)([trq.sw(10)], [mu])                              # (1, 1)
m = Modely("sched", inputs=[trq, gear], outputs=[Output("force", engine)]).build()
m({"trq": np.ones((1, 1, 10)), "gear": [[2.5]]})
# engine.kernel.shape == (4, 10, 1): one 10-tap FIR per gear cell
```

### R7. Composition (reuse a trained block)
```python
from nnodely import Fir, Input, Linear, Modely, Output, ReLU
sig = Input("sig")
smoother = Modely("smoother", inputs=[sig], outputs=[Output("smooth", Fir(out_features=1)([sig.sw(10)]))]).build()
raw = Input("raw")
cmd = Output("cmd", ReLU()([Linear(out_features=1)([smoother([raw.sw(10)])])]))
controller = Modely("controller", inputs=[raw], outputs=[cmd]).build()
controller({"raw": np.ones((1, 1, 10))})
```

### R8. k-step-ahead training with rollback
```python
from nnodely import Fir, Input, Modely, Output, DataLoader
K = 5
x, F = Input("x"), Input("F")
x_pred = Output("x_pred", Fir(out_features=1)([x.sw(5)]) + Fir(out_features=1)([F.last()]))
rb = Modely("rb", inputs=[x, F], outputs=[x_pred])
rb.rollback({x: x_pred}, steps=K)
rb.minimize("ahead", x_pred, Input("x_ahead").last())
rb.build()
data = DataLoader(rb, source={"x": xs[:-K], "F": Fs[:-K], "x_ahead": xs[K:]})
rb.train(data, epochs=20, batch_size=64, lr=1e-2, printer=None)
```

### R9. Mechanical model: acceleration → velocity → position in one pass (no rollout)
```python
from nnodely import Input, Integrate, Linear, Modely, Output
dt, n = 0.01, 50
F, v0, x0 = Input("F"), Input("v0"), Input("x0")
acc = Linear(out_features=1)([F.sw(n)])                   # (1, n): per-sample force -> acceleration
vel = Integrate(dt=dt, init=v0.last())(acc)                # (1, n)
pos = Integrate(dt=dt, init=x0.last())(vel)                # (1, n)
m = Modely("mech", inputs=[F, v0, x0], outputs=[Output("pos", pos)])
m.minimize("pos_err", m.outputs[0], Input("x_meas").sw(n))
m.build()
```
Caution: `v0` and `x0` are read at the current sample, which is the **end** of `F.sw(n)`. Provide their data columns shifted by `n` so that they hold the state just before the window.

---

## 16. Custom layers

Pattern: a symbolic `Layer` subclass returns a serializable Keras layer from `build_layer()`. The output shape is inferred automatically by running the Keras layer on zeros. Override `output_shape(self, *inputs) -> (dim, time, seq)` only when that cannot work (dynamic axes, value-dependent shapes).
```python
import keras
from nnodely.core.layer import Layer

@keras.saving.register_keras_serializable(package="my_project")
class SquashImpl(keras.layers.Layer):
    def __init__(self, gain=1.0, **kwargs):
        super().__init__(**kwargs); self.gain = float(gain)
    def call(self, x):                      # x: [batch, *dim, time, *seq]; list of tensors if multi-input
        return keras.ops.tanh(self.gain * x)
    def get_config(self):
        return {**super().get_config(), "gain": self.gain}

class Squash(Layer):
    def __init__(self, gain=1.0, name=None):
        self.gain = float(gain)
        super().__init__(name=name, gain=self.gain)   # kwargs -> self._properties (re-used when called)
    def build_layer(self):
        return SquashImpl(gain=self.gain, name=self.name)
    def get_config(self):                              # used by Modely.save/load (must match __init__)
        return {"name": self.name, "gain": self.gain}

y = Squash(gain=2.0)([x.sw(3)])
```
Rules:
- Every constructor argument must be passed to `super().__init__(name=name, **kwargs)`, because the node is re-instantiated as `cls(name=..., **properties)` when it is called.
- Use `keras.ops` only, never backend-specific ops.
- Register the Impl for `export_keras`.
- Weights go in the Impl's `build()` through `add_weight`.
- A multi-input layer receives a list of tensors in `call`.
- Classes register themselves for `save` / `load` by class name, so class names must be unique.
- Example of a custom multi-input physics layer: `examples/cheetah/model.py` (`DynamicsStep`, a linear solve).

---

## 17. Checklist before returning code

- [ ] `KERAS_BACKEND` is set before the imports, if the user needs a specific backend.
- [ ] Every Input name is unique; every Output name is unique; data dict keys are exactly these names, including target-only Inputs.
- [ ] Layers receive windows (`.last()`, `.sw()`), not raw Inputs.
- [ ] The shape of every stream is written down (§3). `Fir` returns `(out, 1)`; `Linear` keeps time; combined streams have compatible shapes.
- [ ] Objectives are declared before `build()`. Targets are untransformed Input windows, floats or `None`. Computed targets use the residual form. There is one objective per source.
- [ ] Physical constants are `Parameter(value=guess)` or `Constant`, created outside `Ode` functions.
- [ ] No `-stream`; `-1.0 * stream` instead.
- [ ] Body models (`Loop` / `Roll` / `OdeNet` / `Gradient`) have no objectives. Loop targets are shifted one step from the seed window. OdeNet seeds are shifted by N-1.
- [ ] `train(...)` has explicit `epochs`, `batch_size` and `lr`; `printer=None` for scripts and tests.
- [ ] Residual objectives are removed before `validate`. Inference dicts include target-only inputs.
- [ ] Export format matches the model (§13): `export_keras` for Loop, Roll or OdeNet models.
- [ ] Results are converted with `keras.ops.convert_to_numpy(...)` before NumPy or matplotlib use.

---

## 18. Internals and repository map (for editing the library itself)

```
src/nnodely/
  __init__.py            public exports (see §4)
  core/stream.py         Node, Shape(dim,time,seq), Stream (operator overloads, literal→Constant cache), NODE_REGISTRY
  core/layer.py          Layer base (symbolic call → new node; tensor call → build_layer()), has_batch,
                         BinaryOp Add/Subtract/Multiply/Divide/Power, Identity
  core/modely.py         Modely (build, minimize, train, validate, rollback, save/export), ModelCall, IntermediateOutput
  core/dag.py            next_name counter, flatten (inline ModelCalls per call scope), toposort
  core/dataloader.py     DataLoader (read → per-simulation arrays → sliding windows → alignment, mask, normalize)
  core/registry.py       ModelSerializer: model.json (+ model.weights.h5) via get_config/from_config
  core/validation.py     score_signal, SignalScore, ValidationResult
  layers/input.py        Input (+ sw/last/next → SampleWindow)
  layers/time_ops.py     SampleWindow, Select, Range, TimeSelect, TimeRange, Concatenate, TimeConcatenate
  layers/{linear,fir,localmodel,fuzzify,interpolation,equationlearner,batchnorm,activations,arithmetic,trigonometric}.py
  layers/{parameter,constant,output}.py
  layers/derivative.py   Derivative (autodiff dispatch per backend | Savitzky–Golay banded operator)
  layers/integrate.py    Integrate (cumulative quadrature operator matrix)
  layers/ode.py          Ode (Butcher tableaux inline), OdeNet (+dopri5 while_loop, events)
  layers/loop.py         Loop, LoopImpl (statically unrolled rollout), LoopOutput
  layers/roll.py         Roll/RollImpl, ModelRollImpl (used by Modely.rollback)
  layers/gradient.py     Gradient (tape per backend)
  layers/porthamiltonian.py  experimental
  utils/                 printers, plot (graphviz / html), validation_plot, random (set_seed), utils (loss/optimizer resolution, MaskedLoss)
tests/                   pytest; run all backends with scripts/test_all_backends.sh; KERAS_BACKEND defaults to tensorflow in conftest
docs/                    Sphinx user guide (getting_started, guide/*.rst, api/*.rst)
examples/                pendulum (Ode + Loop), longitudinal_dynamics (Fuzzify/LocalModel vehicle), cheetah (custom layer + Roll), hnn.py (PortHamiltonian, experimental)
```
Mechanics worth knowing:
- Symbolic call: `Layer.__call__(streams)` infers the output shape (by probing `build_layer()` on zeros unless `output_shape` is overridden) and returns a *new* node of the same class with the same name and properties.
- `Modely.build()` flattens composed models, topologically sorts them, and calls each node's `call()` on Keras tensors. Concrete Keras layers are keyed **by node name**, which is why same-named nodes share weights.
- Parameters and Constants have no predecessors. They are called with an arbitrary anchor tensor and produce batch-free tensors, which `has_batch()` tracks so that arithmetic can broadcast them.
- Minimizer sources and targets that are not Outputs are appended as extra Keras outputs. `train` compiles one loss per source output name and builds `y` from the DataLoader column the target traces to (through single-predecessor chains) or from a constant.
