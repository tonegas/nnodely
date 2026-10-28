# Changelog

All notable changes to nnodely are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [2.0.0] - 2026-10-12

nnodely 2.0 is a rewrite of the library on **Keras 3**. The same model runs on
TensorFlow, PyTorch or JAX, chosen with the `KERAS_BACKEND` environment
variable, and trains, validates and exports through Keras. Models saved with
1.x cannot be loaded by 2.0.

### Added

- `Modely`, a model declared by its inputs and outputs and built with
  `build()`; a built model is called on data, or on streams to become a block
  of a larger model.
- `DataLoader` for dicts, DataFrames and CSV files or folders, with several
  simulations, sequences, padding masks and normalization. `delimiter` and
  `header` are passed to `pandas.read_csv`; n/a values are reported. An array
  holds the features of each sample laid out as the input's `dim`, or flat in
  row-major order; another layout of as many values is refused rather than
  reordered.
- `minimize()` with any Keras loss, a `gain` per objective and `seq_weights`
  along a rollout; targets can be computed streams or numbers.
- A layer whose constructor takes only `name` is applied as it is created:
  `Sin(x)` is `Sin()(x)`. These are `Sin`, `Cos`, `Tan`, `Asin`, `Acos`,
  `Atan`, `Atan2`, `Exp`, `Log`, `Log10`, `Sqrt`, `Abs`, `Floor`, `Ceil`,
  `Deg2Rad`, `Sign`, `Negative`, `Sigmoid`, `Tanh`, `Swish` and `Softplus`;
  layers with settings (`ReLU`, `GELU`, `Clamp`, `Sum`, `TimeConcatenate`,
  ...) are configured first.
- An activation, a math or trigonometric layer, `Clamp` or a membership
  function called on a list of streams is applied to each and returns a list:
  `Sin([a, b])` gives `[sin(a), sin(b)]`.
- `Atan2([y, x])`, the two-argument arctangent: the angle of the point
  `(x, y)` in `[-pi, pi]`, with its quadrant. `y` and `x` broadcast as the
  arithmetic operators do.
- A model used as a block keeps the shapes it was declared with: each stream
  it is called on must have the shape its graph reads that input with, and
  another one is refused; an `Input` given itself is read through the block's
  own windows. An Input the block reads without declaring it is the caller's
  Input too. A model with a `rollback` is refused as a block, which would
  evaluate it once: its one-step model is a `Modely` declared on the same
  inputs and outputs, sharing its weights.
- A `Parameter` or `Constant` read by several models - Loop bodies, blocks,
  the outer model - is one object in the saved model too: loaded once,
  listed once, frozen once.
- `export_keras` writes a stand-alone copy that `import_keras` gives back
  for every model: a weight read from several Keras graphs - a body rolled
  out by two Loops, a Parameter read inside and outside a body, a block also
  used in a Loop, several `Differentiate` of one relation - is written once
  per graph, with the same value. Keras could not rebuild such a file.
- `ModelyLoop(name, inputs, outputs, length=None)`: a model declared one step
  at a time, whose closed Inputs are all fed back by a single rollout. Every
  listed Input carries the horizon as its outermost sequence axis - `length`
  steps, or as many as the data has - and every output is a trajectory. A
  closed Input starts from its `init`; without one, from the first window of
  its data when it is listed, zeros where the data leaves it out, and from
  zero when it is not. A model called as a block keeps the feedback of its
  closed Inputs, one per call. `minimize` compares the trajectories of
  streams of the step, and `body` is the one-step `Modely`.
- `Input.close(stream, init=None)`: feeds `stream`, one sample of the Input,
  back into it at the next step, and returns `stream`. The Inputs of a step
  read each other's previous values, so coupled dynamics are written as they
  are, one Input per state variable. A `ModelyLoop` feeds them back; built as
  a `Modely`, the same graph is the one-step model, reading them from data.
- `Buffer(sw=n, init=None)(signal)`: the last `n` values of a signal computed
  one sample per step, the current one included - the window a model reads of
  what another computes, such as a controller's commands fed to a plant. Its
  past values are an Input closed with the signal.
- `Modely.all_inputs`: every Input a model reads - declared or not, inside
  the models it calls, or only by an objective - printed as a table of its
  shape and how it is fed: declared, the stream it is closed with, what the
  data must hold of it and what it starts from.
- `KerasFunction(function)`: any Keras layer, class or instance, as an
  nnodely layer - a custom function in the graph, its streams passed to
  `call` as separate arguments. The model saves, loads and exports with it.
- Membership functions as elementwise layers of their own, 1 at 0:
  `Triangular(radius)`, `Rectangular(radius)`, 1 on `[-left, right)`, and
  `Gaussian(sigma)`; the signal is shifted to center them,
  `Triangular(radius=0.5)([x - c])`. A `radius` is a number, a
  `(left, right)` pair or a `Parameter` of dim `(1,)` or `(2,)`, and `sigma`
  a number or a `Parameter` of dim `(1,)`: a Parameter is trained with the
  model. A width trained to zero or below is held at 1e-7, so the set
  collapses rather than giving NaN. A step has no gradient: a Parameter read
  by rectangles alone does not train, one read elsewhere too trains there.
- `Fuzzify(centers, function=...)` takes any nnodely layer: a membership class
  placed on each center as before, any other layer class or instance applied
  to `x - center` (`function=Sin` gives `sin(x - c)`), or a list of one
  function per center. The membership names (`"Triangular"`, ...) still work.
  A custom function is a Keras layer, class or instance, evaluated on every
  center and saved and exported with the model; its `call` reads one
  argument, `x - c`, and one needing more is refused. `function` has no
  default.
- `Fuzzify` is one layer: its Keras layer computes every membership at once,
  those placed on the centers in one operation for all of them, the bounds
  and the normalization included. The layers it applies to `x - center` are
  computed inside it, and the Parameters they read are predecessors of the
  Fuzzify.
- Trainable `Fuzzify` centers: besides numbers, `centers` is one stream of
  shape `(n,)` - a `Parameter` vector, trained with the model, or a stream
  computed from the inputs, such as a `Fir` with `features=n`, which gives
  every sample its own centers - or a list of scalar streams and numbers.
  The memberships are placed on them and sized by their neighbors as they are
  computed, so the sets move with them, widths included, and the gradient
  reaches them through both; they are used in the order given, ascending. A matrix, a time window or a list item that is not a scalar is
  refused.
- `Fuzzify(..., normalize=True)`: the memberships of each sample are divided
  by their sum, so that they add up to one (all zeros stay zeros), for
  memberships that are not negative; `normalize=False`, the default, keeps
  the values of the functions.
- `Fuzzify(..., bounds=...)` holds the signal within bounds before the
  functions read it: `True` between the lowest and the highest center
  (moving centers as they are computed), so a value beyond the centers is in
  the outer set instead of in none; `(left, right)` between two numbers;
  `False`, the default, not at all.
- `Fuzzify.plot()` draws the memberships with matplotlib as the layer
  computes them now, its Parameters at their current values: after
  training, the trained centers and widths.
- Freezing: `Parameter.freeze()` / `unfreeze()` (or `trainable`, also a
  constructor argument) keep one Parameter fixed through training or release
  it, and `Modely.freeze()` / `unfreeze()` do it for every Parameter a model
  computes with, those of the models it calls and of its Loop, Roll and
  OdeNet bodies included. Before or after `build()`; saved with the model.
  `Modely.parameters` lists them all, finds one by name
  (`model.parameters["kernel"]`), and prints as a table of their names,
  shapes, numbers of values and trainable flags.
- Early stopping in `train()`: the name of a monitored loss (made into a Keras
  `EarlyStopping` with `early_stopping_kwargs`), its configuration, or any
  callback, Keras's or your own.
- `validate()` scores every objective as training does - the same forward
  pass, target and padding mask, and the loss `train()` logs for validation
  data - with system-identification metrics: RMSE, MAE, max error, bias, error
  std, NRMSE, FIT, R², VAF and correlation. They are computed for each channel
  on its own scale, pooled or averaged for the whole signal, and leave out the
  rollout steps padded onto shorter simulations. A rollout also reports its
  RMSE at each step. Figures show prediction, error, parity, error
  distribution and the error along the rollout.
- Calculus blocks `Differentiate` (with respect to an input),
  `DerivativeStep` (with respect to time, over a window), `Derivative` (the
  same stencil one step at a time, in a `ModelyLoop`), `IntegrateStep`,
  `Integrate` (its rules one step at a time, in a `ModelyLoop`), `Ode`
  (Euler, midpoint, Heun, RK4) and `OdeNet` (neural ODEs, adaptive
  Dormand-Prince, hybrid events). `Integrate` and `Derivative` read one
  sample per step: the integral, or the samples before the step, are an
  Input closed with its update, fed back by the `ModelyLoop` with all its
  other closed Inputs, so they need no `Loop` of their own. `IntegrateStep`
  and `DerivativeStep` take their initial condition, a stream or a number, as
  the first argument of the call, `IntegrateStep(dt=dt)(init, rate)`, and
  start from zero when called on the signal alone; `Integrate` and
  `Derivative` take it at declaration, `init=`.
- Recurrence: `ModelyLoop` with `Input.close()`, `Modely.rollback()`, `Loop`
  and `Roll`. A body input its outputs never read, such as the target it is
  trained on, is not an input of a `Loop`, `Roll` or `OdeNet`.
- `LocalModel(input_function, output_function, vectorized=False)(inputs,
  activations)`: several activations are multiplied into joint memberships; a
  function is a factory of the cell index, giving each cell a layer of its
  own, a layer instance is shared by every cell, and `vectorized=True` calls
  it once on all the cells, which it returns on a last dim axis, as
  `Fir(features=N)` does.
- Layers new in 2.0: `BatchNorm`, `LeakyReLU`, `PReLU`, `Swish`, `GELU`,
  `Softplus`, `Asin`, `Acos`, `Atan`, `Exp`, `Log`, `Log10`, `Sqrt`, `Abs`,
  `Floor`, `Ceil`, `Deg2Rad`, `Clamp`, and `DataLoader.get_samples(n)`.
- Window layers `TimeSelect`, `TimeRange`, `TimeConcatenate` and
  `TimeReverse`, the counterparts of `Select`, `Range` and `Concatenate` on
  the window: they take no axis and always keep it, so the newest sample of
  a window, `TimeSelect(-1)`, combines with every sample of it.
- Sequence layers `SeqSelect`, `SeqRange`, `SeqConcatenate`, `SeqStack` and
  `SeqReverse`, on a sequence axis, fixed or dynamic, counted outermost
  first: `axis=0`, the default, is the rollout of the outermost `Loop`.
  `SeqSelect` removes the axis, and so does `SeqRange` when its range is one
  step long: `SeqSelect(-1)` of a collected trajectory, its last step, can
  close a `Loop` callback on an input without that axis. `SeqStack` stacks
  streams of one shape into a new sequence axis.
- `keep_dim` on `Select`: it drops the selected dim axis by default, and
  `keep_dim=True` keeps it with length 1; a stream keeps at least one dim
  axis, so selecting from a vector gives a scalar `(1,)`.
- `Fir` on a multi-dimensional input (1.x took scalars only): every element
  is filtered over its window alone, with a kernel of its own or, with
  `shared_kernel=True`, one for all. `features > 1` adds the channels as
  a trailing dim axis, laid out as the bias: `D=(2, 3)` becomes
  `D=(2, 3, 4)` with 4 channels, and a scalar becomes `D=(4,)` as in 1.x.
- `Fir(features=1, kernel=..., bias=...)` and `Linear(features=1,
  kernel=..., bias=...)`: their weights are `Parameter`s, predecessors of the
  layer in the graph. Each of `kernel` and `bias` is a `Parameter`, of the
  shape of `layer.kernel` or `layer.bias`, or a Keras initializer, by name or
  object, from which the layer makes one (`"<layer>_kernel"`,
  `"<layer>_bias"`). Assigning, training or freezing the Parameter changes
  the layer, and layers given one Parameter share it. `bias=False` leaves the
  bias out. A Fir draws the filter of each element alone.
- `Linear` maps the whole dim of a multi-dimensional input, `(*seq, time,
  *dim)` to `(*seq, time, features)`, or only one dim axis with `axis=k`: with a matrix and a
  bias for each position of the other dim axes, as the elements of a `Fir`, or
  with `shared_kernel=True` one for all.
- `save(path, weights=True)` / `Modely.load(path, weights=True)`, Keras and
  ONNX export, interactive HTML graphs, and Graphviz drawings: one image per
  model and per sub-model, or one flattened image, with a legend.
- Training printers (`"tiny"`, `"legacy"`, `"nnodely"`), `set_seed()` for
  reproducible runs, and `nnodely.__version__`.

### Changed

- Every stream is laid out `(batch, *seq, time, *dim)`, a `Parameter` or a
  `Constant` included: their value is repeated for every sample. The
  sequence axes come first, outermost first: where a `Loop` rolls out (and
  where a collected rollout is added). Then the window, `time`, where `Fir`,
  derivatives, integrals and the time selections act - or no window at all:
  `(batch, *seq, *dim)`. Then the dims. `Shape` holds `seq` (a tuple,
  possibly empty), `time` (an int, or `None` without a window) and `dim`,
  and prints as `Shape(S=(5,), T=3, D=(4,))`; `shape.dimensions` is
  `(seq, time, dim)`. The arrays given to a model or built by `DataLoader`
  are laid out the same, `(samples, *seq, time, *dim)`.
- An `Input` is declared once, with the shape of the data it reads, and does
  not change: `Input("x", dim=4, seq=5, sw=3)` is
  `Shape(S=(5,), T=3, D=(4,))`. `sw=n` is the last `n` samples up to and
  including the current one, `sw=(past, future)` adds future samples,
  `T = past + future`. Without `sw` the Input has no window: one value per
  sample, `(samples, *dim)` in the data. A sequence, `seq=5` or
  `seq=(4, -1)` (outermost first, `-1` or `"dyn"` dynamic), needs a window.
  There is no `sw()`, `next()` or `last()` on an Input: the layers that read
  it take what they need of its window, `TimeSelect(-1)(x)` its newest
  sample, `TimeRange(0, 3)(x)` its first three. `Differentiate(respect_to=x)`
  has the shape of `x`.
- Arithmetic between streams: windows must be equal, or one of them a single
  sample, `T=1`, which applies to every sample of the other's; a stream
  without a window read from data does not combine with one that has a
  window, while a value without one - a number, a `Parameter`, a
  `Constant` - applies to every sample of any stream. Sequences must be the
  same, or one a single step, padded with leading axes as numpy does; dims
  must be equal, or one a scalar; and one stream must already have the
  shape of the result, so a matrix of one sample is not combined with the
  window of a scalar. `Concatenate` repeats a single sample along the
  sequence and window of the others. The layers that act on the window -
  `Fir`, the `Time*` layers, `DerivativeStep`, `IntegrateStep`, `Roll`,
  `rollback` - refuse a stream without one.
- A `Parameter` or `Constant` `value` is all dim by default, without a
  window: an array of shape `(2, 3, 2)` is `D=(2, 3, 2)`. Given with a
  value, `dim`, `time` and `seq` name its axes, `time` one, the window, and
  `seq` one or several, outermost first (`dim=0, time=1, seq=2` makes a
  `(2, 3, 4)` value `S=(4,), T=3, D=(2,)`); without one, they are sizes.
  `.value` and the variable are laid out `(*seq, time, *dim)`, without the
  window when no `time` is given.
- `BatchNorm` normalizes along the first dim axis by default; `Softmax`,
  `PReLU` and `BatchNorm` take an explicit `axis` as a tensor axis, the batch
  as 0, as Keras does: a negative one counts from the last dim axis.
- A layer is identified by the Python object it comes from: applying one layer
  object several times shares its weights, and layers created apart never
  share them, whatever their names.
- Each minimizer logs its loss as `"<minimizer name>_loss"` in the training
  history, each batch weighing as many samples as it holds, as Keras weighs
  `"loss"`: a short last batch no longer counts as a full one.
- The `.nnodely` save format is version 7: a closed Input is saved with the
  stream it is closed with and its `init`. Saves and `.keras` exports made
  with the 2.0 pre-releases, in which windows were taken from an Input,
  streams had several time axes or were laid out `(batch, *dim, time,
  *seq)`, or closed Inputs were saved open, or a `Fuzzify` was a graph of one layer per center, do not load.
- The package installs no backend: install `nnodely[tensorflow]`,
  `nnodely[torch]` or `nnodely[jax]`; `nnodely[onnx]` adds ONNX export.

## Migrating from 1.x

| nnodely 1.x | nnodely 2.0 |
|---|---|
| `model = nnodely()` and `model.addModel("name", outputs)` | `model = Modely("name", inputs=[...], outputs=[...])` |
| `model.neuralizeModel(sample_time)` | `model.build()`; blocks that need a time step take it (`dt=`) |
| `model.addMinimize(name, a, b, loss_function=...)` | `model.minimize(name, source, target, loss=...)`: the prediction first |
| `model.removeMinimize(name)` | `model.remove_minimizer(name)` |
| `model.loadData(name, source)` | `data = DataLoader(model, source=...)` |
| `model.trainModel(...)` | `model.train(data, val_data=..., epochs=..., batch_size=..., lr=...)` |
| `model.analyzeModel(...)`, performance reports | `model.validate(data)` |
| `x.closedLoop(...)`, `x.connect(...)`, `prediction_samples=` | `x.close(stream)` in a `ModelyLoop`; `model.rollback({input: stream}, steps=n)`, `Loop`, `Roll` |
| `x.tw(seconds)`, `x.sw(samples)` | `Input("x", sw=samples)` or `sw=(past, future)`, declared once; `TimeSelect` / `TimeRange` take parts of the window |
| `saveModel` / `loadModel` | `model.save(path)` / `Modely.load(path)` |
| `exportPythonModel`, `saveTorchModel` | `model.export_keras(path)` |
| `exportONNX` / `onnxInference` | `model.export_onnx(path)` / `Modely.validate_onnx(file, inputs)` |
| `Fir(output_dimension, W=..., b=..., W_init=..., b_init=...)` | `Fir(features, kernel=..., bias=...)`: each a `Parameter` or an initializer, and `bias` also `True` or `False` |
| `Linear(output_dimension, W=..., b=..., W_init=..., b_init=...)` | `Linear(features, kernel=..., bias=...)`: each a `Parameter` or an initializer, and `bias` also `True` or `False` |
| `Relu` | `ReLU` |
| `Add`, `Sub`, `Mul`, `Div`, `Pow` | `+`, `-`, `*`, `/`, `**` between streams |
| `Neg` | `Negative` |
| `Differentiate` | `Differentiate(respect_to=x)` with respect to an input; with respect to time `DerivativeStep(dt=...)` over a window, `Derivative(dt=...)` one step at a time in a `ModelyLoop` |
| `ForwardEuler`, `RK2`, `RK4` | `Ode(f, states, dt, method=...)` |
| `NeuralODE` | `OdeNet` |
| `Part` | `Range` |
| `SamplePart`, `TimePart` | `TimeRange` (in samples) |
| `SampleSelect` | `TimeSelect` (`Select` is the same in both) |
| `Integrate(stream, method=...)` | `IntegrateStep(dt=...)(init, rate)`; 2.0's `Integrate` integrates one step at a time in a `ModelyLoop` |
| `LocalModel(input_function, output_function, pass_indexes=...)` | `LocalModel(input_function, output_function, vectorized=False)(inputs, activations)`: a function is a factory of the cell index, a layer instance is shared by every cell |
| `Fuzzify(output_dimension, range=..., functions=...)` | `Fuzzify(centers, function=...)`: the centers listed explicitly; `normalize=True` makes the memberships add up to one |
| `Optimizer`, `SGD`, `Adam` | `train(optimizer="sgd" / "adam" / a Keras optimizer)` |
| `Parameter(name, dimensions, sw=..., values=..., init=...)`, `Constant(name, values, sw=...)` | `Parameter(name, value=..., time=...)` or `Parameter(name, dim=..., time=..., initializer=...)`, `Constant(name, value=..., time=...)` |
| `ParamFun` | the function written with streams and `Parameter`s, or a custom `Layer` |
| `clearNames` | not needed: names never decide which layers share weights |

`Cosh`, `Sech` and `SampleTime` have no 2.0 counterpart; `Identity` is not
exported (it is `nnodely.core.layer.Identity`).

[2.0.0]: https://github.com/tonegas/nnodely/compare/v1.5.4...v2.0.0
