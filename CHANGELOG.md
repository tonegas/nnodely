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
  `header` are passed to `pandas.read_csv`; n/a values are reported.
- `minimize()` with any Keras loss, a `gain` per objective and `seq_weights`
  along a rollout; targets can be computed streams or numbers.
- A layer whose constructor takes only `name` is applied as it is created:
  `Sin(x)` is `Sin()(x)`. These are `Sin`, `Cos`, `Tan`, `Asin`, `Acos`,
  `Atan`, `Atan2`, `Exp`, `Log`, `Log10`, `Sqrt`, `Abs`, `Floor`, `Ceil`,
  `Deg2Rad`, `Sign`, `Negative`, `Sigmoid`, `Tanh`, `Swish`, `Softplus` and
  `TimeConcatenate`; layers with settings (`ReLU`, `GELU`, `Clamp`, `Sum`,
  ...) are configured first.
- An activation, a math or trigonometric layer, `Clamp` or a membership
  function called on a list of streams is applied to each and returns a list:
  `Sin([a, b])` gives `[sin(a), sin(b)]`.
- `Atan2([y, x])`, the two-argument arctangent: the angle of the point
  `(x, y)` in `[-pi, pi]`, with its quadrant.
- `KerasFunction(function)`: any Keras layer, class or instance, as an
  nnodely layer - a custom function in the graph, its streams passed to
  `call` as separate arguments. The model saves, loads and exports with it.
- Membership functions as layers of their own: `Triangular(center, radius)`,
  `Rectangular(center, radius)`, 1 on `[center - left, center + right)` (a
  `radius` can be a `(left, right)` pair),
  and `Gaussian(center, sigma)`.
- `Fuzzify(centers, function=...)` takes any nnodely layer: a membership class
  placed on each center as before, any other layer class or instance applied
  to `x - center` (`function=Sin` gives `sin(x - c)`), or a list of one
  function per center. The membership names (`"Triangular"`, ...) still work.
  A custom function is a Keras layer, class or instance, evaluated on every
  center and saved and exported with the model; its `call` reads one
  argument, `x - c`, and one needing more is refused. The layer of each
  center is named after its class, such as `Fuzzify1_Triangular_0`.
- Trainable `Fuzzify` centers: besides numbers, `centers` is one stream of
  shape `(n,)` - a `Parameter` vector, trained with the model, or a stream
  computed from the inputs, such as a `Fir` with `features=n`, which gives
  every sample its own centers - or a list of scalar streams and numbers.
  The memberships are placed on them and sized by their neighbors as they are
  computed, so the gradient reaches them; they are used in the order given,
  ascending. A matrix, a time window or a list item that is not a scalar is
  refused.
- Freezing: `Parameter.freeze()` / `unfreeze()` (or `trainable`, also a
  constructor argument) keep one Parameter fixed through training or release
  it, and `Modely.freeze()` / `unfreeze()` do it for every Parameter a model
  computes with, those of the models it calls and of its Loop, Roll and
  OdeNet bodies included. Before or after `build()`; saved with the model.
  `Modely.parameters` lists them all, and prints as a table of their names,
  shapes, numbers of values and trainable flags.
- Early stopping in `train()`: the name of a monitored loss (made into a Keras
  `EarlyStopping` with `early_stopping_kwargs`), its configuration, or any
  callback, Keras's or your own.
- `validate()` with system-identification metrics (RMSE, MAE, FIT, R², ...)
  and figures.
- Calculus blocks `Differentiate` (with respect to an input), `Derivative`
  (with respect to time),
  `IntegrateStep`, `Integrate` (an `IntegrateStep` rolled out along a
  horizon by a `Loop`), `Ode` (Euler, midpoint, Heun, RK4) and `OdeNet` (neural ODEs,
  adaptive Dormand-Prince, hybrid events).
- Recurrence: `Modely.rollback()`, `Loop` and `Roll`.
- `LocalModel(input_function, output_function, vectorized=False)(inputs,
  activations)`: several activations are multiplied into joint memberships; a
  function is a factory of the cell index, giving each cell a layer of its
  own, a layer instance is shared by every cell, and `vectorized=True` calls
  it once on all the cells.
- Layers new in 2.0: `BatchNorm`, `LeakyReLU`, `PReLU`, `Swish`, `GELU`,
  `Softplus`, `Asin`, `Acos`, `Atan`, `Exp`, `Log`, `Log10`, `Sqrt`, `Abs`,
  `Floor`, `Ceil`, `Deg2Rad`, `Clamp`, and `DataLoader.get_samples(n)`.
- Sequence layers `SeqSelect`, `SeqRange`, `SeqConcatenate`, `SeqStack`
  and `SeqReverse`, on the outer time axes, fixed or dynamic, counted
  outermost first (`axis=0`, the default, is the outermost). `SeqSelect`
  removes the axis, so `SeqSelect(-1)` of a collected trajectory, its last
  step, can close a `Loop` callback on an input without that axis.
- `Fir` on a multi-dimensional input (1.x took scalars only): every element
  is filtered over its window alone, with a kernel of its own or, with
  `shared_kernel=True`, one for all. `features > 1` adds the channels as
  a leading dim axis: `D=(2, 3)` becomes `D=(4, 2, 3)` with 4 channels, and a
  scalar becomes `D=(4,)` as in 1.x.
- `Fir(features, kernel=..., bias=...)` and `Linear(features,
  kernel=..., bias=...)`: their weights are `Parameter`s, predecessors of the
  layer in the graph. Each of `kernel` and `bias` is a `Parameter`, of the
  shape of `layer.kernel` or `layer.bias`, or a Keras initializer, by name or
  object, from which the layer makes one (`"<layer>_kernel"`,
  `"<layer>_bias"`). Assigning, training or freezing the Parameter changes
  the layer, and layers given one Parameter share it. `bias=False` leaves the
  bias out. A Fir draws the filter of each element alone.
- `Linear` maps the whole dim of a multi-dimensional input, `(*time, *dim)`
  to `(*time, features)`, or only one dim axis with `axis=k`: with a matrix and a
  bias for each position of the other dim axes, as the elements of a `Fir`, or
  with `shared_kernel=True` one for all.
- `save(path, weights=True)` / `Modely.load(path, weights=True)`, Keras and
  ONNX export, interactive HTML graphs, and Graphviz drawings: one image per
  model and per sub-model, or one flattened image, with a legend.
- Training printers (`"tiny"`, `"legacy"`, `"nnodely"`), `set_seed()` for
  reproducible runs, and `nnodely.__version__`.

### Changed

- Every stream is laid out `(batch, *time, *dim)`, a `Parameter` or a
  `Constant` included: their value is repeated for every sample. The time
  axes come outermost first: the outer time, where a `Loop` rolls out (and
  where a collected rollout is added), then the inner time, the window of
  samples, where windows, `Fir`, derivatives and integrals act. A stream may
  have no time axis: a number, a `Parameter` without time, an `Input` read
  whole. `Shape` holds `time` (a tuple) and `dim`; `shape.dimensions` is
  `(time, dim)`. The arrays given to a model or built by `DataLoader` are
  laid out the same, `(samples, *time, *dim)`.
- The sequence of an `Input`, its outer time, is declared on its windows:
  `x.sw(3, seq=5)`, `x.last(seq=(4, -1))` (outermost first, `-1` dynamic).
  An input has one sequence, set by its first window; a later window may
  leave `seq` out, but not change it.
- Arithmetic between streams: streams with time must have the same time
  axes, and one without time applies to every sample; dims must be equal,
  or one a scalar; and one stream must already have the shape of the result,
  so a matrix without time is not combined with the window of a scalar.
  `Concatenate` repeats a stream without time along the time of the others.
- A `Parameter` or `Constant` `value` is all dim by default, with no time
  axis: an array of shape `(2, 3, 2)` is `D=(2, 3, 2)`. Given with a value,
  `dim` and `time` name its axes, `time` one or several, outermost first
  (`dim=0, time=(2, 1)` makes it `T=(2, 3), D=(2,)`); without one, they are
  sizes. `.value` is laid out `(*time, *dim)`.
- `BatchNorm` normalizes along the first dim axis by default; `Softmax`,
  `PReLU` and `BatchNorm` take an explicit `axis` as a tensor axis, the batch
  as 0.
- A layer is identified by the Python object it comes from: applying one layer
  object several times shares its weights, and layers created apart never
  share them, whatever their names.
- Each minimizer logs its loss as `"<minimizer name>_loss"` in the training
  history.
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
| `x.closedLoop(...)`, `x.connect(...)`, `prediction_samples=` | `model.rollback({input: stream}, steps=n)`, `Loop`, `Roll` |
| `x.tw(seconds)` | `x.sw(samples)`; `x.last()` and `x.next()` for one sample |
| `saveModel` / `loadModel` | `model.save(path)` / `Modely.load(path)` |
| `exportPythonModel`, `saveTorchModel` | `model.export_keras(path)` |
| `exportONNX` / `onnxInference` | `model.export_onnx(path)` / `Modely.validate_onnx(file, inputs)` |
| `Fir(output_dimension, W=..., b=..., W_init=..., b_init=...)` | `Fir(features, kernel=..., bias=...)`: each a `Parameter` or an initializer, and `bias` also `True` or `False` |
| `Linear(output_dimension, W=..., b=..., W_init=..., b_init=...)` | `Linear(features, kernel=..., bias=...)`: each a `Parameter` or an initializer, and `bias` also `True` or `False` |
| `Relu` | `ReLU` |
| `Add`, `Sub`, `Mul`, `Div`, `Pow` | `+`, `-`, `*`, `/`, `**` between streams |
| `Neg` | `Negative` |
| `Differentiate` | `Differentiate(respect_to=x)` with respect to an input, `Derivative(dt=...)` with respect to time |
| `ForwardEuler`, `RK2`, `RK4` | `Ode(f, states, dt, method=...)` |
| `NeuralODE` | `OdeNet` |
| `Part` | `Range` |
| `SamplePart`, `TimePart` | `TimeRange` (in samples) |
| `SampleSelect` | `TimeSelect` (`Select` is the same in both) |
| `Integrate(stream, method=...)` | `IntegrateStep(dt=..., init=...)`; 2.0's `Integrate` integrates along a horizon |
| `LocalModel(input_function, output_function, pass_indexes=...)` | `LocalModel(input_function, output_function, vectorized=False)(inputs, activations)`: a function is a factory of the cell index, a layer instance is shared by every cell |
| `Fuzzify(output_dimension, range=..., functions=...)` | `Fuzzify(centers, function=...)`: the centers listed explicitly |
| `Optimizer`, `SGD`, `Adam` | `train(optimizer="sgd" / "adam" / a Keras optimizer)` |
| `Parameter(name, dimensions, sw=..., values=..., init=...)`, `Constant(name, values, sw=...)` | `Parameter(name, value=..., time=...)` or `Parameter(name, dim=..., time=..., initializer=...)`, `Constant(name, value=..., time=...)` |
| `ParamFun` | the function written with streams and `Parameter`s, or a custom `Layer` |
| `clearNames` | not needed: names never decide which layers share weights |

`Cosh`, `Sech` and `SampleTime` have no 2.0 counterpart; `Identity` is not
exported (it is `nnodely.core.layer.Identity`).

[2.0.0]: https://github.com/tonegas/nnodely/compare/v1.5.4...v2.0.0
