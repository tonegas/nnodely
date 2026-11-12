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
  of a larger model. Building again keeps the weights the graph holds;
  `build(keep_weights=False)` gives every weight the model computes with its
  initial value again, in place - frozen ones, those of the blocks it calls
  and of its Loop bodies, the Keras weights of a `KerasFunction` and the
  statistics of a `BatchNorm` included. A build that fails leaves the model
  as it was. Both take a dict keyed by Input or input name, or a list
  in the order of the model's inputs.
- `DataLoader` for dicts, DataFrames and CSV files or folders, with several
  simulations, sequences and padding masks. `delimiter` and
  `header` are passed to `pandas.read_csv`; n/a values are reported. An array
  holds the features of each sample laid out as the input's `dim`, or flat in
  row-major order; another layout of as many values is refused rather than
  reordered. `windowed=True` takes arrays already laid out as the model's
  tensors, `(N, *seq, tw, *dim)`, and uses them as given: dynamic sequence
  lengths, several per input, are read from their shape, and rollouts of
  different lengths are padded and masked.
- `minimize()` with any Keras loss and a `gain` per objective; targets can
  be computed streams or numbers.
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
  Input too.
- A `Parameter` or `Constant` read by several models - Loop bodies, blocks,
  the outer model - is one object in the saved model too: loaded once,
  listed once, frozen once. So is an Input a block reads without declaring
  it, and a model called in several places.
- `export_keras` writes a stand-alone copy that `import_keras` gives back
  for every model: a weight read from several Keras graphs - a body rolled
  out by two Loops, a Parameter read inside and outside a body, a block also
  used in a Loop, several `Gradient` of one relation - is written once
  per graph, with the same value. Keras could not rebuild such a file.
- `ModelyLoop(name, inputs, outputs, length=None)`: a model declared one step
  at a time, whose fed-back Inputs are all fed back by a single rollout - one
  `Loop` over its step. Every listed Input carries the horizon as its
  outermost sequence axis - `length` steps, or as many as the data has - and
  every output is a trajectory. A fed-back Input starts from the first window
  of its data when it is listed, zeros where the data leaves it out, and from
  zero when it is not. A model called as a block keeps the feedback of its
  fed-back Inputs, one per call. `minimize` compares the trajectories of
  streams of the step, and `body` is the one-step `Modely`.
- `Input.feedback(stream)`: feeds `stream`, one sample of the Input, back
  into it at the next step, and returns `stream`. The Inputs of a step read
  each other's previous values, so coupled dynamics are written as they are,
  one Input per state variable. A `Loop` or a `ModelyLoop` feeds them back;
  built as a `Modely`, the same graph is the one-step model, reading them
  from data. The initial value is given where the feedback is applied:
  `Loop(init=)`, or the data of the Input in a `ModelyLoop`.
- `Loop` finds the Inputs its body feeds back - with `Input.feedback`, inside
  the models it calls, or kept by `Integrate`, `Derivative` and `Buffer` - as
  a `ModelyLoop` does, adding an output for an update that is not one:
  `Loop(f=step)(inputs)`, the inputs given as a dict, a list in the order
  the step declares them, or a stream alone. `feedback=` adds to them, for a model that is built
  or loaded, or to reuse one step with different feedbacks. `init=` takes
  only the fed-back Inputs, the inputs of the call only the others, and a
  state has one initial value: two are an error. The `init` of `Integrate`
  that is an Input of the step is an input of the `Loop`, bound to a stream
  of the outer graph. A `Loop` returns every output of its body, in its
  order: with one, it is that output, a stream; with more, it is all of
  them, unpacked (`x, v = Loop(...)(inputs)`) or indexed (`loop[1]`), and
  read as one stream it is an error. An output it adds for an update is not
  returned.
- `Buffer(tw=n, init=None)(signal)`: the last `n` values of a signal computed
  one sample per step, the current one included - the window a model reads of
  what another computes, such as a controller's commands fed to a plant. Its
  past values are a hidden Input fed back with the signal, and `init` holds
  them before the first step as the `init` of `Derivative` holds the samples
  its stencil reads.
- `Modely.inputs`: every Input a model reads - declared or not, inside
  the models it calls, or only by an objective - and every state of the Loops
  it rolls out, printed as a table of its shape and how it is fed: declared,
  the stream it is fed back with, what the data must hold of it and what it
  starts from. A ModelyLoop prints the rows of the Loop it is a shortcut for,
  its Inputs with the horizon.
- `Modely.outputs`: the outputs a model is declared with, in that order,
  printed as a table of their shapes, the stream each names - in a
  ModelyLoop, the stream of the step it collects - the Inputs a loop feeds
  that stream back into and the minimizers that read it.
- `Modely.minimizers`: the minimizers of a model - those its `train()`
  minimizes, not those of the models it calls - printed as a table of their
  source and target, loss and gain. Setting the gain or loss of a row
  changes that minimizer in the model, from the next `train()`:
  `model.minimizers[0]["gain"] = 0.1`.
- The tables a model prints - `inputs`, `outputs`, `models`, `parameters`
  and `minimizers` - are drawn as boxes, the table alone: its first column
  is named after what it lists (`input`, `output`, `model`, `parameter`,
  `minimizer`), and the `parameters` table ends with a `total` row of the
  values, all and trainable. A model prints (`print(model)`, its repr) as one
  too: its name, then the names of the Inputs it reads, its outputs and its
  minimizers. The `"legacy"` training printer draws its loss table the same
  way, the training time as its last row. In the nnodely colours when the
  standard output is a terminal or a notebook at the time the table is turned
  into text, plain otherwise. `NO_COLOR` prints them plain - for a table
  written to a file from a terminal - and `FORCE_COLOR` colours them anyway.
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
  computes with, those of the models it calls and of its Loop bodies
  included. Given a name or a list of names, they act on those alone: a
  Parameter by its name, or a model the graph reaches - a block, a Loop
  body, the model of a LocalModel - by its name, for all its Parameters:
  `model.freeze(["k", "g"])`, `model.freeze("submodel")`. A name several
  carry raises, listing them; one is frozen by itself,
  `model.parameters[2].freeze()`. A name that matches nothing raises, and
  nothing is frozen. Before or
  after `build()`; saved with the model.
  `Modely.parameters` lists them all, finds one by name
  (`model.parameters["kernel"]`), and prints as a table of their names,
  shapes, numbers of values and trainable flags - a section per model when
  the models it calls have Parameters, each under the first model whose
  graph reaches it - and a total of the values, all and trainable.
- Name tags. Nothing keeps two objects from sharing a name - two layers named
  alike, a block and the model calling it, two copies of one loaded model,
  which only names reach. Every call taking a name - `parameters[...]`,
  `models[...]`, `inputs[...]`, `outputs[...]`, `minimizers[...]`, the sides
  of `minimize`, `freeze` and `unfreeze` - looks it up among the objects the
  graph reaches, in that order, the model's own first: a name several carry
  raises `ValueError` listing them, each with its index and the model it is
  in. An item is picked by its name or by its index, never both: every table
  - `parameters`, `models`, `inputs`, `outputs`, `minimizers` - prints an
  `index` column, and `model.parameters[2]` picks that row, the same after
  save and load; renaming one, `obj.name = "new_name"`, gives each its own. Inside nnodely the
  objects are used, not their names, so anything can be renamed and is still
  found: a renamed Input is read by its new name, a renamed output returned
  under it, a Loop body keeps rolling out, a `Gradient` follows its input;
  `model.minimizers["fit"].name = "track"` renames a minimizer, as any
  object is renamed; `remove_minimizer` and a `validate` result take a name
  or an index. Training
  and validation ask for a new `build()` after an Input or output is
  renamed. Two calls of one block, or a stream named like an output, are
  minimized apart; HTML graphs draw every node and
  every call apart (`blk_call`, `blk_call~2`). `inputs` gives the Inputs of
  the graph, one row per object - two renamed alike are listed apart - so a
  model can read one a loaded block reads; `outputs` tells the stream an
  Input is fed back with by its object, and a ModelyLoop's state shares the
  row of the Input it starts from, renamed or not. Input and output
  names stay one per model, the keys of the data: two Inputs of one name are
  refused at build, and renamed alike after it, when the model is run on
  data. Saved so, they load as the two Inputs they are, which a rename lets
  run; a block whose inputs share a name is called with a list, by position,
  or with a dict keyed by the Inputs themselves. A model or a block called
  with a dict resolves every key to its Input at once, as `inputs[...]`
  does: a name several Inputs carry raises listing them, a name none carries
  is ignored, as another column of the data, and an Input of another model
  raises (once taken for the model's own of its name).
  The state a layer keeps,
  named like another Input of the model, takes a free name. Node classes are
  saved with their module, so a class of your own named like one of
  nnodely's - a `Sin` - never takes its place on load; a class saved by its
  name alone loads when one class has that name.
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
  distribution and the error along the rollout, one file per objective,
  `<model>_<index>_<objective>.png`. The result is the list of the scores,
  picked by objective name or index as the other tables are.
- Calculus blocks `Gradient` (with respect to an input), `DerivativeStep`
  (with respect to time, over a window), `Derivative` (the same stencil one
  step at a time, in a `ModelyLoop`), `IntegrateStep`, `Integrate` (its rules
  one step at a time, in a `ModelyLoop`) and `Ode` (Euler, midpoint, Heun,
  RK4). `DerivativeStep(dt=dt)` and `Derivative(dt=dt)` are first
  derivatives, the backward difference of order `poly_order` - 1 by
  default, `(x[i] - x[i-1]) / dt`, each order more one more sample read and
  exact on a polynomial of one degree more; the second derivative is one
  applied to the other's result.
  `Integrate` and `Derivative` read one sample per step: the integral
  (`<name>_<rate>`, and `<name>_<rate>_rate` for the trapezoid), or the
  samples before the step (`<name>_<signal>`), are hidden Inputs fed back by the
  `ModelyLoop` or `Loop` with all its other fed-back Inputs, so they need no
  `Loop` of their own; their result is named after the block, and their
  `init` is the only initial value of that state. `IntegrateStep` and
  `DerivativeStep` take their initial condition, a stream or a number, as
  the first argument of the call, `IntegrateStep(dt=dt)(init, rate)`, and
  start from zero when called on the signal alone; `Integrate`,
  `Derivative` and `Buffer` take it at declaration, `init=`, each checked by
  one rule: one sample, or the samples read before the first, a scalar dim
  or the signal's, no sequence or the signal's. The trapezoidal rule
  averages the first rate with the rate before it: `init_rate`, called on
  after `init`, `IntegrateStep(method="trapezoidal", dt=dt)(init, init_rate,
  rate)`, or configured, `Integrate(..., init_rate=)`; zero without one.
- Recurrence: `ModelyLoop` with `Input.feedback()`, and `Loop`. A body input
  its outputs never read, such as the target it is trained on, is not an
  input of a `Loop`.
- `LocalModel(input_function, output_function, vectorized=False)(inputs,
  activations)`: several activations are multiplied into joint memberships; a
  function is a factory of the cell index, giving each cell a layer of its
  own, a layer instance is shared by every cell, and `vectorized=True` calls
  it once on all the cells, which it returns on a last dim axis, as
  `Fir(features=N)` does. Each call is a block, its cells a model of their
  own, which `export_html` shows on a page of its own.
- `EquationLearner(functions, linear_in=, linear_out=, threshold=)`: an input
  projection gives each function its arguments, and `linear_out` combines
  the results. Functions are names - the trigonometric, math and activation
  layers, `gaussian`, `triangular`, `rectangular`, and `add`, `subtract`,
  `multiply`, `divide`, `power`, `atan2` of two arguments - layer classes or
  instances, or callables on streams. `threshold` drops the coefficients of
  both projections below it from what the model computes, while they keep
  training; `learner.threshold` can be changed on a built model, which is
  saved with it. `learner.plot()` draws the learned equation as LaTeX and
  returns it as a sympy `Eq`.
- Stream operators `-x`, `abs(x)`, `+x`, `2 ** x` and `x % y`, the modulo
  taking the sign of the divisor, as Python's does.
- Math layers `Sinh`, `Cosh`, `Asinh`, `Acosh`, `Atanh`, `Expm1`, `Log1p`,
  `Log2`, `Square`, `Round` and `Rad2Deg`; `Maximum([a, b, ...])` and
  `Minimum([a, b, ...])`, elementwise and broadcast as the operators are, so
  that a bound can be a learned Parameter; `Mean`, `Max`, `Min` and `Norm`
  over the dim axes, as `Sum`, `Norm` with gradient 0 at the zero vector;
  and `Cross(axis=-1)([a, b])`, the cross product of 3-vectors. The
  elementwise ones are EquationLearner names too: `sinh`, `cosh`, `asinh`,
  `acosh`, `atanh`, `expm1`, `log1p`, `log2`, `square`, `round`, and of two
  arguments `maximum`, `minimum`, `mod`.
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
  be fed back by a `Loop` into an input without that axis. `SeqStack` stacks
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
  bias out. A Fir draws the filter of each element alone. The slopes of a
  `PReLU` (`"<layer>_alpha"`) and the scale and offset of a `BatchNorm`
  (`"<layer>_gamma"`, `"<layer>_beta"`) are Parameters too, listed and
  frozen with the others; a BatchNorm holds its moving statistics itself.
- `Linear` maps the whole dim of a multi-dimensional input, `(*seq, tw,
  *dim)` to `(*seq, tw, features)`, or only one dim axis with `axis=k`: with a matrix and a
  bias for each position of the other dim axes, as the elements of a `Fir`, or
  with `shared_kernel=True` one for all.
- `save(path, weights=True)` / `Modely.load(path, weights=True)`, in
  cascade: the path is a folder with the model's graph and weights, and
  every model it calls as a block, or rolls out in a `Loop`, saved the same
  way in a folder of its own, nested as the models are. The loaded model
  calls the same models - `models` lists them, `freeze` reaches them by
  name, one called several times is one model again - keeps the feedback of
  the fed-back Inputs they declare, and each folder loads on its own. Keras
  and ONNX export, and interactive HTML graphs. The HTML graph colours every node
  by its role, with a legend that highlights a role,
  a panel that explains the node selected - its shape axis by axis, what it
  reads and feeds - search, and the flattened page draws each sub-model's
  body in a region of its own. Selecting a node keeps its colours, and the
  layout switches between left to right and force at any size.
- Training printers (`"tiny"`, `"legacy"`, `"nnodely"`), `set_seed()` for
  reproducible runs, and `nnodely.__version__`. The `"nnodely"` console
  charts the total loss along the epochs, on a log scale. In a notebook,
  `"nnodely"` and `"tiny"` redraw one output of the cell in place, a few
  times a second, instead of appending a frame at every repaint.
- Tutorials: 13 Jupyter notebooks in `docs/tutorial`, shown in the
  documentation with their outputs - model design and the `(S*, T, D*)` shape
  convention, data management, training, validation and the HTML graph, save,
  load and export, `Parameter` and `Constant`, partitioning, `Fir` and
  `Linear`, `Fuzzify` and `LocalModel`, recurrent models (`ModelyLoop`,
  `Loop`, `Buffer`), calculus (`IntegrateStep`, `DerivativeStep`,
  `Integrate`, `Derivative`, `Gradient`), custom functions and layers
  (`KerasFunction`, `Layer` subclasses) and the `EquationLearner`. Each guide
  page links its tutorials.

### Changed

- Every stream is laid out `(batch, *seq, tw, *dim)`, a `Parameter` or a
  `Constant` included: their value is repeated for every sample. The
  sequence axes come first, outermost first: where a `Loop` rolls out (and
  where a collected rollout is added). Then the time window, `tw`, where `Fir`,
  derivatives, integrals and the time selections act - or no window at all:
  `(batch, *seq, *dim)`. Then the dims. `Shape` holds `seq` (a tuple,
  possibly empty), `tw` (an int, or `None` without a window) and `dim`,
  and prints as `Shape(S=(5,), T=3, D=(4,))`, a stream as
  `Input(name='x', shape=Shape(...))`; `shape.dimensions` is
  `(seq, tw, dim)`. The arrays given to a model or built by `DataLoader`
  are laid out the same, `(samples, *seq, tw, *dim)`. `tw` is the one name
  of the window everywhere: `Input(tw=)`, `Buffer(tw=)`, `Parameter(tw=)`,
  `Constant(tw=)`, `Shape(tw=)`, `stream.tw`, `shape.tw_axis`.
- An `Input` is declared once, with the shape of the data it reads, and does
  not change: `Input("x", dim=4, seq=5, tw=3)` is
  `Shape(S=(5,), T=3, D=(4,))`. `tw=n` is the last `n` samples up to and
  including the current one, `tw=(past, future)` adds future samples,
  `T = past + future`. Without `tw` the Input has no window: one value per
  sample, `(samples, *dim)` in the data. A sequence, `seq=5` or
  `seq=(4, -1)` (outermost first, `-1` or `"dyn"` dynamic), needs a window.
  `dim` is one feature by default, each axis at least 1: `dim=0` or `dim=()`
  is refused. There is no `sw()`, `next()` or `last()` on an Input: the layers that read
  it take what they need of its window, `TimeSelect(-1)(x)` its newest
  sample, `TimeRange(0, 3)(x)` its first three. `Gradient(wrt=x)`
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
  `Fir`, the `Time*` layers, `DerivativeStep`, `IntegrateStep` - refuse a
  stream without one.
- A `Parameter` or `Constant` declares its shape as an `Input` does, with
  `tw` for the window: `seq`, `tw` and `dim` are sizes - `dim` one
  feature by default, each axis at least 1, and a sequence of fixed lengths
  that needs a window. A `value` fills that shape: a list of numbers holds
  exactly as many, read in the order `(*seq, tw, *dim)`
  (`value=[1, ..., 10], tw=2, dim=5`); an array of several axes has the
  declared sizes, in any order, its axes moved into place (a `(2, 3, 4, 5)`
  array with `seq=5, tw=3, dim=(2, 4)` is `S=(5,), T=3, D=(2, 4)`).
  Anything else raises `ValueError`. Without `dim`, `seq` and `tw` take
  the axes of the value of their sizes, and the axes they leave are the dim,
  in their order (a `(2, 4, 6)` value with `tw=4` is `T=4, D=(2, 6)`); with
  neither, every axis is dim and the value has no window: `value=[1, 2, 3]`
  is `D=(3,)`. A `Constant` requires a value; a `Parameter` without one is
  drawn by its `initializer` in the declared shape. `.value` and the
  variable are laid out `(*seq, tw, *dim)`, without the window when no
  `tw` is given.
- `BatchNorm` normalizes along the first dim axis by default; `Softmax`,
  `PReLU` and `BatchNorm` take an explicit `axis` as a tensor axis, the batch
  as 0, as Keras does: a negative one counts from the last dim axis.
- A layer is identified by the Python object it comes from: applying one layer
  object several times shares its weights, and layers created apart never
  share them, whatever their names.
- Each minimizer logs its loss as `"<minimizer name>_loss"` in the training
  history, each batch weighing as many samples as it holds, as Keras weighs
  `"loss"`: a short last batch no longer counts as a full one.
- The `.nnodely` save format is version 14: a model is saved with every
  model it calls in a folder of its own, a fed-back Input with the stream it
  is fed back with and the `init` of the layer that keeps it, and a `Loop`
  with the outputs it returns. Saves and `.keras` exports made
  with the 2.0 pre-releases, in which windows were taken from an Input,
  streams had several tw axes or were laid out `(batch, *dim, tw,
  *seq)`, or fed-back Inputs were saved open, or the window was saved as `time` and `sw`, or a `Fuzzify` was a graph of one layer per center, or a `PReLU` or `BatchNorm` held its weights in its Keras layer, or blocks were inlined, or a `Parameter` or `Constant` was saved with the axes of its value in place of their sizes, or a `Loop`, a `Gradient` or the optional inputs were saved by name, do not load;
  nor do saves of an earlier format version.
- The stream fed back into an Input is named `feedback` everywhere:
  `Input.feedback(stream)` and `Loop(feedback=)` (was `callback=`). The old
  names are gone, and so is `Input.close`; the initial value of a fed-back
  Input is given where the feedback is applied. The Keras layer of a `Loop`
  names it `feedback` in its configuration too, so a `.keras` export of one
  made with 2.0.0 does not load.
- A `Loop` over a body with several outputs is no longer a stream of its
  first `callback` output: it returns them all, and read as one stream it
  raises.
- `Integrate` (now `IntegrateStep`) takes `method=`, the name `Ode` uses, for
  2.0.0's `solver=`. The trapezoidal rule no longer keeps a rectangle on the
  first interval: it averages the first rate with the one before it,
  `init_rate`, zero without one. `Derivative(respect_to=x)` with respect to
  an Input is `Gradient(wrt=x)`.
- `Modely.inputs` is the table of every Input the model reads (see Added).
  In 2.0.0 it was the list of the Inputs the model was declared with, which
  is no longer an attribute: keep the Inputs a model is declared with, to
  call it as a block in their order. Attributes that are not part of the API
  start with an underscore, as in Python: `Modely.order`, a node's `preds`,
  `DataLoader.dataset` and the settings a layer keeps, such as `Loop.f` or
  `Linear.features`. `Modely.minimizers` is the table of the minimizers (see
  Added): in 2.0.0 it was the list the model trains on. Its rows set the
  gain and loss of a minimizer; `minimize` declares one and
  `remove_minimizer` removes one.
- The package installs no backend: install `nnodely[tensorflow]`,
  `nnodely[torch]` or `nnodely[jax]`; `nnodely[onnx]` adds ONNX export.

### Removed

- `Roll`, `OdeNet` and `Modely.rollback()`. A model rolled out on its own
  predictions is a `Loop` or a `ModelyLoop`; a neural ODE is an `Ode` step,
  the layers of its field created outside `f`, rolled out by a `Loop`.
  Saves and `.keras` exports holding one of them do not load.
- The `seq_weights` of `minimize()`, in 2.0.0 the weight of each step of a
  rollout or window. It is planned for a future version.

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
| `x.closedLoop(...)`, `x.connect(...)`, `prediction_samples=` | `x.feedback(stream)` in a `ModelyLoop` or a `Loop` |
| `x.tw(seconds)`, `x.sw(samples)` | `Input("x", tw=samples)` or `tw=(past, future)`, declared once - `tw` counts samples, not seconds as 1.x's `tw()`; `TimeSelect` / `TimeRange` take parts of the window |
| `saveModel` / `loadModel` | `model.save(path)` / `Modely.load(path)` |
| `exportPythonModel`, `saveTorchModel` | `model.export_keras(path)` |
| `exportONNX` / `onnxInference` | `model.export_onnx(path)` / `Modely.validate_onnx(file, inputs)` |
| `Fir(output_dimension, W=..., b=..., W_init=..., b_init=...)` | `Fir(features, kernel=..., bias=...)`: each a `Parameter` or an initializer, and `bias` also `True` or `False` |
| `Linear(output_dimension, W=..., b=..., W_init=..., b_init=...)` | `Linear(features, kernel=..., bias=...)`: each a `Parameter` or an initializer, and `bias` also `True` or `False` |
| `Relu` | `ReLU` |
| `Add`, `Sub`, `Mul`, `Div`, `Pow` | `+`, `-`, `*`, `/`, `**` between streams |
| `Neg` | `-x` or `Negative` |
| `Differentiate` | `Gradient(wrt=x)` with respect to an input; with respect to time `DerivativeStep(dt=...)` over a window, `Derivative(dt=...)` one step at a time in a `ModelyLoop` |
| `ForwardEuler`, `RK2`, `RK4` | `Ode(f, states, dt, method=...)` |
| `NeuralODE` | an `Ode` step, the layers of its field created outside `f`, rolled out by a `Loop` |
| `Part` | `Range` |
| `SamplePart`, `TimePart` | `TimeRange` (in samples) |
| `SampleSelect` | `TimeSelect` (`Select` is the same in both) |
| `Integrate(stream, method=...)` | `IntegrateStep(method=..., dt=...)(init, rate)`; 2.0's `Integrate` integrates one step at a time in a `ModelyLoop` |
| `LocalModel(input_function, output_function, pass_indexes=...)` | `LocalModel(input_function, output_function, vectorized=False)(inputs, activations)`: a function is a factory of the cell index, a layer instance is shared by every cell |
| `Fuzzify(output_dimension, range=..., functions=...)` | `Fuzzify(centers, function=...)`: the centers listed explicitly; `normalize=True` makes the memberships add up to one |
| `Optimizer`, `SGD`, `Adam` | `train(optimizer="sgd" / "adam" / a Keras optimizer)` |
| `Parameter(name, dimensions, tw=..., values=..., init=...)`, `Constant(name, values, tw=...)` | `Parameter(name, value=..., tw=...)` or `Parameter(name, dim=..., tw=..., initializer=...)`, `Constant(name, value=..., tw=...)` |
| `ParamFun` | the function written with streams and `Parameter`s, or a custom `Layer` |
| `clearNames` | not needed: names never decide which layers share weights |

`Cosh`, `Sech` and `SampleTime` have no 2.0 counterpart; `Identity` is not
exported (it is `nnodely.core.layer.Identity`).

[2.0.0]: https://github.com/tonegas/nnodely/compare/v1.5.4...v2.0.0
