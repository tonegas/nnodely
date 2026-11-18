# Changelog

All notable changes to nnodely are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [2.0.0]

nnodely 2.0 is a rewrite of the library on **Keras 3**. The same model runs on
TensorFlow, PyTorch or JAX, chosen with the `KERAS_BACKEND` environment
variable, and trains, validates and exports through Keras 3.15.1 or later,
which clears five Keras security advisories of earlier versions. It needs
Python 3.11 or later, as Keras does, and is tested on 3.11 to 3.13. Models
saved with 1.x cannot be loaded by 2.0: see
[Migrating from 1.x](#migrating-from-1x) for the 1.x names and their
replacements.

One rule reads everything nnodely returns: an object prints its fields, and
a row of a printed table has the fields of the object it stands for.

### Added

- **`Modely`**, a model declared by its inputs and outputs and built with
  `build()`. A built model is called on data, or on streams to become a block
  of a larger model. Both calls take a dict keyed by Input or input name, or
  a list in the order of the model's inputs. Building again keeps the weights
  the graph holds. `build(keep_weights=False)` gives every weight the model
  computes with its initial value again, in place: frozen ones, those of the
  blocks it calls and of its Loop bodies, the Keras weights of a
  `KerasFunction` and the statistics of a `BatchNorm` included. A build that
  fails leaves the model as it was. A model declares every Input it reads,
  including one a layer starts from (`Integrate(init=x0)`) and one it gives a
  block; one it reads undeclared is refused when the model is declared, and
  the error lists each of them. Only hidden states and Inputs only an
  objective reads are left out: a fed-back Input is declared, or hidden. The
  Keras models a build makes are private.
- **`DataLoader`** for dicts, DataFrames and CSV files or folders, with
  several simulations, sequences and padding masks. `delimiter` and `header`
  are passed to `pandas.read_csv`; n/a values are reported. An array holds
  the features of each sample laid out as the input's `dim`, or flat in
  row-major order; another layout of as many values is refused rather than
  reordered. `windowed=True` takes arrays already laid out as the model's
  tensors, `(N, *seq, tw, *dim)`, and uses them as given: dynamic sequence
  lengths, several per input, are read from their shape, and rollouts of
  different lengths are padded and masked. `DataLoader.get_samples(n)`
  returns `n` consecutive samples of one simulation, ready for a call of the
  model. Its keys are resolved as a call of the model resolves them.
- **`minimize()`** with any Keras loss and a `gain` per objective. A target
  is an Input, a stream computed from the data, a number, or nothing, which
  drives the source to zero. Both sides are streams of the model and have
  one shape, a number aside: a stream of another graph (a block called, the
  body of a Loop, the step `model.loop()` rolls out) or a target of another
  shape is refused when the objective is declared, naming the output to minimize
  or the `seq` to declare.
- **`Modely.minimizers`**: the objectives a model trains, not those of the
  models it calls, printed as a table of their source and target, loss and
  gain. Setting the gain, loss or name of a row changes that objective from
  the next `train()`: `model.minimizers[0]["gain"] = 0.1`.
  `model.minimizers[name].remove()` removes one; `Minimizer`, a row, is in
  the API reference.
- **`train()` returns a `TrainResult`**, read by its attributes: `model`,
  `samples`, `epochs`, `minimizers` (a snapshot) and `losses`, an array per
  epoch under the keys Keras logs: `loss`, `loss_<objective>` and their
  `val_` versions. The model does not keep the result.
- **Early stopping in `train()`**: the name of a monitored loss, one of the
  keys of `losses` (made into a Keras `EarlyStopping` with
  `early_stopping_kwargs`), its configuration, or any callback, Keras's or
  your own.
- **Training printers** `"tiny"`, `"legacy"` (the default) and `"nnodely"`,
  also as the classes `TinyPrinter`, `LegacyPrinter` and `NNodelyPrinter`,
  or any Keras callback. The `"nnodely"` printer charts the total loss along
  the epochs, on a log scale. In a notebook, `"nnodely"` and `"tiny"` redraw
  one output of the cell in place, a few times a second. A loss the logs lack
  is left blank, and the total row is bold.
- **`validate()` returns a `ValidationResult`**, read as a `TrainResult`
  is: `model`, `samples`, `minimizers`, `losses` (`loss`,
  `loss_<objective>`), `signals` and `figures`. Every objective is scored as
  training scores it: the same forward pass, target and padding mask, and
  the loss `train()` logs for validation data. Each `Signal` holds the source
  and target compared, whole, laid out as the stream, and the
  system-identification metrics as functions: RMSE, MAE, max error, bias,
  error std, NRMSE, FIT, R², VAF and correlation. Without arguments a metric
  is a number; with `seq=`, `tw=` or `dim=` it is an array along the axes it
  keeps. The rollout steps padded onto shorter simulations are left out.
  `result.plot()` draws prediction, error, parity, error distribution and
  the error along the rollout, one file per objective,
  `<model>_<index>_<objective>.png`, and draws again elsewhere or with the
  loss curves without validating again. The table prints only with
  `verbose=True`.
- **A layer whose constructor takes only `name` is applied as it is
  created**: `Sin(x)` is `Sin()(x)`. These are `Sin`, `Cos`, `Tan`, `Asin`,
  `Acos`, `Atan`, `Atan2`, `Exp`, `Log`, `Log10`, `Sqrt`, `Abs`, `Floor`,
  `Ceil`, `Deg2Rad`, `Sign`, `Negative`, `Sigmoid`, `Tanh`, `Swish` and
  `Softplus`. Layers with settings (`ReLU`, `GELU`, `Clamp`, `Sum`,
  `TimeConcatenate`, ...) are configured first. The layers show their real
  signature in `help()` and in the documentation.
- **An activation, a math or trigonometric layer, `Clamp` or a membership
  function** called on a list of streams is applied to each and returns a
  list: `Sin([a, b])` gives `[sin(a), sin(b)]`.
- **Stream operators** `+`, `-`, `*`, `/`, `**` and `%` between streams and
  numbers, on either side (`2 ** x`), the modulo taking the sign of the
  divisor as Python's does; `-x`, `abs(x)` and `+x`.
- **Layers new in 2.0**: `BatchNorm`, `LeakyReLU`, `PReLU`, `Swish`, `GELU`,
  `Softplus`, `Asin`, `Acos`, `Atan`, `Exp`, `Log`, `Log10`, `Sqrt`, `Abs`,
  `Floor`, `Ceil`, `Deg2Rad`, `Negative` and `Clamp`.
- **Math layers** `Sinh`, `Asinh`, `Acosh`, `Atanh`, `Log2`, `Square`,
  `Round` and `Rad2Deg`; `Maximum([a, b, ...])` and
  `Minimum([a, b, ...])`, elementwise and broadcast as the operators are, so
  that a bound can be a learned Parameter; `Mean`, `Max`, `Min` and `Norm`
  over the dim axes, as `Sum`, `Norm` with gradient 0 at the zero vector;
  and `Cross(axis=-1)([a, b])`, the cross product of 3-vectors.
- **`Atan2([y, x])`**, the two-argument arctangent: the angle of the point
  `(x, y)` in `[-pi, pi]`, with its quadrant. `y` and `x` broadcast as the
  arithmetic operators do.
- **Window layers** `TimeSelect`, `TimeRange`, `TimeConcatenate` and
  `TimeReverse`, the counterparts of `Select`, `Range` and `Concatenate` on
  the window. They take no axis. `TimeSelect(-1, keep_tw=False)` drops the
  window, so the newest sample combines with every sample of the window.
- **Sequence layers** `SeqSelect`, `SeqRange`, `SeqConcatenate`, `SeqStack`
  and `SeqReverse`, on a sequence axis, fixed or dynamic, counted outermost
  first: `axis=0`, the default, is the rollout of the outermost `Loop`.
  `SeqSelect` removes the axis, and so does `SeqRange` when its range is one
  step long: `SeqSelect(-1)` of a collected trajectory, its last step, can
  be fed back by a `Loop` into an input without that axis. `SeqStack` stacks
  streams of one shape into a new sequence axis.
- **`keep_dim` on `Select`**: it drops the selected dim axis by default, and
  `keep_dim=True` keeps it with length 1. A stream keeps at least one dim
  axis, so selecting from a vector gives a scalar `(1,)`.
- **`Fir` on a multi-dimensional input**: every element is filtered over its
  window alone, with a kernel of its own or, with `shared_kernel=True`, one
  for all. `features > 1` adds the channels as a trailing dim axis, laid out
  as the bias: `D=(2, 3)` becomes `D=(2, 3, 4)` with 4 channels, and a
  scalar becomes `D=(4,)` as in 1.x.
- **`Linear` maps the whole dim of a multi-dimensional input**,
  `(*seq, tw, *dim)` to `(*seq, tw, features)`, or only one dim axis with
  `axis=k`: with a matrix and a bias for each position of the other dim
  axes, as the elements of a `Fir`, or with `shared_kernel=True` one for all.
- **The weights of a layer are `Parameter`s**, predecessors of the layer in
  the graph. `kernel` and `bias` of `Fir` and `Linear` are each a
  `Parameter`, of the shape of `layer.kernel` or `layer.bias`, or a Keras
  initializer, by name or object, from which the layer makes one, a
  `Weight` (a `Parameter` subclass) named `<layer>_kernel` or
  `<layer>_bias`. `bias=False` leaves the bias out. Assigning, training or
  freezing the Parameter changes the layer, and layers given one Parameter
  share it. The slopes of a `PReLU` (`<layer>_alpha`) and the scale and
  offset of a `BatchNorm` (`<layer>_gamma`, `<layer>_beta`) are Parameters
  too; a BatchNorm holds its moving statistics itself. Any `Layer` subclass
  can declare weights held by Parameters (`_weight_labels`,
  `_weight_shapes`), as Keras layers declare theirs with `add_weight`.
  `Parameter(draw=...)` fills a value with draws of a smaller shape, each
  drawn as a tensor of its own, as `Fir` draws the filter of each element.
- **Freezing**: `Parameter.freeze()` / `unfreeze()` (or `trainable`, also a
  constructor argument) keep one Parameter fixed through training or release
  it. `Modely.freeze()` / `unfreeze()` do it for every Parameter a model
  computes with, those of the models it calls and of its Loop bodies
  included. Given a name or a list of names, they act on those alone: a
  Parameter by its name, or a model the graph reaches (a block, a Loop body,
  the model of a LocalModel) for all its Parameters:
  `model.freeze(["k", "g"])`, `model.freeze("submodel")`. A name that
  matches nothing raises, and nothing is frozen. Freezing works before or
  after `build()` and is saved with the model.
- **`Modely.parameters`** lists every Parameter a model computes with, finds
  one by name (`model.parameters["kernel"]`) or index, and prints as a table
  of names, shapes, numbers of values and trainable flags. When the models
  it calls have Parameters, a `model` column groups the rows of each model
  between rules, each Parameter under the first model whose graph reaches
  it. A `total` row counts the values, all and trainable.
- **`Modely.inputs`**: every Input a model reads (those it declares, those
  only an objective reads, its hidden states, inside the models it calls
  too) and every state of the Loops it rolls out. It prints as a table of
  their shapes, their flags (`required` or `optional` for a declared input,
  `target` for training only, `hidden` for the model's own memory) and how
  each is fed: `<- v`, what it reads at every step, and `x₀ = v`, what it
  starts from, with `'name'` for the data of an Input. A state shows both,
  the stream it is fed back with and its initial value. A rollout of `model.loop()`
  prints the rows of the Loop it is a shortcut for, its Inputs with the
  horizon.
- **`Modely.outputs`**: the outputs a model is declared with, in that order,
  printed as a table of their shapes, `-> s`, the Inputs a loop feeds the
  stream of each back into, and the objectives that read it.
- **Tables** (`inputs`, `outputs`, `parameters`, `minimizers`, `models`)
  pick an item by its name or by its index, never both, from the `idx`
  column they print; `model.parameters[2]` picks the same row after save
  and load. `keys()`, `values()` and `items()` return lists, and `in` takes
  a name or an object. `print` draws the table as a box, its first column
  named after what it lists; `repr` gives the same items as
  `Class(\n  [i] 'name': Object(...),\n)`. A missing window prints `-`.
  Tables are coloured when the standard output is a terminal or a notebook,
  plain otherwise; `NO_COLOR` prints them plain and `FORCE_COLOR` colours
  them anyway.
- **Objects print on one line**, `Class(field=value, ...)`, the same with
  `print`, the objects they refer to by name, their attributes giving the
  objects. A declared stream (`Input`, `Output`, `Parameter`, `Constant`)
  prints its class; a stream a call returns (of a layer, an operator, a
  model, a Loop) prints `Stream(name=..., shape=...)`. A layer as an object
  prints the Parameter of each weight (its initializer before the first
  call) and the shape of its last call, `Fir(name=..., kernel=...,
  bias=..., shape=...)`. `Minimizer`, `Loop`, `Integrate`, `LocalModel`,
  `EquationLearner` and the printers print the same way. `print(model)` is a
  table of the Inputs it reads, its outputs and its objectives, with the
  number of Parameters and `(built)`. `print(loader)` is a row per input,
  with its sequence as the loader resolved it, its window, its dim and the
  shape of its array, under the model's name, the number of samples and the
  source.
- **Names and identity.** Nothing keeps two objects from sharing a name: two
  layers named alike, a block and the model calling it, two copies of one
  loaded model. A name the user types (`parameters[...]`, `models[...]`,
  `inputs[...]`, `outputs[...]`, `minimizers[...]`, the sides of `minimize`,
  `freeze` and `unfreeze`, the keys of a call or of a `DataLoader`) is
  resolved to its object where it is given, among the objects the graph
  reaches, the model's own first. A name several objects carry raises
  `ValueError` listing them, each with its index and the model it is in;
  renaming one, `obj.name = "new_name"`, tells them apart. Inside nnodely
  only objects are used, so anything can be renamed and is still found: a
  renamed Input is read by its new name, a renamed output returned under it,
  a Loop body keeps rolling out, a `Gradient` follows its input. A call of a
  model or block keyed by name ignores a name no Input carries, as another
  column of the data, and refuses an Input of another model.
- **Names reach Keras unique.** A clash of names the user chose (Inputs,
  outputs, objectives) raises: at `build()`, at a call, at `save`, and
  before training, validation and export when a name changed since the
  build, which asks for a new `build()`. Names nnodely generates take the
  first free `_2`, `_3`, ...: Keras layers named alike, the kept state of
  `Integrate`, `Derivative` or `Buffer` named like an Input, the copies a
  Loop makes, saved folders and HTML pages. A call of a model is named as
  the model. The user's objects are never renamed.
- **A model used as a block keeps the shapes it was declared with**: each
  stream it is called on must have the shape its graph reads that input
  with, and another one is refused; an `Input` given itself is read through
  the block's own windows. A block reads only the Inputs it declares, each
  given a stream by the call. A block whose Inputs share a name is called
  with a list, by position, or with a dict keyed by the Inputs themselves.
- **A `Parameter` or `Constant` read by several models** (Loop bodies,
  blocks, the outer model) is one object in the saved model too: loaded
  once, listed once, frozen once. So is an Input a model reads and gives a
  block, and a model called in several places.
- **`Input.feedback(stream, *, init=None, hidden=False)`**: feeds `stream`,
  one sample of the Input, back into it at the next step, and returns
  `stream`. The Inputs of a step read each other's previous values, so
  coupled dynamics are written as they are, one Input per state variable. A
  `Loop` or `model.loop()` feeds them back; built as a `Modely`, the same
  graph is the one-step model. A state is declared among the inputs of its
  model (read from data by a `Modely`, started from `Loop(init=)` or from the
  data of the Input in a `model.loop()` rollout, given by the call in a
  block), or
  hidden, `hidden=True`: the model's own memory, as the state of an
  `Integrate`. A hidden state cannot be declared, and starts from `init` (a
  number, a `Parameter` for a trainable initial state, a `Constant` or a
  declared Input), zero without one, in every rollout and every call; a
  `Modely` reads `init` in its place. It keeps the name it is given: named
  like another Input of the model, it is refused.
- **`model.loop(seq=None)`** returns a new `Modely`, the rollout of a model
  declared one step at a time, of the name of the model: its fed-back Inputs
  are all fed back by a single rollout, one `Loop` over the model, its step,
  which stays a one-step model and shares its weights with the rollout.
  Every listed Input
  carries the horizon as its outermost sequence axis (`seq` steps, or as
  many as the data has), and every output is a trajectory. A declared
  fed-back Input starts from the first window of its data, zeros where the
  data leaves it out, and a hidden one from its `init`. A model called as a
  block keeps the feedback of its fed-back Inputs, one per call. The model
  is minimized on its outputs, the trajectories, against targets declared
  with the horizon (`Input("x_meas", seq=-1, tw=1)`); a stream of the step
  is refused, naming the output that collects it. The one-step model it
  rolls out is `rollout.models[0]`: the model itself when it declares every
  state and outputs every stream a state is fed back with, else a model of
  its graph and name with its states among its inputs.
- **`Loop(f=step)(inputs)`** rolls a step out, finding the Inputs its body
  feeds back (with `Input.feedback`, inside the models it calls, or kept by
  `Integrate`, `Derivative` and `Buffer`) as `model.loop()` does, and adding
  an output for an update that is not one. The inputs are given as a dict, a
  list in the order the step declares them, or a stream alone. `feedback=`
  adds feedbacks, for a model that is built or loaded, or to reuse one step
  with different feedbacks. `init=` takes only the fed-back Inputs, the
  inputs of the call only the others, and a state has one initial value. A
  body input its outputs never read, such as the target it is trained on,
  is not an input of the `Loop`. With one output a `Loop` is that output, a
  stream. With several, its outputs are `Output`s of the loop, named
  `<loop name>_<output name>`, unpacked (`x, v = Loop(...)(inputs)`) or
  indexed by position (`loop[1]`) or by the name of the output in the body
  (`loop["v"]`); read as one stream it raises. An output it adds for an
  update is not returned.
- **`Buffer(tw=n, init=None)(signal)`**: the last `n` values of a signal
  computed one sample per step, the current one included: the window a model
  reads of what another computes, such as a controller's commands fed to a
  plant. Its past values are a hidden Input fed back with the signal, and
  `init` holds them before the first step.
- **Calculus blocks.** `Gradient(wrt=x)` differentiates with respect to an
  input, also inside a block called several times in a `Loop` or a
  `model.loop()` rollout, against the state of each call.
  `DerivativeStep(dt=dt)` differentiates with respect to time over a window,
  `Derivative(dt=dt)` with the same stencil one step at a time in a
  `model.loop()` rollout: the backward difference of order `poly_order`, 1 by
  default, `(x[i] - x[i-1]) / dt`, each order more reading one sample more
  and exact on a polynomial of one degree more; the second derivative is one
  applied to the other's result. `IntegrateStep` integrates over a window and
  `Integrate` one step at a time in a `model.loop()` rollout, with
  `method="euler"` or `"trapezoidal"`. `Ode(f, states, dt, method=)`
  integrates a vector field with Euler, midpoint, Heun or RK4. `IntegrateStep`,
  `Integrate`, `DerivativeStep` and `Derivative` take the time step `dt`, 1.0
  by default.
  `Integrate` and `Derivative` keep their state (the integral,
  `<name>_<rate>`, with `<name>_<rate>_rate` for the trapezoid, or the
  samples before the step, `<name>_<signal>`) in hidden Inputs fed back by
  the `model.loop()` rollout or `Loop` with its other fed-back Inputs, so
  they need no `Loop` of their own; their result is named after the block.
  `IntegrateStep` and `DerivativeStep` take their initial condition, a stream
  or a number,
  as the first argument of the call, `IntegrateStep(dt=dt)(init, rate)`, and
  start from zero when called on the signal alone. `Integrate`, `Derivative`
  and `Buffer` take it at declaration, `init=`, each checked by one rule:
  one sample, or the samples read before the first, a scalar dim or the
  signal's, no sequence or the signal's. The trapezoidal rule averages the
  first rate with the rate before it, `init_rate`, zero without one:
  `IntegrateStep(method="trapezoidal", dt=dt)(init, init_rate, rate)`, or
  `Integrate(..., init_rate=)`.
- **`KerasFunction(function)`**: any Keras layer, class or instance, as an
  nnodely layer: a custom function in the graph, its streams passed to
  `call` as separate arguments. The model saves, loads and exports with it.
- **Membership functions as elementwise layers of their own**, 1 at 0:
  `Triangular(radius)`, `Rectangular(radius)`, 1 on `[-left, right)`, and
  `Gaussian(sigma)`. The signal is shifted to center them,
  `Triangular(radius=0.5)([x - c])`. A `radius` is a number, a
  `(left, right)` pair or a `Parameter` of dim `(1,)` or `(2,)`, and `sigma`
  a number or a `Parameter` of dim `(1,)`: a Parameter is trained with the
  model. A width trained to zero or below is held at 1e-7, so the set
  collapses rather than giving NaN. A step has no gradient: a Parameter read
  by rectangles alone does not train.
- **`Fuzzify(centers, function=...)`** is one layer, whose Keras layer
  computes every membership at once. `function` has no default and takes a
  membership class placed on each center, a membership name (`"Triangular"`,
  ...), any other layer class or instance applied to `x - center`
  (`function=Sin` gives `sin(x - c)`), or a list of one function per center.
  A custom function is a Keras layer evaluated on every center and saved and
  exported with the model; its `call` reads one argument, `x - c`. The
  Parameters the functions read are predecessors of the Fuzzify.
- **Trainable `Fuzzify` centers**: besides numbers, `centers` is one stream
  of shape `(n,)`, such as a `Parameter` vector trained with the model or a
  `Fir` with `features=n`, which gives every sample its own centers, or a
  list of scalar streams and numbers. The memberships are placed on them and
  sized by their neighbors as they are computed, so the sets move with them,
  widths included, and the gradient reaches them through both. They are used
  in the order given, ascending. A matrix, a time window or a list item that
  is not a scalar is refused.
- **`Fuzzify(..., normalize=True)`** divides the memberships of each sample
  by their sum, so that they add up to one (all zeros stay zeros), for
  memberships that are not negative. `normalize=False`, the default, keeps
  the values of the functions.
- **`Fuzzify(..., bounds=...)`** holds the signal within bounds before the
  functions read it: `True` between the lowest and the highest center
  (moving centers as they are computed), so a value beyond the centers is in
  the outer set instead of in none; `(left, right)` between two numbers;
  `False`, the default, not at all.
- **`Fuzzify.plot()`** draws the memberships with matplotlib as the layer
  computes them now, its Parameters at their current values.
- **`LocalModel(input_function, output_function, vectorized=False)(inputs,
  activations)`**: several activations are multiplied into joint
  memberships. A function is a factory of the cell index, giving each cell a
  layer of its own; a layer instance is shared by every cell;
  `vectorized=True` calls it once on all the cells, which it returns on a
  last dim axis, as `Fir(features=N)` does. Each call is a block, its cells
  a model of their own, which `export_html` shows on a page of its own.
- **`EquationLearner(functions, linear_in=, linear_out=, threshold=)`**: an
  input projection gives each function its arguments, and `linear_out`
  combines the results. Functions are names (the trigonometric, math and
  activation layers, `gaussian`, `triangular`, `rectangular`, and `add`,
  `subtract`, `multiply`, `divide`, `power`, `atan2`, `maximum`, `minimum`,
  `mod` of two arguments), layer classes or instances, or callables on
  streams. `threshold` drops the coefficients of both projections below it
  from what the model computes, while they keep training;
  `learner.threshold` can be changed on a built model, which is saved with
  it. `learner.plot()` draws the learned equation as LaTeX, every stream a
  variable of its own, and returns it as a sympy `Eq`.
- **`save(path, weights=True)` / `Modely.load(path, weights=True)`**, in
  cascade: the path is a folder with the model's graph, `model.json`, and
  weights, `model.weights.h5`, and every model it calls as a block or rolls
  out in a `Loop`, saved the same way in a folder of its own, nested as the
  models are. The loaded model calls the same models (`models` lists them,
  `freeze` reaches them by name, one called several times is one model
  again), keeps the feedback of the fed-back Inputs they declare, and each
  folder loads on its own. `save` builds a model not built yet to write its
  weights, and refuses Inputs that share a name. A block keeps weights of
  its own in its folder only if it was built on its own, as a Loop body
  always is; `weights="all"` saves every block with its own weights, so
  each folder is a trained model. Node classes are saved with their module,
  so a class of your own named like one of nnodely's never takes its place
  on load.
- **`export_keras` and `Modely.import_keras`**: a stand-alone Keras copy of
  every model, a weight read from several Keras graphs (a body rolled out by
  two Loops, a Parameter read inside and outside a body, several `Gradient`
  of one relation) written once per graph with the same value.
  **`export_onnx` and `Modely.validate_onnx`** on every backend, through
  tf2onnx, the PyTorch exporter or jax2onnx. Every axis
  of the file is fixed: the batch is `batch_size` samples, one by default,
  and a sequence of dynamic length, such as a rollout of `model.loop()`, is
  exported `seq` steps long; without it the export raises. Every block
  exports, `Cross` and `Log2` included, but a `Gradient`
  with respect to an Input: it exports from JAX, from TensorFlow when tf2onnx
  converts the gradient of every layer it differentiates, and not from
  PyTorch. A file with an operator ONNX does not define is removed, and the
  error names the operators. Both exports refuse a model renamed since its
  build.
- **`export_html`**, an interactive HTML graph of the model. Every node is
  coloured by its role, with a legend that highlights a role, and labelled
  with its name; two nodes of one name are both drawn. A panel explains the
  node selected (its shape axis by axis, what it reads and feeds), with
  search; the flattened page draws each block's body in a region of its own,
  and a block has a page of its own named after it. The layout switches
  between left to right and force at any size.
- **`set_seed()` and `get_seed()`** for reproducible runs, and
  `nnodely.__version__`.
- **A splash at import**, one band of 80 columns by 6 lines: a neural network
  of 3, 5 and 2 neurons feeds the nn of the wordmark, which drives a mass on
  a spring and a damper, with the versions of nnodely and Keras and the
  backend under the wordmark. It shows only in a terminal or a notebook;
  `NNODELY_SPLASH=0` turns it off. It replaces the banner 1.x printed.
- **Tutorials**: 13 Jupyter notebooks in `docs/tutorial`, shown in the
  documentation with their outputs: model design and the shape convention,
  data management, training, validation and the HTML graph, save, load and
  export, `Parameter` and `Constant`, partitioning, `Fir` and `Linear`,
  `Fuzzify` and `LocalModel`, recurrent models (`model.loop()`, `Loop`,
  `Buffer`), calculus, custom functions and layers (`KerasFunction`, `Layer`
  subclasses) and the `EquationLearner`.
- **Documentation for the release**: every public name has a reference page
  of its own, in the numpydoc style, with its calling forms, every argument
  with its type and default, the shape it returns, the errors it raises, and
  examples from the basic use to the edge cases. The guide pages each follow
  one example through numbered steps, with the output of every step and a
  table of the common errors, and link their tutorials. A glossary defines
  the terms. A test runs every example of the reference pages, every guide
  page and the README, and checks what they print.

- **Options with a fixed set of names are `StrEnum`s**, each in the module
  that uses it and exported by `nnodely`: `IntegrationMethod`
  (`IntegrateStep`, `Integrate`), `OdeMethod` (`Ode`), `InterpolationMode`
  (`Interpolation`), `MembershipName` (`Fuzzify`), `EquationFunction`
  (`EquationLearner`), `PrinterName` (`train(printer=)`) and `OnShort`
  (`DataLoader(on_short=)`). A member is its string:
  `method=IntegrationMethod.TRAPEZOIDAL` and `method="trapezoidal"` are the
  same, and a model saves the string. An unknown name raises where it is
  given, `on_short` included. `seq` keeps `-1` and `"dyn"`.

### Changed

- **The backend is chosen at the import**: `KERAS_BACKEND`, else the
  `backend` of `~/.keras/keras.json`, else TensorFlow. Without
  `KERAS_BACKEND`, a backend that is not installed gives way to the first
  installed of TensorFlow, PyTorch and JAX, so `nnodely[jax]` alone runs on
  JAX. A `KERAS_BACKEND` that is not installed fails the import, naming it,
  how to install it and the backends installed. `nnodely.get_backend()`
  returns the backend in use.
- **nnodely computes on the CPU by default, on every backend**, where its
  small models run fastest. `NNODELY_DEVICE=gpu`, set before the import, takes the accelerator of the
  backend, and the import fails, saying how to have one, when there is none.
  `nnodely.get_device()` returns the device in use.
- **The package installs no backend**: install `nnodely[tensorflow]`,
  `nnodely[torch]` or `nnodely[jax]`, which brings the converter Keras
  exports ONNX with from it (tf2onnx, onnxscript, jax2onnx); `onnx` and
  `onnxruntime` are dependencies of nnodely itself.
- **Every stream is laid out `(batch, *seq, tw, *dim)`**, a `Parameter` or a
  `Constant` included: their value is repeated for every sample. The
  sequence axes come first, outermost first, where a `Loop` rolls out. Then
  the time window, `tw`, where `Fir`, derivatives, integrals and the time
  selections act, or no window at all: `(batch, *seq, *dim)`. Then the dims.
  `Shape` holds `seq` (a tuple, possibly empty), `tw` (an int, or `None`
  without a window) and `dim`, and prints as `Shape(S=(5,), T=3, D=(4,))`;
  `shape.dimensions` is `(seq, tw, dim)`. The arrays given to a model or
  built by `DataLoader` are laid out the same, `(samples, *seq, tw, *dim)`.
  `tw` is the one name of the window everywhere, and it counts samples, not
  seconds.
- **An `Input` is declared once**, with the shape of the data it reads, and
  does not change: `Input("x", dim=4, seq=5, tw=3)` is
  `Shape(S=(5,), T=3, D=(4,))`. `tw=n` is the last `n` samples up to and
  including the current one; `tw=(past, future)` adds future samples,
  `T = past + future`. Without `tw` the Input has no window: one value per
  sample, `(samples, *dim)` in the data. A sequence, `seq=5` or
  `seq=(4, -1)` (outermost first, `-1` or `"dyn"` dynamic), needs a window.
  `dim` is one feature by default, each axis at least 1: `dim=0` or
  `dim=()` is refused. There is no `tw()`, `sw()`, `z()`, `next()` or
  `last()` on an Input: the layers that read it take what they need of its
  window, `TimeSelect(-1)(x)` its newest sample, `TimeRange(0, 3)(x)` its
  first three.
- **Arithmetic between streams follows one rule for every stream**, an
  Input, a computed stream, a number, a `Parameter` or a `Constant`: an axis
  a stream lacks broadcasts, an axis it has must match. Windows must be
  equal, or one of them absent, which applies to every sample of the other's
  window; a single sample, `T=1`, is a window like any other and does not
  combine with a longer one. Sequences must be equal, or one of them
  absent. Dims broadcast as Keras broadcasts them, from the last axis, each
  axis equal or 1. One stream must already have the shape of the result.
  `Concatenate` repeats a single sample along the sequence and window of the
  others. The layers that act on the window (`Fir`, the `Time*` layers,
  `DerivativeStep`, `IntegrateStep`) refuse a stream without one.
- **A `Parameter` or `Constant` declares its shape as an `Input` does**, with
  `seq`, `tw` and `dim` as sizes: `dim` one feature by default, each axis at
  least 1, and a sequence of fixed lengths that needs a window. A `value`
  fills that shape. A list of numbers holds exactly as many, read in the
  order `(*seq, tw, *dim)` (`value=[1, ..., 10], tw=2, dim=5`); an array of
  several axes has the declared sizes, in any order, its axes moved into
  place (a `(2, 3, 4, 5)` array with `seq=5, tw=3, dim=(2, 4)` is
  `S=(5,), T=3, D=(2, 4)`). Anything else raises `ValueError`. Without
  `dim`, `seq` and `tw` take the axes of the value of their sizes, and the
  axes they leave are the dim, in their order (a `(2, 4, 6)` value with
  `tw=4` is `T=4, D=(2, 6)`); with neither, every axis is dim and the value
  has no window: `value=[1, 2, 3]` is `D=(3,)`. A `Constant` requires a
  value; a `Parameter` without one is drawn by its `initializer`. `.value`
  and the variable are laid out `(*seq, tw, *dim)`, without the window when
  no `tw` is given.
- **A layer is identified by the Python object it comes from**: applying one
  layer object several times shares its weights, and layers created apart
  never share them, whatever their names. `clearNames` is not needed.
- **The fed-back stream is named `feedback` everywhere**:
  `Input.feedback(stream)` and `Loop(feedback=)` replace `closedLoop`,
  `connect`, `ClosedLoop`, `Connect` and `prediction_samples`.
- **Each objective logs its loss as `loss_<objective>`** in the training
  history, beside the totals `loss` and `val_loss`, so no name an objective
  takes can collide with them; each batch weighs as many samples as it
  holds, as Keras weighs `loss`.
- **`BatchNorm` normalizes along the first dim axis by default**; `Softmax`,
  `PReLU` and `BatchNorm` take an explicit `axis` as a tensor axis, the
  batch as 0, as Keras does: a negative one counts from the last dim axis.
- **Attributes that are not part of the API start with an underscore**, as
  in Python: a model's order, a node's predecessors, the settings a layer
  keeps, the Keras models a build makes.
- **A `Modely` keeps lists of its own**: the `inputs` and `outputs` lists it
  is declared with are copied, so changing them afterwards leaves the model
  as it was declared.
- **A model's graph is sorted and inlined without recursion**: declaring a
  model, `flatten()` and `build()` take a graph of any depth. Keras still
  builds its functional models recursively: a graph about a thousand layers
  deep, such as a simulation unrolled in Python, is rolled out with a `Loop`
  or `model.loop()`, or built after raising `sys.setrecursionlimit`.
- **The save format is a folder**, `model.json` and `model.weights.h5`, at
  format version 18. Saves and exports made with 1.x do not load.

### Removed

- **The 1.x API that has a 2.0 replacement**: see
  [Migrating from 1.x](#migrating-from-1x).
- **`Sech`, `SampleTime` and `ParamFun`**, without a direct counterpart:
  `1 / Cosh(x)`, the `dt=` of the blocks that need a time step, and a
  function written with streams and `Parameter`s or a custom `Layer`.
- **The visualizers** (`TextVisualizer`, `MPLVisualizer`,
  `MPLNotebookVisualizer`), **`StandardExporter`**, **`logger`** and the
  initializers `init_negexp`, `init_lin`, `init_constant`, `init_exp`.
  Training prints through a printer, validation draws its own figures, the
  model saves and exports itself, and a `Parameter` takes a Keras
  initializer or a `value`.
- **`Identity`** is no longer exported: it is `nnodely.core.layer.Identity`.

## Migrating from 1.x

| nnodely 1.x | nnodely 2.0 |
|---|---|
| `model = Modely()` or `nnodely()`, then `model.addModel("name", outputs)` | `model = Modely("name", inputs=[...], outputs=[...])` |
| `model.removeModel(name)` | declare a new `Modely` with the outputs you keep |
| `model.neuralizeModel(sample_time)` | `model.build()`; blocks that need a time step take it (`dt=`) |
| `model(inputs)` | `model(inputs)` on a built model, a dict keyed by Input or input name, or a list |
| `model.addMinimize(name, a, b, loss_function=...)` | `model.minimize(name, source, target, loss=..., gain=...)`: the prediction first |
| `model.removeMinimize(name)` | `model.minimizers[name].remove()` |
| `model.loadData(name, source)` | `data = DataLoader(model, source=...)` |
| `model.getSamples(...)` | `data.get_samples(n)` |
| `model.filterData(...)` | filter the arrays before giving them to `DataLoader` |
| `model.trainModel(...)` | `model.train(data, val_data=..., epochs=..., batch_size=..., lr=...)`, returning a `TrainResult` |
| `model.analyzeModel(...)`, performance reports | `model.validate(data)`, returning a `ValidationResult` |
| `model.trainAndAnalyze(...)` | `model.train(...)` then `model.validate(...)` |
| `x.closedLoop(...)`, `x.connect(...)`, `ClosedLoop`, `Connect`, `addClosedLoop`, `addConnect`, `prediction_samples=` | `x.feedback(stream)` in a `model.loop()` rollout or a `Loop` |
| `model.resetStates(...)` | the initial value of a fed-back Input: `Loop(init=)`, the data of the Input in a `model.loop()` rollout, or the `init` of a hidden state, `x.feedback(stream, init=x0, hidden=True)` |
| `x.tw(seconds)`, `x.sw(samples)`, `x.z(...)` | `Input("x", tw=samples)` or `tw=(past, future)`, declared once, in samples; `TimeSelect` / `TimeRange` take parts of the window |
| `model.resetSeed(seed)` | `set_seed(seed)` |
| `saveModel` / `loadModel` | `model.save(path)` / `Modely.load(path)` |
| `exportPythonModel` / `importPythonModel`, `saveTorchModel` / `loadTorchModel` | `model.export_keras(path)` / `Modely.import_keras(file)` |
| `exportONNX` / `onnxInference` | `model.export_onnx(path)` / `Modely.validate_onnx(file, inputs)` |
| `exportReport`, the visualizers | `model.validate(data, out_dir=...)` figures, a training `printer`, `model.export_html(path)` |
| `StandardExporter`, `getWorkspace()` | the methods of the model, with the path given to each |
| `Fir(output_dimension, W=..., b=..., W_init=..., b_init=...)` | `Fir(features, kernel=..., bias=...)`: each a `Parameter` or an initializer, and `bias` also `True` or `False` |
| `Linear(output_dimension, W=..., b=..., W_init=..., b_init=...)` | `Linear(features, kernel=..., bias=...)`, as `Fir` |
| `init_negexp`, `init_lin`, `init_constant`, `init_exp` | a Keras initializer, or `Parameter(value=...)` |
| `Relu` | `ReLU` |
| `Add`, `Sub`, `Mul`, `Div`, `Pow` | `+`, `-`, `*`, `/`, `**` between streams |
| `Neg` | `-x` or `Negative` |
| `Sech` | `1 / Cosh(x)` |
| `Differentiate` | `Gradient(wrt=x)` with respect to an input; with respect to time `DerivativeStep(dt=...)` over a window, `Derivative(dt=...)` one step at a time in a `model.loop()` rollout |
| `Integrate(stream, method=...)` | `IntegrateStep(method=..., dt=...)(init, rate)`; 2.0's `Integrate` integrates one step at a time in a `model.loop()` rollout |
| `ForwardEuler`, `RK2`, `RK4` | `Ode(f, states, dt, method=...)` |
| `NeuralODE` | an `Ode` step, the layers of its field created outside `f`, rolled out by a `Loop` |
| `Part` | `Range` |
| `SamplePart`, `TimePart` | `TimeRange` (in samples) |
| `SampleSelect` | `TimeSelect` (`Select` is the same in both) |
| `LocalModel(input_function, output_function, pass_indexes=...)` | `LocalModel(input_function, output_function, vectorized=False)(inputs, activations)`: a function is a factory of the cell index, a layer instance is shared by every cell |
| `Fuzzify(output_dimension, range=..., functions=...)` | `Fuzzify(centers, function=...)`: the centers listed explicitly; `normalize=True` makes the memberships add up to one |
| `Optimizer`, `SGD`, `Adam` | `train(optimizer="sgd" / "adam" / a Keras optimizer)` |
| `Parameter(name, dimensions, tw=..., values=..., init=...)`, `Constant(name, values, tw=...)` | `Parameter(name, value=..., tw=...)` or `Parameter(name, dim=..., tw=..., initializer=...)`, `Constant(name, value=..., tw=...)` |
| `ParamFun` | the function written with streams and `Parameter`s, or a custom `Layer` |
| `SampleTime` | the `dt=` of the blocks that need a time step |
| `clearNames` | not needed: names never decide which layers share weights |
| `Identity` | `nnodely.core.layer.Identity` |

[2.0.0]: https://github.com/tonegas/nnodely/compare/v1.5.4...v2.0.0
