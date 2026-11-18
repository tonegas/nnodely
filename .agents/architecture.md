# Architecture

How the source in `src/nnodely/` fits together. Read this before changing
library code.

## The two graphs

A model exists twice:

1. **The nnodely graph**, built in plain Python as the user writes it. Every
   node is a `Node` (`core/stream.py`) with its predecessors `_preds`. Inputs,
   Parameters, Constants and the results of layers are `Stream`s, nodes with
   a `Shape` (`core/shape.py`); a call of a block is a `ModelCall`, a plain
   `Node`. Nothing touches Keras yet.
2. **The Keras graph**, made by `Modely.build()` (`core/modely.py`): the
   nnodely graph is walked in topological order (`core/dag.py`) and every
   layer node is turned into a Keras layer applied to Keras tensors.

Most bugs come from the gap between the two: a shape declared in the nnodely
graph that the Keras layer does not produce, or a name valid in one and not
in the other.

## Shapes

`Shape(S, T, D)`: `S` the sequence axes (a tuple, possibly empty), `T` the
window (`None` when the stream has none), `D` the feature axes (a tuple, at
least `(1,)`). A tensor is laid out `(batch, *S, T, *D)`, the `T` axis absent
when `T` is `None`. `core/shape.py` holds the helpers (`BATCH_AXIS`,
`transpose`, `Shape.from_tensor_shape`, `Shape.replace`).

Two streams combine (`+ - * / **`, `Shape.broadcast`) axis by axis; an axis
a stream lacks broadcasts, an axis it has must match:

- **Window**: equal, or absent in one. `T=1` is a window: it does not
  combine with `T=3`.
- **Sequence**: equal, or absent in one; a dynamic axis matches any length.
- **Dim**: as Keras broadcasts, from the last axis, each axis equal or 1.
- **One operand already has the result's shape**: no two-way broadcast.

## Layers

A layer is **two classes**, see `layers/fir.py` for a complete example:

- **The nnodely layer**, a subclass of `Layer` (`core/layer.py`). It is a node
  of the nnodely graph. It is created with its settings, called on streams
  (`Fir(features=2)(x)`), and computes the output shape with
  `output_shape(*inputs)`. Its `build_layer()` returns the Keras layer.
  Calling it returns a **new node** sharing `_source` with the layer object,
  so every call of one layer object shares its weights.
- **The Keras layer**, `<Name>Impl(keras.layers.Layer)`, decorated with
  `@keras.saving.register_keras_serializable(package="nnodely")`. It does the
  computation in `call()` with `keras.ops`, and implements `get_config()`
  with every constructor argument, so that `export_keras` round-trips.

Elementwise operations of `keras.ops` (`Sin`, `Exp`, `Atan2`) subclass
`UnaryOp` / `BinaryOp` in `layers/arithmetic.py` and only set `_operation`.
Any layer whose constructor takes only `name` (`Sin`, `Sigmoid`) can be
applied as created, `Sin(x)`, through the metaclass `_LayerType`.

Layers with weights declare `_weight_labels` and `_weight_defaults`, and
implement `_weight_shapes(x)`. The weights are `Parameter` nodes
(`layers/parameter.py`) that become extra predecessors: the Keras layer
receives `[x, *weights]` and owns no weight itself.

## Models

- `Modely` (`core/modely.py`): inputs, outputs, objectives (`minimize`),
  `build`, `train`, `validate`, `save` / `load`, exports. The Keras model
  that trains the objectives is in `core/minimizer.py`. A `Modely` called on streams is a block of another model.
- `Loop` (`layers/loop.py`) and `Modely.loop`: multi-step rollouts and
  recurrent models.
- `DataLoader` (`core/dataloader.py`): turns arrays or files into the
  windows and sequences a built model reads.
- Results and printed tables: `core/results.py`, `utils/printers.py`. Every
  printed object follows one rule: a row of a table has the fields of the
  object it stands for.

## Names and identity

Names are how the user reads a graph. Objects are how the library tells
streams apart. Two rules keep the two from mixing; every change that reads,
compares or assigns a name must respect them.

### Rule 1: resolve user strings at the entry

A string the user types is accepted, but it is resolved to its object **where
the user gives it**. A name several objects carry is refused there, with the
listing `_pick` prints. After that, inside the library, only objects are used:

- never a name to find an object again;
- never an object turned back into its name to compare it.

All resolution goes through `_resolve(key, candidates, ...)` in `core/dag.py`,
which raises `NameNotFound` (a `KeyError` and a `ValueError`). The entry
points that already use it:

- the tables `inputs`, `outputs`, `parameters`, `models` and `minimizers`,
  which all inherit `Table` (`core/dag.py`) and share its `__getitem__`, by
  name or by index;
- a call of a model, and the DataLoader keys and `format`
  (`Modely._input_of`, `Modely._given_by_input`);
- `freeze` / `unfreeze`, the sides of `minimize`;
- the keys, `init` and feedback of a Loop.

A new public method that takes a name or an object follows the same pattern:
resolve once with `_resolve`, then work with the object. A new collection
shown to the user is a `Table`.

### Rule 2: check uniqueness at the Keras boundary

Keras needs names: it takes and returns dicts keyed by input and output
names, logs losses under string keys, and refuses two layers of one name.
Wherever nnodely hands names to Keras (build, call, train, validate, save,
export), it checks that they are unique. It never renames the user's objects,
not even for a moment, since that would hide the clash.

The two kinds of name are treated differently:

- **Names the user chooses** (Inputs, outputs, minimizers): a clash raises,
  and the message tells the user which object to rename.
- **Names nnodely generates** (Keras layers, the kept state of `Integrate`,
  `Derivative` and `Buffer`, copies made inside a Loop, saved folders, HTML
  sub-pages): they take a free name with `_free_name(name, taken)` in
  `core/dag.py`, which gives `name`, then `name_2`, `name_3`, ... Do not write
  another numbering loop. `_next_name` names a node the user gave no name.

The checks in place, to keep and to reuse:

| Check | Where |
| --- | --- |
| Two Inputs of one name refused | `Modely.build`, and `save` via `_check_inputs_apart` |
| Two outputs of one name refused | build and call |
| Names changed since the build refused | `_check_names_built`, before train, validate, `export_keras`, `export_onnx` |
| Data made for another build refused | `Modely._training_arrays` |
| A renamed Input reaches Keras at the next build | `_refresh_keras_inputs` (`core/modely.py`) |
| Keras layers named alike | `_claim_keras_name` (`core/layer.py`) |
| Kept states yield to user Inputs; a hidden state the user named, `feedback(hidden=True)`, raises | `_rename_hidden_states` (`core/modely.py`) |

The Keras key of a node is `Modely._keys[id(node)]`, written by the build.
Read it directly: a node the build never saw is a `KeyError`, never a guess
from its current name.

### Identity keys

A dict or set that tells objects apart is keyed by `id(obj)`, not by the
object and never by its name (`Modely._keys`, `_Step.key()` in
`layers/loop.py`). `Stream` overloads the arithmetic operators, and one day
might overload `==`; `id()` stays correct whatever happens. An `id` is
unique only while its object lives: key only objects the graph holds.

Inside a Loop, a node and its copies are one Input: compare them with
`_Step.key(node)`, which keys the Input a node stands for. Never compare two
Inputs by name to decide they read the same data.

### What stays a string

These are boundaries, protected by rule 2. They stay strings on purpose:

- Input and output names as keys of data and results; CSV columns.
- Keras input, output and layer names; ONNX I/O names.
- Keras log keys: a minimizer logs `loss_<name>` (`val_loss_<name>` on
  validation data), so no user name can collide with the totals `loss` and
  `val_loss`. The `early_stopping` monitor is one of these keys.
- File and folder names.
- Class and operation registries for serialization, and a user's Keras layer
  class known by its registered name (`KerasFunction`).
- Option values. A closed set of names is a `StrEnum` of the module that
  uses it (`IntegrationMethod` in `layers/calculus.py`, `OdeMethod`,
  `InterpolationMode`, `MembershipName`, `EquationFunction`, `OnShort`),
  exported by `nnodely`. Its members are the strings, so a caller passes
  either, and configs, messages and tables hold `member.value`. Open sets stay plain strings: initializer, optimizer and
  loss names (Keras's), the `keras.ops` names of the arithmetic Impls. So
  does a word among numbers: `seq="dyn"`, `seq_length="full"`,
  `weights="all"`.
- The fields of a `TrainResult` or `ValidationResult` (`core/results.py`):
  a result is a snapshot, so `model` and the `minimizers` and `losses`
  dicts hold the names as they were when it was made. It holds no stream or
  model, which would keep the graph alive and mix two moments.
- Every error message and printed table. `NameNotFound.args[0]` is the
  message, as for every exception of the library.

### Patterns that follow from the rules

- Every node that holds a model keeps it in `_model` (declared on `Node`,
  None otherwise): a `ModelCall`, a `Loop` body, `LocalModel`,
  `EquationLearner`. Read `node._model`, never probe attributes by name.
- A layer that must follow the copy of a node when a block is inlined
  overrides `Layer._copied(memo)` (see `Gradient._copied`): `core/dag.py`
  stays unaware of specific layers.
- A call of a model is named as the model (`blk`, not `blk_call`).
- An action on an item of a table is a method of its row, picked first:
  `model.minimizers["fit"].remove()`, `model.parameters[2].freeze()`. A
  removed row refuses later settings.
- An option string is resolved to what it names once, where it enters, so
  an unknown one fails there, not deep in a call: a closed set by its
  `StrEnum`, `IntegrationMethod(method)`, with the message listing the
  values; the Impls of `layers/arithmetic.py` resolve their operation in
  `__init__` with `_operation`.
- A key of a model call or of the DataLoader that is an Input of another
  model is refused. A string key that names no Input is ignored.
- A file nnodely writes for each of several objects is named by position, or
  by `_free_name`, never by a slug of a user name alone: `Fit` and `fit`
  collide on macOS and Windows.

## Saving and exporting

- `save` / `load` (`core/registry.py`, `ModelSerializer`) write the nnodely
  graph as JSON plus weights. Each node is saved by its class path
  (`_NODE_REGISTRY` in `core/stream.py`) and rebuilt with
  `get_config()` / `from_config()`. **A change to what is saved bumps
  `ModelSerializer.VERSION`.**
- `export_keras`, `export_onnx` (`core/export.py`) export the Keras graph.
  ONNX is unavailable from the jax backend.
- `export_html` (`utils/html_export.py`, templates in `utils/templates/`)
  draws the graph as a web page.

## Where things live

| Path | Content |
| --- | --- |
| `core/` | graph, shapes, model, data, training, saving |
| `layers/` | every public layer, one family per file |
| `utils/` | printers, plots, HTML export, seeding, splash |
| `__init__.py` | the public API: imports and `__all__` |
