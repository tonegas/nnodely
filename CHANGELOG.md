# Changelog

All notable changes to nnodely are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [2.0.0] - 2026-10-07

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
- A layer with nothing to configure is applied as it is created: `Sin(x)` is
  `Sin()(x)`, for the math, trigonometric and parameter-free activation layers
  and `TimeConcatenate`.
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
- `Fir` on a multi-dimensional input (1.x took scalars only): every element
  is filtered over its window alone, with a kernel of its own or, with
  `shared_kernel=True`, one for all. `out_features > 1` adds the channels as
  a leading dim axis: `D=(2, 3)` becomes `D=(4, 2, 3)` with 4 channels, and a
  scalar becomes `D=(4,)` as in 1.x.
- `save(path, weights=True)` / `Modely.load(path, weights=True)`, Keras and
  ONNX export, interactive HTML graphs, and Graphviz drawings: one image per
  model and per sub-model, or one flattened image, with a legend.
- Training printers (`"tiny"`, `"legacy"`, `"nnodely"`), `set_seed()` for
  reproducible runs, and `nnodely.__version__`.

### Changed

- Every stream is laid out `(batch, *dim, time, *seq)`, a `Parameter` or a
  `Constant` included: their value is repeated for every sample.
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
| `Relu` | `ReLU` |
| `Add`, `Sub`, `Mul`, `Div`, `Pow` | `+`, `-`, `*`, `/`, `**` between streams |
| `Neg` | `Negative` |
| `Differentiate` | `Differentiate(respect_to=x)` with respect to an input, `Derivative(dt=...)` with respect to time |
| `ForwardEuler`, `RK2`, `RK4` | `Ode(f, states, dt, method=...)` |
| `NeuralODE` | `OdeNet` |
| `Part`, `SamplePart`, `TimePart`, `SampleSelect` | `Range`, `TimeRange`, `TimeSelect`, `Select` |
| `ParamFun` | the function written with streams and `Parameter`s, or a custom `Layer` |
| `clearNames` | not needed: names never decide which layers share weights |

`Cosh`, `Sech`, `SampleTime` and `Identity` have no 2.0 counterpart.

[2.0.0]: https://github.com/tonegas/nnodely/compare/v1.5.4...v2.0.0
