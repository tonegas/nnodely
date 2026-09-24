Datasets
========

A :class:`~nnodely.DataLoader` turns recorded signals into the samples a model
is trained and validated on. It reads the model's inputs, their windows and
their sequence lengths, so the model has to be **built** before the loader is
created.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from nnodely import DataLoader, Fir, Input, Modely, Output

   x = Input("x")
   u = Input("u", dim=2)
   y = Output("y", Fir(out_features=1)([x.sw(3)]) + Fir(out_features=1)([u.last()]))
   model = Modely("example", inputs=[x, u], outputs=[y])
   model.minimize("fit", y, Input("y_target").last())
   model.build()

Sources
-------

A **dictionary** maps every input name to one array per signal. The first axis
is time, the remaining ones are the input's features:

.. code-block:: python

   n = 200
   signals = {
       "x": np.sin(np.linspace(0, 10, n)),
       "u": np.random.rand(n, 2),
       "y_target": np.cos(np.linspace(0, 10, n)),
   }
   loader = DataLoader(model, source=signals)
   print(loader)

A **DataFrame**, a **CSV file**, or a **folder** of CSV files can be used as
well. ``format`` then maps each input name to its columns: a column name, a
positional index, or a list of columns for an input with several features.
Columns that no input reads are ignored:

.. code-block:: python

   frame = pd.DataFrame({
       "time": np.arange(n) * 0.01,
       "pos": signals["x"],
       "u1": signals["u"][:, 0],
       "u2": signals["u"][:, 1],
       "out": signals["y_target"],
   })
   frame.to_csv("run_0.csv", index=False)

   format = {"x": "pos", "u": ["u1", "u2"], "y_target": 4}
   from_frame = DataLoader(model, source=frame, format=format)
   from_file = DataLoader(model, source="run_0.csv", format=format)
   from_folder = DataLoader(model, source=".", format=format, csv_glob="run_*.csv")

``delimiter`` and ``header`` are passed to the CSV reader.

Simulations
-----------

A **list** of dictionaries or DataFrames, or a folder with several files, is a
set of *simulations*: independent recordings that may have different lengths.
A window never spans two simulations:

.. code-block:: python

   first = {name: values[:120] for name, values in signals.items()}
   second = {name: values[120:] for name, values in signals.items()}
   loader = DataLoader(model, source=[first, second])
   print(len(loader))   # 2 fewer samples than one simulation of the same length

Samples
-------

Every sample is a dictionary with one array per input, shaped
``(*dim, time, *seq)``. The loader can be indexed, iterated, and converted to
the full arrays:

.. code-block:: python

   sample = loader[0]
   print({name: value.shape for name, value in sample.items()})
   arrays = loader.as_dict()   # {name: (samples, *dim, time, *seq)}

``step`` sets how many samples separate two consecutive windows:

.. code-block:: python

   sparse = DataLoader(model, source=signals, step=10)

Sequences
---------

An input declared with ``seq`` carries a trajectory of windows. The loader
builds one sliding window per sequence length, on top of the time windows.
Sequences are used by rollouts (see :doc:`recurrent`):

.. code-block:: python

   s = Input("s", seq=20)
   seq_model = Modely("sequence", inputs=[s], outputs=[Output("s_out", s.last() * 2.0)])
   seq_model.build()

   seq_loader = DataLoader(seq_model, source={"s": np.arange(100.0)})
   print(seq_loader[0]["s"].shape)   # (1, 1, 20)

A sequence length can be left dynamic with ``seq=-1``. It is then fixed when
the data is loaded:

- ``seq_length=n`` uses sequences of ``n`` steps.
- ``seq_length="full"`` makes every simulation one sample spanning all of it.
  Simulations of different lengths are padded, and ``loader.mask`` marks the
  real steps. Training ignores the padded steps.

Dynamic sequences are meant to be consumed by a :class:`~nnodely.Loop`, which
rolls out over whatever length it is given (see :doc:`recurrent`).

.. code-block:: python

   d = Input("d", seq=-1)
   dyn_model = Modely("dynamic", inputs=[d], outputs=[Output("d_out", d.last())])
   dyn_model.build()

   fixed = DataLoader(dyn_model, source={"d": np.arange(100.0)}, seq_length=10)
   full = DataLoader(
       dyn_model,
       source=[{"d": np.arange(50.0)}, {"d": np.arange(80.0)}],
       seq_length="full",
   )
   print(len(full), full.mask.shape)   # 2 (2, 80)

``on_short`` decides what happens to a simulation too short to fill one sample:
``"error"`` (the default) raises, ``"skip"`` leaves it out.

Normalization
-------------

Normalization is explicit and local to a loader.
:meth:`~nnodely.DataLoader.normalize` fits the statistics per feature and
transforms the loader in place. :meth:`~nnodely.DataLoader.denormalize` maps
values back, including model predictions, which are matched to the input their
target reads:

.. code-block:: python

   loader = DataLoader(model, source=signals)
   loader.normalize(method="standard")            # or "minmax", feature_range=(-1, 1)
   prediction = model(loader[0])
   physical = loader.denormalize({"y": np.asarray(prediction["y"])})
   loader.denormalize()                           # restore the original data
