Loading Data
============

At the end of this page you will be able to turn recorded signals into the
samples a model is trained and validated on: from arrays, DataFrames or CSV
files, from several recordings, and as sequences for multi-step models.

**Prerequisites:** :doc:`models`.

**Tutorial:** :doc:`../tutorial/02_data_management`.

**The example.** A water tank. The model predicts the next level from the
last three levels and from the two inflows of now. Its target is the level
of the next sample.

Step 1: a built model
---------------------

A :class:`~nnodely.DataLoader` reads the inputs of a model, their windows and
their sequences, so **build the model before creating the loader**:

.. code-block:: python

   import numpy as np
   import pandas as pd
   from nnodely import DataLoader, Fir, Input, Modely, Output

   level = Input("level", tw=3)              # level[t-2], level[t-1], level[t]
   inflow = Input("inflow", dim=2, tw=1)     # two inflows, now
   next_level = Input("next_level", tw=(0, 1))  # level[t+1], the target
   prediction = Output("prediction", Fir()(level) + Fir()(inflow))

   model = Modely("tank", inputs=[level, inflow], outputs=[prediction])
   model.minimize("fit", prediction, next_level)
   model.build()

Step 2: how samples are cut
---------------------------

A **dictionary** maps every input name to one array per signal. The first
axis is time; the other axes are the features of the input, laid out as its
``dim``. With signals that count the samples, the windows are easy to read:

.. code-block:: python

   counting = {
       "level": np.arange(10.0),
       "inflow": np.arange(20.0).reshape(10, 2),
       "next_level": np.arange(10.0),
   }
   loader = DataLoader(model, source=counting)
   print(loader)

.. code-block:: text
   :class: output

   ┏━━━━━━━━━━━━┳━━━━━┳━━━━━━━━┳━━━━━━┳━━━━━━━━━━━┓
   ┃ input      ┃ seq ┃ tw     ┃ dim  ┃ shape     ┃
   ┡━━━━━━━━━━━━┻━━━━━┻━━━━━━━━┻━━━━━━┻━━━━━━━━━━━┩
   │ tank: 7 samples, in-memory dict              │
   ├────────────┬─────┬────────┬──────┬───────────┤
   │ level      │ ()  │ 3      │ (1,) │ (7, 3, 1) │
   │ inflow     │ ()  │ 1      │ (2,) │ (7, 1, 2) │
   │ next_level │ ()  │ (0, 1) │ (1,) │ (7, 1, 1) │
   └────────────┴─────┴────────┴──────┴───────────┘

Every sample is cut around one current sample ``t``, the same for every
input. The first sample needs two levels before ``t``, and the last one a
level after it, so 10 samples of signal give 7 samples:

.. code-block:: python

   sample = loader[0]   # t = 2
   print(sample["level"].ravel(), sample["inflow"].ravel(), sample["next_level"].ravel())

.. code-block:: text
   :class: output

   [0. 1. 2.] [4. 5.] [3.]

A window ``tw=n`` reads the ``n`` samples up to ``t``. A window
``tw=(p, f)`` reads ``p`` samples up to ``t`` and the ``f`` after it. A
signal of ``N`` samples gives ``N - (P - 1) - F`` samples, with ``P`` the
longest past and ``F`` the longest future among the inputs. A sequence of
``L`` steps needs ``L - 1`` samples more (see Step 6).

Step 3: samples and arrays
--------------------------

Every sample is a dictionary with one array per input, shaped ``(*seq, tw,
*dim)``, or ``(*dim)`` for an input without a window. A sample has **no
batch axis**. The loader can be indexed, iterated, and converted to the
arrays of all its samples:

.. code-block:: python

   print({name: value.shape for name, value in loader[0].items()})
   arrays = loader.as_dict()       # every sample: (samples, *seq, tw, *dim)
   batch = loader.get_samples(2)   # the first 2 samples, with a batch axis
   print(arrays["level"].shape, batch["level"].shape)

.. code-block:: text
   :class: output

   {'level': (3, 1), 'inflow': (1, 2), 'next_level': (1, 1)}
   (7, 3, 1) (2, 3, 1)

To run the model on one sample, call it on ``loader.get_samples(1)``, which
has the batch axis.

``step`` sets how many samples separate two consecutive windows:

.. code-block:: python

   print(len(DataLoader(model, source=counting, step=2)))

.. code-block:: text
   :class: output

   4

Step 4: DataFrames and CSV files
--------------------------------

A **DataFrame**, a **CSV file**, or a **folder** of CSV files works as well.
``format`` maps each input name to its columns:

- a column name, or its position;
- a list of columns, for an input with several features.

Columns that no input reads are ignored:

.. code-block:: python

   n = 200
   t = np.arange(n) * 0.1
   frame = pd.DataFrame({
       "time": t,
       "h": np.sin(t),
       "q1": np.cos(t),
       "q2": np.ones(n),
       "h_next": np.sin(t + 0.1),
   })
   frame.to_csv("run_0.csv", index=False)

   format = {"level": "h", "inflow": ["q1", "q2"], "next_level": 4}
   from_frame = DataLoader(model, source=frame, format=format)
   from_file = DataLoader(model, source="run_0.csv", format=format)
   from_folder = DataLoader(model, source=".", format=format, csv_glob="run_*.csv")
   print(len(from_frame), len(from_file), len(from_folder))

.. code-block:: text
   :class: output

   197 197 197

``delimiter`` and ``header`` are passed to ``pandas.read_csv``.
``header=None`` reads a file without a header row, whose columns ``format``
then names by position. A missing cell, such as an empty one or ``NaN``,
stays a NaN, and the loader warns about it.

Step 5: several recordings
--------------------------

A **list** of dictionaries or DataFrames, or a folder of several files, is a
set of *simulations*: independent recordings, possibly of different lengths.
A window never spans two simulations:

.. code-block:: python

   first = {name: values[:6] for name, values in counting.items()}
   second = {name: values[6:] for name, values in counting.items()}
   loader = DataLoader(model, source=[first, second])
   print(len(loader))   # 3 samples from the first, 1 from the second

.. code-block:: text
   :class: output

   4

``on_short`` decides what happens to a simulation too short to give one
sample: ``"error"``, the default, raises, and ``"skip"`` leaves it out.

Step 6: sequences
-----------------

An Input declared with ``seq`` reads a trajectory: ``seq`` consecutive
windows. The loader cuts sequences of that many steps, sliding by one step.
Rollouts read them (see :doc:`recurrent`):

.. code-block:: python

   s = Input("s", seq=20, tw=1)
   seq_model = Modely("sequence", inputs=[s], outputs=[Output("s_out", s * 2.0)])
   seq_model.build()

   seq_loader = DataLoader(seq_model, source={"s": np.arange(100.0)})
   print(len(seq_loader), seq_loader[0]["s"].shape)   # 20 steps of one sample

.. code-block:: text
   :class: output

   81 (20, 1, 1)

A sequence can be left dynamic, ``seq=-1``. Its length is then set when the
data is loaded:

- ``seq_length=n`` cuts sequences of ``n`` steps;
- ``seq_length="full"`` makes every simulation one sample spanning all of
  it. The dynamic length must be the outermost one. Shorter simulations are
  padded to the longest, and ``loader.mask`` marks their real steps.
  Training and validation leave the padded steps out.

.. code-block:: python

   d = Input("d", seq=-1, tw=1)
   dyn_model = Modely("dynamic", inputs=[d], outputs=[Output("d_out", d)])
   dyn_model.build()

   fixed = DataLoader(dyn_model, source={"d": np.arange(100.0)}, seq_length=10)
   full = DataLoader(
       dyn_model,
       source=[{"d": np.arange(50.0)}, {"d": np.arange(80.0)}],
       seq_length="full",
   )
   print(len(fixed), len(full), full.mask.shape)

.. code-block:: text
   :class: output

   91 2 (2, 80)

A :class:`~nnodely.ModelyLoop` reads every Input as a dynamic sequence (see
:doc:`recurrent`). A fed-back Input that it declares is optional in the
data: left out, it reads zeros.

Step 7: data already windowed
-----------------------------

Data whose windows are already cut, by another tool or by hand, is passed
with ``windowed=True``. Every array is then laid out as the tensor of its
input, ``(samples, *seq, tw, *dim)``, and used as it is: sample ``i`` of
every input is row ``i`` of its array. The last axis may be left out for an
input of one feature:

.. code-block:: python

   windows = DataLoader(
       model,
       source={
           "level": np.random.rand(500, 3),         # (N, tw): one feature
           "inflow": np.random.rand(500, 1, 2),     # (N, tw, dim)
           "next_level": np.random.rand(500, 1),
       },
       windowed=True,
   )
   print(len(windows))

.. code-block:: text
   :class: output

   500

A dynamic sequence length is read from the shape of the arrays. The
simulations of a list may have rollouts of different lengths: the shorter
ones are padded, and ``loader.mask`` marks the real steps.

The flag is explicit because the shape alone cannot tell raw signals from
windows. ``windowed=True`` takes dictionaries only, and refuses ``step`` and
``seq_length``, which only shape the windows the loader cuts.

Common errors
-------------

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Error
     - Cause
     - Fix
   * - ``Model m is not built``
     - The loader was created before ``build()``.
     - Call ``model.build()`` first.
   * - ``'simulation 0' is missing required inputs: ['u']``
     - The data has no signal for an input of the model.
     - Add it, or map it to a column with ``format``.
   * - ``'simulation 0' has only 5 samples, but the model requires at least 20``
     - A recording shorter than the longest window.
     - Pass ``on_short="skip"`` to leave it out.
   * - ``step has no meaning with windowed=True``
     - ``step`` or ``seq_length`` with windowed data.
     - Slice the arrays instead.

See also
--------

:class:`~nnodely.DataLoader`, :class:`~nnodely.Input`,
:meth:`Modely.train <nnodely.Modely.train>`, :doc:`recurrent`.
