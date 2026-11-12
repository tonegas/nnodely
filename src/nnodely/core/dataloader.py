from __future__ import annotations

import warnings
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Sequence,
    TypeAlias,
)

import numpy as np
import pandas as pd

from nnodely.core.shape import OUTER_AXIS, slice_axis
from nnodely.layers.input import Input, _seq_cell, _tw_cell
from nnodely.utils.printers import _table

sliding_window_view = np.lib.stride_tricks.sliding_window_view

#: A dict keyed by Input, by input name, or by both: three types, since a
#: dict of names alone is not a dict of ``Input | str`` keys to a checker.
InputData: TypeAlias = (
    Mapping[str, Any] | Mapping[Input, Any] | Mapping[Input | str, Any]
)

#: The columns feeding one input, in a ``format``: a column name, a
#: positional index, or a list of them, one per feature.
Columns: TypeAlias = str | int | Sequence[str | int]

InputFormat: TypeAlias = (
    Mapping[str, Columns] | Mapping[Input, Columns] | Mapping[Input | str, Columns]
)


class DataLoader:
    """
    Build a training dataset from simulation data.

    A simulation is one contiguous recording. Every window is built inside a
    single simulation, so no window ever spans two of them, and simulations may
    have different lengths.

    Sources::

        dict / list[dict]            arrays keyed by Input or input name
        DataFrame / list[DataFrame]  columns keyed by input name
        path to a .csv file          one simulation
        path to a folder             one simulation per file matching csv_glob

    For DataFrame and CSV sources, ``format`` maps an Input, or its name, to
    the column feeding it - a column name or a positional index - or to a list
    of columns when the input carries more than one feature::

        format={"vel": "vel", trq: 1, "alt": ["alt1", ..., "alt21"]}

    Every key - of a dict source, of ``format``, of :meth:`get_input` - is
    resolved to its Input as calling the model resolves it: a name several
    Inputs carry, or an Input the model does not read, raises. Columns, keys
    and ``format`` entries that name no input of the model are ignored.

    Windows:

    - Temporal windows, ``T``, come from ``Input(..., tw=...)``. All
      inputs are aligned so that every window ends on the same sample; an
      Input without a window reads that sample, with no tw axis.
    - Sequence windows, the sequence axes, come from ``Input(..., seq=...)``:
      one sliding window per declared length, applied on top of the temporal
      windows. Only one length may be dynamic (``-1``), and ``seq_length``
      resolves it.
    - ``step`` is the jump between one dataset sample and the next; every
      sequence level itself advances one sample at a time.
    - ``seq_length="full"`` resolves the dynamic length, which must be the
      outermost sequence axis, to the whole simulation, so every simulation
      yields exactly one sample. Simulations of different lengths are padded
      on that axis and ``mask`` marks the real steps; ``step`` is meaningless
      there and is rejected.
    - ``on_short`` decides what happens to a simulation with too few samples to
      fill the windows: raise, or leave it out.

    Final dataset format, laid out as the model's tensors::

        {name: np.ndarray of shape (N, *seq, tw, *dim)}

    Windowed sources:

    - ``windowed=True`` takes a dict, or a list of dicts, whose arrays (or
      nested lists) are already laid out that way, and uses them as they are:
      no window is built, nothing is aligned or dropped, sample ``i`` of every
      input is the ``i``-th row given. The last axis may be left out for an
      input of a single feature.
    - A dynamic sequence length is read from the shape, so an input may
      declare several, and ``seq_length`` is not needed.
    - The inputs whose outermost sequence axis is dynamic share one rollout.
      Its length may differ from one simulation to another: shorter ones are
      padded on that axis, and ``mask`` marks the real steps. Every other
      dynamic length has to match between simulations.
    - The flag is explicit because the shape alone cannot tell the two
      readings apart: an ``(N, 1, 1)`` array is ``N`` samples of one value
      either way, but a raw target ``tw=(0, 1)`` is shifted one sample ahead,
      and an already windowed one is not.

    Example::

        x = Input("x", tw=5)
        y = Output("y", Fir(features=1)([x]))
        model = Modely("model", inputs=[x], outputs=[y]).build()

        loader = DataLoader(model, source={"x": np.arange(10.0)})
        loader[0]  # {"x": array([[0.], [1.], [2.], [3.], [4.]])}

        windows = np.random.rand(100, 5, 1)  # (N, tw, dim)
        loader = DataLoader(model, source={"x": windows}, windowed=True)
        len(loader)  # 100
    """

    def __init__(
        self,
        model: Any,
        source: str
        | Path
        | InputData
        | pd.DataFrame
        | Sequence[InputData | pd.DataFrame],
        format: InputFormat | None = None,
        csv_glob: str = "*.csv",
        delimiter: str = ",",
        header: int | Sequence[int] | Literal["infer"] | None = "infer",
        dtype: Any = np.float32,
        seq_length: int | Literal["full"] | None = None,
        step: int = 1,
        on_short: Literal["error", "skip"] = "error",
        windowed: bool = False,
    ):
        if step < 1:
            raise ValueError(f"step must be a positive integer, got {step}.")
        if seq_length == "full" and step > 1:
            raise ValueError(
                "step has no meaning with seq_length='full': the sequence already "
                "spans the whole simulation, so there is a single starting point."
            )
        if not isinstance(windowed, bool):
            raise TypeError(f"windowed is True or False, got {windowed!r}.")
        if windowed and seq_length is not None:
            raise ValueError(
                "seq_length has no meaning with windowed=True: windowed data "
                "already carries every sequence length in its shape."
            )
        if windowed and step > 1:
            raise ValueError(
                "step has no meaning with windowed=True: windowed data is used as "
                "it is given, so slice it to keep fewer samples."
            )

        self._model = model
        self._csv_glob = csv_glob
        self._delimiter = delimiter
        self._header = header
        self._dtype = dtype
        self._seq_length = seq_length
        self._step = step
        self._on_short = on_short
        self._windowed = windowed

        if model.model is None:
            raise ValueError(
                f"Model {model.name} is not built. Make sure to call {model.name}.build() first."
            )
        # The data is keyed by the names the model was built with.
        model._check_names_built()

        # The Inputs the data feeds, as the build made them: the dataset is
        # keyed by their names, the keys of the Keras model.
        self._nodes = list(model.train_inputs)
        if not self._nodes:
            raise ValueError("Could not infer any inputs from model.inputs")
        self._format = self._by_node(format or {})
        self._input_specs = {node: [node.past, node.future] for node in self._nodes}

        # Number of CSV files the dataset was assembled from; None when the
        # samples came from an in-memory dict and no file is involved.
        self._num_files: int | None = None

        # Windowed data carries its sequence lengths, dynamic ones included,
        # in its shape: there is nothing to resolve.
        self._sequence_specs = {
            node: tuple(node.seq) if windowed else self._resolve_sequence(node)
            for node in self._nodes
        }

        self._dynamic_inputs = {
            node
            for node, sequence in self._sequence_specs.items()
            if any(length is None for length in sequence)
        }
        self.mask: np.ndarray | None = None
        if windowed:
            dataset = self._build_windowed(self._read_windowed(source))
        else:
            dataset = self._build(self._read(source))
        # Keyed by name only here, where the data meets Keras.
        self._dataset = {node.name: values for node, values in dataset.items()}
        if self.mask is not None:
            self._warn_uncollected_loops()

        self._num_steps = min(len(values) for values in self._dataset.values())

    @property
    def inputs(self) -> List[str]:
        """Names of the inputs the dataset holds."""
        return list(self._dataset.keys())

    def __repr__(self) -> str:
        """The dataset as a table, for `print(loader)`: a row per input, with
        its sequence as the loader resolved it, its window, its dim and the
        shape of its array, under the model's name, the number of samples and
        where they came from."""
        if self._num_files is None:
            source = "in-memory dict" + (", windowed" if self._windowed else "")
        else:
            source = f"{self._num_files} file{'' if self._num_files == 1 else 's'}"
        rows = []
        for node in self._nodes:
            values = self._dataset[node.name]
            rows.append(
                (
                    node.name,
                    _seq_cell(values.shape[1 : 1 + len(node.seq)]),
                    _tw_cell(node),
                    str(tuple(node.dim)),
                    str(tuple(values.shape)),
                )
            )
        title = f"{self._model.name}: {len(self)} samples, {source}"
        return _table(("input", "seq", "tw", "dim", "shape"), [(title, rows)])

    def __len__(self) -> int:
        return self._num_steps

    def get_input(self, key: Input | str) -> np.ndarray:
        """All the samples of one input - the Input, or its name - shaped
        ``(N, *tw, *dim)``."""
        node = self._node_of(self._model._input_of(key))
        if node is None:
            raise KeyError(
                f"The dataset holds no data of the Input "
                f"{getattr(key, 'name', key)!r}: "
                f"its inputs are {self.inputs}."
            )
        return self._dataset[node.name]

    def get_step(self, idx: int) -> Dict[str, Any]:
        """Sample ``idx`` as ``{name: array}``; the same as ``loader[idx]``."""
        if idx < 0 or idx >= self._num_steps:
            raise IndexError(f"idx out of range: {idx} (len={self._num_steps})")
        return {k: v[idx] for k, v in self._dataset.items()}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.get_step(idx)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for i in range(self._num_steps):
            yield self.get_step(i)

    def get_samples(
        self, n: int, start: int = 0, simulation: int | str = 0
    ) -> Dict[str, np.ndarray]:
        """``n`` consecutive samples of one simulation, laid out like as_dict().

        Consecutive samples are consecutive time steps (``step`` apart when the
        loader was built with ``step > 1``), so running the model on them
        returns a time series of ``n`` predictions. They never cross from one
        simulation into the next, where two unrelated trajectories would join.
        With ``windowed=True`` they are consecutive rows, in the order given.

        Parameters
        ----------
        n:
            The number of samples.
        start:
            The first sample, counted within that simulation.
        simulation:
            The index of the simulation, in reading order, or its label - the
            CSV file name, or ``"simulation <i>"`` for in-memory sources.
        """
        label, offset, count = self._simulation(simulation)
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError(f"n must be a positive integer, got {n!r}.")
        if isinstance(start, bool) or not isinstance(start, (int, np.integer)):
            raise TypeError(f"start must be an integer, got {type(start).__name__}.")
        if start < 0 or start + n > count:
            raise ValueError(
                f"'{label}' has {count} samples, so {n} consecutive samples "
                f"starting at {start} are not available."
            )
        first = offset + int(start)
        return {
            name: np.array(values[first : first + int(n)])
            for name, values in self._dataset.items()
        }

    def _simulation(self, simulation: int | str) -> tuple[str, int, int]:
        """A simulation's label, first sample and sample count."""
        labels = [label for label, _, _ in self._simulations]
        if isinstance(simulation, str):
            if simulation not in labels:
                raise ValueError(
                    f"No simulation named {simulation!r}. The dataset holds {labels}."
                )
            return self._simulations[labels.index(simulation)]
        if isinstance(simulation, bool) or not isinstance(
            simulation, (int, np.integer)
        ):
            raise TypeError(
                "simulation must be an index or a label, got "
                f"{type(simulation).__name__}."
            )
        if not 0 <= simulation < len(self._simulations):
            raise IndexError(
                f"simulation {simulation} out of range: the dataset holds "
                f"{len(self._simulations)} simulations {labels}."
            )
        return self._simulations[simulation]

    def as_dict(self) -> Dict[str, np.ndarray]:
        """The whole dataset as ``{name: array of shape (N, *tw, *dim)}``."""
        return self._dataset

    # ------------------------------------------------------------------
    # Model specs
    # ------------------------------------------------------------------

    def _resolve_sequence(self, node) -> tuple[int | None, ...]:
        """The lengths of an input's sequence axes, outermost first."""
        sequence_axes = node.seq
        dynamic = [
            index for index, length in enumerate(sequence_axes) if length is None
        ]
        if len(dynamic) > 1:
            raise ValueError(
                f"Input '{node.name}' declares more than one dynamic sequence length. "
                "Building the windows resolves a single dynamic sequence dimension; "
                "data already windowed carries every length in its shape, so pass "
                "it with windowed=True."
            )
        if dynamic and self._seq_length is None:
            raise ValueError(
                "Some inputs have undefined sequence length. "
                "Please specify seq_length in training, or pass data already "
                "windowed with windowed=True."
            )
        if dynamic and self._seq_length == "full" and dynamic[0] != 0:
            raise ValueError(
                f"Input '{node.name}' declares its dynamic sequence length at position "
                f"{dynamic[0]}, but seq_length='full' only resolves the outermost one."
            )

        sequence = []
        for length in sequence_axes:
            if length is None:
                if self._seq_length == "full":
                    # Resolved per simulation, once its own length is known.
                    sequence.append(None)
                    continue
                length = self._seq_length
            if length < 1:  # type: ignore
                raise ValueError(
                    f"Input '{node.name}' has invalid sequence length {length}."
                )
            sequence.append(int(length))  # type: ignore
        return tuple(sequence)

    def _node_of(self, input_):
        """The input the data feeds for ``input_``, an Input of the graph: the
        copy the build made of it, or None if the data feeds none."""
        return next(
            (node for node in self._nodes if self._model._original(node) is input_),
            None,
        )

    def _by_node(self, mapping: Mapping) -> dict:
        """``mapping``, keyed by Input or input name, keyed by the inputs the
        data feeds: every key resolved as calling the model resolves it, one
        that names no input of the data left out."""
        keyed = {}
        for input_, value in self._model._given_by_input(mapping).items():
            node = self._node_of(input_)
            if node is not None:
                keyed[node] = value
        return keyed

    def _optional(self, node) -> bool:
        """Whether the data may leave ``node`` out: it then reads zeros."""
        return self._model._original(node) in self._model._optional_inputs

    # ------------------------------------------------------------------
    # Read: source -> one {node: (samples, *dim)} array set per simulation
    # ------------------------------------------------------------------

    def _read(self, source) -> List[tuple[str, Dict[str, np.ndarray]]]:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Source does not exist: {path}")
            if path.is_dir():
                files = sorted(path.glob(self._csv_glob))
                if not files:
                    raise FileNotFoundError(
                        f"No CSV files matching '{self._csv_glob}' found in {path}"
                    )
            else:
                files = [path]
            self._num_files = len(files)
            return [
                (
                    file.name,
                    self._read_frame(
                        pd.read_csv(file, sep=self._delimiter, header=self._header),  # type: ignore
                        file.name,
                    ),
                )
                for file in files
            ]

        items = source if isinstance(source, (list, tuple)) else [source]
        simulations = []
        for index, item in enumerate(items):
            label = f"simulation {index}"
            if isinstance(item, pd.DataFrame):
                simulations.append((label, self._read_frame(item, label)))
            elif isinstance(item, dict):
                simulations.append((label, self._read_mapping(item, label)))
            else:
                raise TypeError(
                    f"Unsupported source {type(item).__name__}: "
                    "expected a dict, a DataFrame or a path."
                )
        return simulations

    def _columns(self, node) -> list:
        """Columns feeding an input: one per feature, in declaration order."""
        column = self._format.get(node, node.name)
        if isinstance(column, (list, tuple, range)):
            return list(column)
        return [column]

    def _read_frame(self, df: pd.DataFrame, label: str) -> Dict[str, np.ndarray]:
        if len(df) == 0:
            raise ValueError(f"'{label}' is empty.")

        missing = []
        left_out = set()
        for node in self._input_specs:
            absent = []
            for column in self._columns(node):
                if isinstance(column, (int, np.integer)):
                    if not 0 <= column < df.shape[1]:
                        absent.append(f"{node.name} -> index {column}")
                elif column not in df.columns:
                    absent.append(f"{node.name} -> '{column}'")
            # An optional input none of whose columns is there reads zeros.
            if self._optional(node) and len(absent) == len(self._columns(node)):
                left_out.add(node)
            else:
                missing.extend(absent)
        if missing:
            raise ValueError(
                f"'{label}' is missing required columns or indices: {missing}"
            )

        data = {}
        for node in self._input_specs:
            if node in left_out:
                data[node] = np.zeros((len(df), *node.dim), self._dtype)
                continue
            values = np.stack(
                [
                    (
                        df.iloc[:, column]  # type: ignore
                        if isinstance(column, (int, np.integer))
                        else df[column]
                    ).to_numpy(dtype=self._dtype)
                    for column in self._columns(node)
                ],
                axis=-1,
            )
            data[node] = self._as_samples(node, values, label)
        return data

    def _given_inputs(self, mapping: dict, label: str) -> dict:
        """The inputs a dict gives, ``{node: values}``, once every required
        one is found in it."""
        given = self._by_node(mapping)
        absent = [node for node in self._input_specs if node not in given]
        missing = [node.name for node in absent if not self._optional(node)]
        if missing or len(absent) == len(self._input_specs):
            raise ValueError(
                f"'{label}' is missing required inputs: "
                f"{missing or [node.name for node in absent]}"
            )
        return {node: given[node] for node in self._input_specs if node in given}

    def _read_mapping(self, mapping: dict, label: str) -> Dict[Any, np.ndarray]:
        data = {}
        for node, values in self._given_inputs(mapping, label).items():
            if isinstance(values, (pd.Series, pd.DataFrame)):
                values = values.to_numpy(dtype=self._dtype)
            data[node] = self._as_samples(node, values, label)

        lengths = {node.name: values.shape[0] for node, values in data.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"Input arrays are not aligned in length: {lengths}.")
        if min(lengths.values()) == 0:
            raise ValueError(f"'{label}' contains an empty input array.")
        # An optional input left out reads zeros.
        length = next(iter(lengths.values()))
        return {
            node: data[node]
            if node in data
            else np.zeros((length, *node.dim), self._dtype)
            for node in self._input_specs
        }

    def _as_samples(self, node, values: Any, label: str) -> np.ndarray:
        """Check the feature count against dim and lay the input out as [samples, *dim]."""
        name = node.name
        values = np.asarray(values, dtype=self._dtype)
        if values.ndim == 0:
            raise ValueError(f"Input '{name}' must be at least 1D, got scalar")

        dim = tuple(node.dim)
        feature_size = int(np.prod(values.shape[1:], dtype=int))
        expected_size = int(np.prod(dim, dtype=int))
        if feature_size != expected_size:
            hint = ""
            if self._windowed_shape(node, values.shape) is not None:
                hint = (
                    f" It is laid out as the tensor of '{name}', "
                    f"{self._tensor_layout(node)}: pass windowed=True if its "
                    "windows are already built."
                )
            raise ValueError(
                f"Input '{name}' in '{label}' provides {feature_size} values per "
                f"timestep, but its dim={dim} requires {expected_size}.{hint}"
            )
        # The values are reshaped to dim, which keeps their order: laid out
        # as dim, or flat, they land where they belong; any other layout of
        # as many values would be reordered without a word. Axes of one value
        # place nothing, so they are left out of the comparison.
        layout = tuple(size for size in values.shape[1:] if size != 1)
        if len(layout) > 1 and layout != tuple(size for size in dim if size != 1):
            raise ValueError(
                f"Input '{name}' in '{label}' is laid out {values.shape[1:]} per "
                f"timestep, but its dim={dim}: give it as (timesteps, *dim), or "
                f"flat, (timesteps, {expected_size}), the features in row-major "
                "order."
            )
        self._warn_missing(name, values, label)
        return np.reshape(values, (values.shape[0], *dim))

    @staticmethod
    def _warn_missing(name: str, values: np.ndarray, label: str) -> None:
        """Warn about the n/a values of an input, which are kept."""
        # A cell pandas reads as n/a - empty, "n/a", "NaN" - stays a NaN, and a
        # NaN reaching a minimizer turns its loss, and every weight it trains,
        # into NaN.
        missing = int(np.isnan(values).sum())
        if missing:
            # Raised from the DataLoader(...) call: __init__ -> read -> read
            # one simulation -> check one input -> here.
            warnings.warn(
                f"'{label}': input '{name}' has {missing} n/a values, kept as NaN.",
                UserWarning,
                stacklevel=6,
            )

    # ------------------------------------------------------------------
    # Read windowed: source -> one {name: (N, *seq, tw, *dim)} set per
    # simulation, used as given
    # ------------------------------------------------------------------

    def _read_windowed(self, source) -> List[tuple[str, Dict[str, np.ndarray]]]:
        items = source if isinstance(source, (list, tuple)) else [source]
        if not items:
            raise ValueError("The source holds no simulation: the list is empty.")
        simulations = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise TypeError(
                    f"windowed=True reads a dict of arrays, or a list of them, got "
                    f"{type(item).__name__}: a DataFrame or a CSV file holds one "
                    "sample per row, which the loader windows itself, so leave "
                    "windowed False for it."
                )
            label = f"simulation {index}"
            simulations.append((label, self._read_windowed_mapping(item, label)))
        return simulations

    def _rolls_out(self, node) -> bool:
        """Whether an input's outermost sequence axis is dynamic: the rollout
        that every such input shares, and the axis padded and masked."""
        sequence = self._sequence_specs[node]
        return bool(sequence) and sequence[0] is None

    def _read_windowed_mapping(
        self, mapping: dict, label: str
    ) -> Dict[Any, np.ndarray]:
        # A loop, not a comprehension: before Python 3.12 that is a frame of
        # its own, and the n/a warning would point inside the loader.
        data = {}
        for node, values in self._given_inputs(mapping, label).items():
            data[node] = self._as_windows(node, values, label)

        counts = {node.name: values.shape[0] for node, values in data.items()}
        if len(set(counts.values())) > 1:
            raise ValueError(
                f"'{label}': the inputs hold different numbers of samples: {counts}."
            )
        count = next(iter(counts.values()))
        if count == 0:
            raise ValueError(f"'{label}' contains an empty input array.")
        rollouts = {
            node.name: values.shape[OUTER_AXIS]
            for node, values in data.items()
            if self._rolls_out(node)
        }
        if len(set(rollouts.values())) > 1:
            raise ValueError(
                f"'{label}': the inputs whose outermost sequence axis is dynamic "
                f"share one rollout, but its lengths differ: {rollouts}."
            )

        # An optional input left out reads zeros, over the rollout of the
        # others when it shares it, one step of any other dynamic length.
        rollout = next(iter(rollouts.values()), 1)
        zeros = {}
        for node in self._input_specs:
            if node in data:
                continue
            layout = list(node.shape.tuple)
            if self._rolls_out(node):
                layout[0] = rollout
            layout = [1 if size is None else size for size in layout]
            zeros[node] = np.zeros((count, *layout), self._dtype)
        return {
            node: data[node] if node in data else zeros[node]
            for node in self._input_specs
        }

    def _tensor_layout(self, node) -> str:
        """An input's windowed layout as an error prints it: (N, dyn, 10, 1)."""
        sizes = ["dyn" if size is None else str(size) for size in node.shape.tuple]
        return f"(N, {', '.join(sizes)})"

    def _windowed_shape(self, node, shape: tuple) -> tuple | None:
        """``shape`` read as the tensor of an input, ``(N, *seq, tw, *dim)``,
        or None when it is not: every fixed axis its size, every dynamic one
        at least a step. The last axis may be left out for a single feature."""
        expected = node.shape.tuple
        shape = tuple(shape)
        if tuple(node.dim) == (1,) and len(shape) == len(expected):
            shape = (*shape, 1)
        if len(shape) != len(expected) + 1:
            return None
        for size, given in zip(expected, shape[1:]):
            if (given < 1) if size is None else (given != size):
                return None
        return shape

    def _as_windows(self, node, values: Any, label: str) -> np.ndarray:
        """Check an already windowed input against the layout of its tensor."""
        name = node.name
        try:
            values = np.asarray(values, dtype=self._dtype)
        except ValueError as error:
            raise ValueError(
                f"Input '{name}' in '{label}' cannot be read as one array: {error}. "
                "Windowed data gives every sample the same shape; samples of "
                "different rollout lengths go in simulations of their own, a list "
                "of dicts, which the loader pads and masks."
            ) from error
        shape = self._windowed_shape(node, values.shape)
        if shape is None:
            raise ValueError(
                f"Input '{name}' in '{label}' is {values.shape}, but windowed data "
                f"is laid out as its tensor, {self._tensor_layout(node)}, for its "
                f"{node.shape}: the samples first, then every "
                "sequence axis, the window when there is one and the dim, whose "
                "last axis may be left out for a single feature."
            )
        self._warn_missing(name, values, label)
        return np.reshape(values, shape)

    # ------------------------------------------------------------------
    # Window: simulations -> {node: (N, *tw, *dim)}
    # ------------------------------------------------------------------

    def _build(
        self, simulations: List[tuple[str, Dict[Any, np.ndarray]]]
    ) -> Dict[Any, np.ndarray]:
        max_past = max(spec[0] for spec in self._input_specs.values())
        max_future = max(spec[1] for spec in self._input_specs.values())

        # Every input is windowed over the same range of end samples: it opens
        # once the deepest past window is covered and closes early enough to
        # leave room for the deepest future one.
        first = max(max_past - 1, 0)
        # A dynamic sequence is resolved per simulation, so only the levels
        # declared up front can be required of every simulation.
        fixed_spans = {
            node: sum(length - 1 for length in sequence if length is not None)
            for node, sequence in self._sequence_specs.items()
        }
        required = first + max_future + max(fixed_spans.values()) + 1
        # Every dynamic input shares one rollout axis, so its length is the one
        # the deepest of them can fill.
        dynamic_span = max(
            (fixed_spans[node] for node in self._dynamic_inputs), default=0
        )

        chunks: Dict[Any, List[np.ndarray]] = {node: [] for node in self._input_specs}
        lengths: List[int] = []
        skipped = []
        # Where each simulation's samples sit once all of them are concatenated:
        # a time series only runs within one simulation, never across two.
        self._simulations: List[tuple[str, int, int]] = []
        offset = 0
        for label, simulation in simulations:
            samples = next(iter(simulation.values())).shape[0]
            if samples < required:
                if self._on_short == "skip":
                    skipped.append(label)
                    continue
                raise ValueError(
                    f"'{label}' has only {samples} samples, but the model requires at "
                    f"least {required}. Pass on_short='skip' to leave short "
                    "simulations out."
                )

            windows = samples - max_future - first
            length = windows - dynamic_span
            sequences = {
                node: tuple(
                    length if declared is None else declared for declared in sequence
                )
                for node, sequence in self._sequence_specs.items()
            }
            spans = {
                node: sum(size - 1 for size in sequence)
                for node, sequence in sequences.items()
            }
            max_span = max(spans.values())
            for node, values in simulation.items():
                chunks[node].append(
                    self._windows(
                        node,
                        values,
                        first,
                        max_future,
                        max_span - spans[node],
                        sequences[node],
                    )
                )
            lengths.append(length)
            count = chunks[next(iter(chunks))][-1].shape[0]
            self._simulations.append((label, offset, count))
            offset += count

        if len(skipped) == len(simulations):
            raise ValueError(
                f"Every simulation is shorter than the {required} samples the model "
                "requires."
            )

        # A dynamic sequence spans its whole simulation, so simulations of
        # different lengths only line up once the outermost sequence axis is padded.
        if self._dynamic_inputs and len(set(lengths)) > 1:
            widest = max(lengths)
            for node in self._dynamic_inputs:
                chunks[node] = [self._pad(values, widest) for values in chunks[node]]
            self.mask = np.arange(widest) < np.asarray(lengths)[:, np.newaxis]

        return {
            node: np.concatenate(windows, axis=0) for node, windows in chunks.items()
        }

    def _build_windowed(
        self, simulations: List[tuple[str, Dict[Any, np.ndarray]]]
    ) -> Dict[Any, np.ndarray]:
        """Stack windowed simulations, padding the rollouts that are shorter."""
        rolled = [node for node in self._input_specs if self._rolls_out(node)]

        # Only the rollout is padded, and the mask covers only that axis, so
        # every other dynamic length has to be the same in every simulation.
        first_label, first = simulations[0]
        for label, simulation in simulations[1:]:
            for node, values in simulation.items():
                kept = slice(2 if node in rolled else 1, None)
                if values.shape[kept] != first[node].shape[kept]:
                    raise ValueError(
                        f"Input '{node.name}' is {values.shape[1:]} per sample in "
                        f"'{label}', but {first[node].shape[1:]} in '{first_label}': "
                        "only the outermost dynamic sequence axis, the rollout, may "
                        "differ between simulations - it is padded and masked - so "
                        "every other length must match."
                    )

        self._simulations = []
        offset = 0
        counts, rollouts = [], []
        for label, simulation in simulations:
            count = next(iter(simulation.values())).shape[0]
            self._simulations.append((label, offset, count))
            offset += count
            counts.append(count)
            if rolled:
                rollouts.append(simulation[rolled[0]].shape[OUTER_AXIS])

        chunks = {
            node: [simulation[node] for _, simulation in simulations]
            for node in self._input_specs
        }
        if len(set(rollouts)) > 1:
            widest = max(rollouts)
            for node in rolled:
                chunks[node] = [self._pad(values, widest) for values in chunks[node]]
            # One row per sample: every sample of a simulation has its rollout.
            self.mask = np.repeat(
                np.arange(widest) < np.asarray(rollouts)[:, np.newaxis],
                counts,
                axis=0,
            )

        # Concatenated even when there is one simulation: the dataset is a
        # copy, which the caller's arrays changing later does not reach.
        return {node: np.concatenate(values, axis=0) for node, values in chunks.items()}

    def _warn_uncollected_loops(self) -> None:
        """Warn about a Loop that returns only its last, padded, rollout step."""
        uncollected = sorted(
            node.name
            for node in self._model._order
            if getattr(node, "_collect", None) is False
        )
        if not uncollected:
            return
        warnings.warn(
            f"{uncollected} roll out with collect=False while the simulations have "
            "different lengths. Such a Loop returns the state at the padded rollout "
            "length, which for a shorter simulation lies past the end of its data, "
            "and the mask cannot correct it: an uncollected output has no rollout "
            "axis to mask. Use collect=True, or simulations of equal length.",
            UserWarning,
            stacklevel=3,
        )

    @staticmethod
    def _pad(values: np.ndarray, width: int) -> np.ndarray:
        """Extend the outermost sequence axis to ``width`` by repeating its last step.

        Repeating keeps the padded part of a rollout inside the range the model
        was fitted on; the mask is what removes it from the loss.
        """
        missing = width - values.shape[OUTER_AXIS]
        if missing <= 0:
            return values
        last = slice_axis(values, OUTER_AXIS, slice(-1, None))
        return np.concatenate(
            [values, np.repeat(last, missing, axis=OUTER_AXIS)],
            axis=OUTER_AXIS,
        )

    def _windows(
        self,
        node,
        values: np.ndarray,
        first: int,
        max_future: int,
        lead: int,
        sequence: tuple[int, ...],
    ) -> np.ndarray:
        """Temporal windows of one input, then one sliding window per
        sequence axis, laid out ``(N, *seq, tw, *dim)``, or ``(N, *dim)``
        for an input without a window."""
        past, future = self._input_specs[node]
        width = past + future
        last = values.shape[0] - max_future  # one past the last aligned end sample

        # Window i covers values[i:i + width] and ends on sample i + past - 1,
        # so the shared range of end samples selects the windows to keep. An
        # Input without a window is the one sample, without the window axis.
        windows = sliding_window_view(values, window_shape=width, axis=0)
        windows = np.moveaxis(windows[first - past + 1 : last - past + 1], -1, 1)
        if node.shape.tw is None:
            windows = windows[:, 0]

        # Each sliding window appends its axis; moved right after the samples it
        # becomes the outermost axis, so the innermost outer axis comes first.
        for length in reversed(sequence):
            windows = sliding_window_view(windows, window_shape=length, axis=0)
            windows = np.moveaxis(windows, -1, 1)

        # A sequence window is aligned to its final temporal sample. Inputs with
        # shorter/no seq dimensions therefore skip earlier samples so every model
        # input refers to the same endpoint.
        if lead:
            windows = windows[lead:]
        if self._step > 1:
            windows = windows[:: self._step]
        return np.ascontiguousarray(windows, dtype=self._dtype)
