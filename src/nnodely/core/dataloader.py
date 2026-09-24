from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Union

import numpy as np
import pandas as pd

sliding_window_view = np.lib.stride_tricks.sliding_window_view


class DataLoader:
    """
    Build a training dataset from simulation data.

    A simulation is one contiguous recording. Every window is built inside a
    single simulation, so no window ever spans two of them, and simulations may
    have different lengths.

    Sources:
        dict / list[dict]            arrays keyed by input name
        DataFrame / list[DataFrame]  columns keyed by input name
        path to a .csv file          one simulation
        path to a folder             one simulation per file matching csv_glob

    ``format`` maps an input name to the column feeding it - a column name or a
    positional index - or to a list of columns when the input carries more than
    one feature:

        format={"vel": "vel", "trq": 1, "alt": ["alt1", ..., "alt21"]}

    Columns that no input maps to are ignored.

    Windows:
    - Temporal windows come from ``Input.sw()``. All inputs are aligned so that
      every window ends on the same sample.
    - Sequence windows come from ``Input(seq=...)``: one sliding window per
      declared length, applied on top of the temporal windows, the outermost
      sequence last. Only one length may be dynamic (``None``), and
      ``seq_length`` resolves it.
    - ``step`` is the jump between one dataset sample and the next; every
      sequence level itself advances one sample at a time.
    - ``seq_length="full"`` resolves the dynamic length to the whole simulation,
      so every simulation yields exactly one sample. Simulations of different
      lengths are padded on the rollout axis and ``mask`` marks the real steps;
      ``step`` is meaningless there and is rejected.
    - ``on_short`` decides what happens to a simulation with too few samples to
      fill the windows: raise, or leave it out.

    Final dataset format:
        {name: np.ndarray of shape (N, *dim, time, *seq)}

    Example:
        data_1 = Input('data_1', dim=1)
        data_2 = Input('data_2', dim=1).sw(1)
        ...
        model = Model(..., inputs=[data_1, data_2], ...)

        loader = DataLoader(model, folder="data")
        sample = loader[0]

        sample == {
            "data_1": np.array([1,2,3,4,5]),
            "data_2": np.array([2])
        }
    """

    def __init__(
        self,
        model: Any,
        source: str | Path | dict | pd.DataFrame | list,
        format: dict[str, Any] | None = None,
        csv_glob: str = "*.csv",
        delimiter: str = ",",
        header: Union[int, None, Literal["infer"]] = "infer",
        dtype: Any = np.float32,
        seq_length: int | Literal["full"] | None = None,
        step: int = 1,
        on_short: Literal["error", "skip"] = "error",
    ):
        if step < 1:
            raise ValueError(f"step must be a positive integer, got {step}.")
        if seq_length == "full" and step > 1:
            raise ValueError(
                "step has no meaning with seq_length='full': the sequence already "
                "spans the whole simulation, so there is a single starting point."
            )

        self.model = model
        self.format = format
        self.csv_glob = csv_glob
        self.delimiter = delimiter
        self.header = header
        self.dtype = dtype
        self.seq_length = seq_length
        self.step = step
        self.on_short = on_short

        if model.model is None:
            raise ValueError(
                f"Model {model.name} is not built. Make sure to call {model.name}.build() first."
            )

        self.input_nodes = {node.name: node for node in model.train_inputs}
        if not self.input_nodes:
            raise ValueError("Could not infer any inputs from model.inputs")
        self.input_specs = {
            name: [node.past, node.future] for name, node in self.input_nodes.items()
        }

        # Number of CSV files the dataset was assembled from; None when the
        # samples came from an in-memory dict and no file is involved.
        self.num_files: int | None = None

        self.sequence_specs = {
            name: self._resolve_sequence(node)
            for name, node in self.input_nodes.items()
        }

        self._dynamic_inputs = {
            name
            for name, sequence in self.sequence_specs.items()
            if any(length is None for length in sequence)
        }
        self.mask: np.ndarray | None = None
        self.padded_inputs: set[str] = set()
        self.dataset = self._build(self._read(source))
        if self.mask is not None:
            self._warn_uncollected_loops()

        self.normalization_stats: Dict[str, Dict[str, Any]] = {}
        self._original_dataset: Dict[str, np.ndarray] | None = None
        self._normalization_aliases = self._build_normalization_aliases()
        self._num_steps = min(len(values) for values in self.dataset.values())

    @property
    def inputs(self) -> List[str]:
        return list(self.dataset.keys())

    def __repr__(self) -> str:
        """Table of what the loader actually built, for `print(loader)`."""
        width = 80
        label = 30
        lines = [" nnodely Model Dataset ".center(width, "=")]

        def row(name: str, value: Any) -> None:
            lines.append(f"{name + ':':<{label}}{value}")

        row("Dataset Name", self.model.name)
        if self.num_files is None:
            row("Source", "in-memory dict")
        else:
            row("Number of files", self.num_files)
        row("Total number of samples", len(self))
        for name, values in self.dataset.items():
            row(f"Shape of {name}", tuple(values.shape))
        if self.normalization_stats:
            methods = {stats["method"] for stats in self.normalization_stats.values()}
            row(
                "Normalization",
                f"{'/'.join(sorted(methods))} "
                f"({len(self.normalization_stats)}/{len(self.dataset)} inputs)",
            )

        lines.append("=" * width)
        return "\n".join(lines)

    def __len__(self) -> int:
        return self._num_steps

    def get_input(self, name: str) -> np.ndarray:
        return self.dataset[name]

    def get_step(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= self._num_steps:
            raise IndexError(f"idx out of range: {idx} (len={self._num_steps})")
        return {k: v[idx] for k, v in self.dataset.items()}

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

        simulation: index of the simulation, in reading order, or its label -
            the CSV file name, or ``"simulation <i>"`` for in-memory sources.
        start: the first sample, counted within that simulation.
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
            for name, values in self.dataset.items()
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
        return self.dataset

    # ------------------------------------------------------------------
    # Model specs
    # ------------------------------------------------------------------

    def _resolve_sequence(self, node) -> tuple[int | None, ...]:
        dynamic = [index for index, length in enumerate(node.seq) if length is None]
        if len(dynamic) > 1:
            raise ValueError(
                f"Input '{node.name}' declares more than one dynamic sequence length. "
                "Only a single dynamic sequence dimension is supported."
            )
        if dynamic and self.seq_length is None:
            raise ValueError(
                "Some inputs have undefined sequence length. "
                "Please specify seq_length in training."
            )
        if dynamic and self.seq_length == "full" and dynamic[0] != len(node.seq) - 1:
            raise ValueError(
                f"Input '{node.name}' declares its dynamic sequence length at position "
                f"{dynamic[0]}, but seq_length='full' only resolves the outermost one."
            )

        sequence = []
        for length in node.seq:
            if length is None:
                if self.seq_length == "full":
                    # Resolved per simulation, once its own length is known.
                    sequence.append(None)
                    continue
                length = self.seq_length
            if length < 1:  # type: ignore
                raise ValueError(
                    f"Input '{node.name}' has invalid sequence length {length}."
                )
            sequence.append(int(length))  # type: ignore
        return tuple(sequence)

    # ------------------------------------------------------------------
    # Read: source -> one {name: (samples, *dim)} array set per simulation
    # ------------------------------------------------------------------

    def _read(self, source) -> List[tuple[str, Dict[str, np.ndarray]]]:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Source does not exist: {path}")
            if path.is_dir():
                files = sorted(path.glob(self.csv_glob))
                if not files:
                    raise FileNotFoundError(
                        f"No CSV files matching '{self.csv_glob}' found in {path}"
                    )
            else:
                files = [path]
            self.num_files = len(files)
            return [
                (file.name, self._read_frame(pd.read_csv(file), file.name))
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

    def _columns(self, name: str) -> list:
        """Columns feeding an input: one per feature, in declaration order."""
        column = self.format.get(name, name) if self.format else name
        if isinstance(column, (list, tuple, range)):
            return list(column)
        return [column]

    def _read_frame(self, df: pd.DataFrame, label: str) -> Dict[str, np.ndarray]:
        if len(df) == 0:
            raise ValueError(f"'{label}' is empty.")

        missing = []
        for name in self.input_specs:
            for column in self._columns(name):
                if isinstance(column, (int, np.integer)):
                    if not 0 <= column < df.shape[1]:
                        missing.append(f"{name} -> index {column}")
                elif column not in df.columns:
                    missing.append(f"{name} -> '{column}'")
        if missing:
            raise ValueError(
                f"'{label}' is missing required columns or indices: {missing}"
            )

        data = {}
        for name in self.input_specs:
            values = np.stack(
                [
                    (
                        df.iloc[:, column]  # type: ignore
                        if isinstance(column, (int, np.integer))
                        else df[column]
                    ).to_numpy(dtype=self.dtype)
                    for column in self._columns(name)
                ],
                axis=-1,
            )
            data[name] = self._as_samples(name, values, label)
        return data

    def _read_mapping(self, mapping: dict, label: str) -> Dict[str, np.ndarray]:
        missing = [name for name in self.input_specs if name not in mapping]
        if missing:
            raise ValueError(f"'{label}' is missing required inputs: {missing}")

        data = {}
        for name in self.input_specs:
            values = mapping[name]
            if isinstance(values, (pd.Series, pd.DataFrame)):
                values = values.to_numpy(dtype=self.dtype)
            data[name] = self._as_samples(name, values, label)

        lengths = {name: values.shape[0] for name, values in data.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"Input arrays are not aligned in length: {lengths}.")
        if min(lengths.values()) == 0:
            raise ValueError(f"'{label}' contains an empty input array.")
        return data

    def _as_samples(self, name: str, values: Any, label: str) -> np.ndarray:
        """Check the feature count against dim and lay the input out as [samples, *dim]."""
        values = np.asarray(values, dtype=self.dtype)
        if values.ndim == 0:
            raise ValueError(f"Input '{name}' must be at least 1D, got scalar")

        dim = tuple(self.input_nodes[name].dim)
        feature_size = int(np.prod(values.shape[1:], dtype=int))
        expected_size = int(np.prod(dim, dtype=int))
        if feature_size != expected_size:
            raise ValueError(
                f"Input '{name}' in '{label}' provides {feature_size} values per "
                f"timestep, but its dim={dim} requires {expected_size}."
            )
        return np.reshape(values, (values.shape[0], *dim))

    # ------------------------------------------------------------------
    # Window: simulations -> {name: (N, *dim, time, *seq)}
    # ------------------------------------------------------------------

    def _build(
        self, simulations: List[tuple[str, Dict[str, np.ndarray]]]
    ) -> Dict[str, np.ndarray]:
        max_past = max(spec[0] for spec in self.input_specs.values())
        max_future = max(spec[1] for spec in self.input_specs.values())

        # Every input is windowed over the same range of end samples: it opens
        # once the deepest past window is covered and closes early enough to
        # leave room for the deepest future one.
        first = max(max_past - 1, 0)
        # A dynamic sequence is resolved per simulation, so only the levels
        # declared up front can be required of every simulation.
        fixed_spans = {
            name: sum(length - 1 for length in sequence if length is not None)
            for name, sequence in self.sequence_specs.items()
        }
        required = first + max_future + max(fixed_spans.values()) + 1
        # Every dynamic input shares one rollout axis, so its length is the one
        # the deepest of them can fill.
        dynamic_span = max(
            (fixed_spans[name] for name in self._dynamic_inputs), default=0
        )

        chunks: Dict[str, List[np.ndarray]] = {name: [] for name in self.input_specs}
        lengths: List[int] = []
        skipped = []
        # Where each simulation's samples sit once all of them are concatenated:
        # a time series only runs within one simulation, never across two.
        self._simulations: List[tuple[str, int, int]] = []
        offset = 0
        for label, simulation in simulations:
            samples = next(iter(simulation.values())).shape[0]
            if samples < required:
                if self.on_short == "skip":
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
                name: tuple(
                    length if declared is None else declared for declared in sequence
                )
                for name, sequence in self.sequence_specs.items()
            }
            spans = {
                name: sum(size - 1 for size in sequence)
                for name, sequence in sequences.items()
            }
            max_span = max(spans.values())
            for name, values in simulation.items():
                chunks[name].append(
                    self._windows(
                        name,
                        values,
                        first,
                        max_future,
                        max_span - spans[name],
                        sequences[name],
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
        # different lengths only line up once the rollout axis is padded.
        if self._dynamic_inputs and len(set(lengths)) > 1:
            widest = max(lengths)
            for name in self._dynamic_inputs:
                chunks[name] = [self._pad(values, widest) for values in chunks[name]]
            self.mask = np.arange(widest) < np.asarray(lengths)[:, np.newaxis]
            self.padded_inputs = set(self._dynamic_inputs)

        return {
            name: np.concatenate(windows, axis=0) for name, windows in chunks.items()
        }

    def _warn_uncollected_loops(self) -> None:
        """Warn about a Loop that returns only its last, padded, rollout step."""
        uncollected = sorted(
            node.name
            for node in self.model.order
            if getattr(node, "collect", None) is False
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
        """Extend the rollout axis to ``width`` by repeating its last step.

        Repeating keeps the padded part of a rollout inside the range the model
        was fitted on; the mask is what removes it from the loss.
        """
        missing = width - values.shape[-1]
        if missing <= 0:
            return values
        return np.concatenate(
            [values, np.repeat(values[..., -1:], missing, axis=-1)], axis=-1
        )

    def _mask_view(self, rank: int) -> np.ndarray:
        """The rollout mask shaped to broadcast against a dataset array."""
        assert self.mask is not None
        return self.mask.reshape(
            self.mask.shape[0], *([1] * (rank - 2)), self.mask.shape[1]
        )

    def _windows(
        self,
        name: str,
        values: np.ndarray,
        first: int,
        max_future: int,
        lead: int,
        sequence: tuple[int, ...],
    ) -> np.ndarray:
        """Temporal windows of one input, then one sliding window per seq length."""
        past, future = self.input_specs[name]
        width = past + future
        last = values.shape[0] - max_future  # one past the last aligned end sample

        if width > 0:
            # Window i covers values[i:i + width] and ends on sample i + past - 1,
            # so the shared range of end samples selects the windows to keep.
            windows = sliding_window_view(values, window_shape=width, axis=0)
            windows = windows[first - past + 1 : last - past + 1]
        else:
            # No declared window: the input contributes a single sample.
            windows = values[first:last][..., np.newaxis]

        for length in sequence:
            windows = sliding_window_view(windows, window_shape=length, axis=0)

        # A sequence window is aligned to its final temporal sample. Inputs with
        # shorter/no seq dimensions therefore skip earlier samples so every model
        # input refers to the same endpoint.
        if lead:
            windows = windows[lead:]
        if self.step > 1:
            windows = windows[:: self.step]
        return np.ascontiguousarray(windows, dtype=self.dtype)

    # ------------------------------------------------------------------
    # Explicit normalization
    # ------------------------------------------------------------------

    def normalize(
        self,
        method: Literal["minmax", "standard"] = "minmax",
        names: list[str] | tuple[str, ...] | None = None,
        feature_range: tuple[float, float] = (-1.0, 1.0),
    ) -> "DataLoader":
        """Fit normalization statistics and transform this loader in place.

        Normalization is explicit and local to this DataLoader. It does not
        modify Modely inference or any other loader created from the same data.
        Statistics are fitted independently for every input feature while
        reducing over samples, time, and sequence axes.
        """
        if method not in ("minmax", "standard"):
            raise ValueError("method must be either 'minmax' or 'standard'.")
        if not np.issubdtype(np.dtype(self.dtype), np.floating):
            raise TypeError("DataLoader normalization requires a floating-point dtype.")
        low, high = (float(feature_range[0]), float(feature_range[1]))
        if method == "minmax" and not low < high:
            raise ValueError("feature_range must satisfy low < high.")

        selected = list(self.dataset) if names is None else list(names)
        unknown = [name for name in selected if name not in self.dataset]
        if unknown:
            raise ValueError(f"Unknown dataset inputs for normalization: {unknown}.")

        if self._original_dataset is None:
            self._original_dataset = {
                name: np.array(values, copy=True)
                for name, values in self.dataset.items()
            }
        source = self._original_dataset

        # Reapplying normalization always starts from the original prepared
        # windows, so transformations never compound.
        self.dataset = {
            name: np.array(values, copy=True) for name, values in source.items()
        }
        self.normalization_stats = {}

        for name in selected:
            values = source[name].astype(np.float64, copy=False)
            dim_rank = self.input_nodes[name].shape.dim_rank
            reduce_axes = (0, *range(1 + dim_rank, values.ndim))

            # Padded rollout steps are repeats, not observations, so they are
            # left out of the statistics while still being transformed.
            padded = name in self.padded_inputs
            observed = (
                np.where(self._mask_view(values.ndim), values, np.nan)
                if padded
                else values
            )
            mean, std = (np.nanmean, np.nanstd) if padded else (np.mean, np.std)
            smallest, largest = (np.nanmin, np.nanmax) if padded else (np.min, np.max)

            if method == "standard":
                offset = mean(observed, axis=reduce_axes, keepdims=True)
                scale = std(observed, axis=reduce_axes, keepdims=True)
                constant = scale <= np.finfo(np.float32).eps
                safe_scale = np.where(constant, 1.0, scale)
                normalized = (values - offset) / safe_scale
            else:
                minimum = smallest(observed, axis=reduce_axes, keepdims=True)
                maximum = largest(observed, axis=reduce_axes, keepdims=True)
                span = maximum - minimum
                constant = span <= np.finfo(np.float32).eps
                safe_scale = np.where(constant, 1.0, span)
                offset = minimum
                scale = safe_scale
                normalized = (values - offset) / scale
                normalized = normalized * (high - low) + low
                normalized = np.where(constant, (low + high) / 2.0, normalized)

            self.normalization_stats[name] = {
                "method": method,
                "offset": offset,
                "scale": scale,
                "constant": constant,
                "feature_range": (low, high),
            }
            self.dataset[name] = normalized.astype(self.dtype)

        return self

    def denormalize(
        self,
        data: Dict[str, Any] | np.ndarray | None = None,
        *,
        name: str | None = None,
    ):
        """Undo this loader's fitted normalization.

        With no data, restore the loader's original prepared dataset in place
        and return ``self``. A dictionary or array is inverse-transformed and
        returned without modifying the loader. For an array, ``name`` selects
        the statistics to use.
        """
        if self._original_dataset is None:
            if data is None:
                return self
            raise ValueError("normalize() must be called before denormalize().")

        if data is None:
            self.dataset = {
                key: np.array(values, copy=True)
                for key, values in self._original_dataset.items()
            }
            return self

        if isinstance(data, dict):
            return {
                key: self._denormalize_values(key, values)
                if self._normalization_name(key) is not None
                else np.asarray(values)
                for key, values in data.items()
            }

        if name is None:
            raise ValueError("name is required when denormalizing an array.")
        return self._denormalize_values(name, data)

    def _denormalize_values(self, name: str, values: Any) -> np.ndarray:
        stats_name = self._normalization_name(name)
        if stats_name is None:
            raise ValueError(f"No normalization statistics are available for {name!r}.")
        stats = self.normalization_stats[stats_name]
        values = np.asarray(values, dtype=np.float64)
        offset = self._match_stat_rank(stats["offset"], values.ndim)
        scale = self._match_stat_rank(stats["scale"], values.ndim)
        constant = self._match_stat_rank(stats["constant"], values.ndim)

        if stats["method"] == "minmax":
            low, high = stats["feature_range"]
            values = (values - low) / (high - low)
        restored = values * scale + offset
        restored = np.where(constant, offset, restored)
        return restored.astype(self.dtype)

    @staticmethod
    def _match_stat_rank(stat: np.ndarray, rank: int) -> np.ndarray:
        while stat.ndim > rank and stat.shape[0] == 1:
            stat = stat[0]
        if stat.ndim != rank:
            raise ValueError(
                f"Data rank {rank} is incompatible with normalization rank {stat.ndim}."
            )
        return stat

    def _normalization_name(self, name: str) -> str | None:
        if name in self.normalization_stats:
            return name
        alias = self._normalization_aliases.get(name)
        return alias if alias in self.normalization_stats else None

    def _build_normalization_aliases(self) -> Dict[str, str]:
        aliases = {}

        def find_input_name(node):
            if node.name in self.input_nodes:
                return node.name
            preds = getattr(node, "preds", [])
            if len(preds) == 1:
                return find_input_name(preds[0])
            return None

        for minimizer in self.model.minimizers:
            target_name = find_input_name(minimizer["target"])
            if target_name is not None:
                aliases[minimizer["source"].name] = target_name
        return aliases
