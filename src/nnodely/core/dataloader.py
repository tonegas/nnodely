from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal

import numpy as np
import pandas as pd


@dataclass
class DataLoader:
    """
    Build a dataset from CSV files using the inputs declared in the model.

    Rules:
    - Every model input is treated the same way.
    - The input name must match a CSV column name.
    - The input sample window is inferred from the input object:
        Input("data_1").sw(5)  -> windows of length 5
        Input("data_2").sw(1)  -> windows of length 1
    - All windows are aligned in time.
    - Multiple CSV files are concatenated sample-wise.

    Final dataset format:
        {
            "data_1": np.ndarray of shape (N, 5),
            "data_2": np.ndarray of shape (N, 1),
            ...
        }

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

    dataset: Dict[str, np.ndarray] = field(default_factory=dict)
    align: Literal["trim", "error"] = "trim"

    def __init__(
        self,
        model: Any,
        source: str | dict,
        format: dict[str, str | int] | None = None,
        trim: bool = False,
        csv_glob: str = "*.csv",
        dtype: Any = np.float32,
        seq_length: int | None = None,
    ):
        self.model = model
        self.format = format
        self.trim = trim
        self.csv_glob = csv_glob
        self.dtype = dtype
        self.seq_length = seq_length

        if model.model is None:
            raise ValueError(
                f"Model {model.name} is not built. Make sure to call {model.name}.build() first."
            )
        self.input_specs = {
            node.name: [node.past, node.future] for node in model.train_inputs
        }
        self.input_nodes = {node.name: node for node in model.train_inputs}
        if not self.input_specs:
            raise ValueError("Could not infer any inputs from model.inputs")

        self.sequence_specs = {}
        for node in model.train_inputs:
            sequence = []
            for length in node.seq:
                if length is None:
                    if self.seq_length is None:
                        raise ValueError(
                            "Some inputs have undefined sequence length. "
                            "Please specify seq_length in training."
                        )
                    length = self.seq_length
                if length < 1:
                    raise ValueError(
                        f"Input '{node.name}' has invalid sequence length {length}."
                    )
                sequence.append(int(length))
            self.sequence_specs[node.name] = tuple(sequence)

        sequence_lengths = [
            length for sequence in self.sequence_specs.values() for length in sequence
        ]
        self.max_sequence_length = max(sequence_lengths, default=0)

        if isinstance(source, str):
            self.source = Path(source)
            if not self.source.exists():
                raise FileNotFoundError(f"Source does not exist: {self.source}")
            if not self.source.is_dir():
                raise NotADirectoryError(f"Not a folder: {self.source}")
            if self.format is not None and not isinstance(self.format, dict):
                raise TypeError(
                    "format must be a dict mapping input name to column name or index"
                )
            self.dataset = self._build_from_folder()
        elif isinstance(source, dict):
            self.dataset = self._build_from_dict(source)

        self.normalization_stats: Dict[str, Dict[str, Any]] = {}
        self._original_dataset: Dict[str, np.ndarray] | None = None
        self._normalization_aliases = self._build_normalization_aliases()
        self._num_steps = self._infer_num_steps()

    @property
    def inputs(self) -> List[str]:
        return list(self.dataset.keys())

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

    def get_train_data(self, batch_size: int) -> Iterator[Dict[str, Any]]:
        for i in range(0, self._num_steps, batch_size):
            batch = {k: v[i : i + batch_size] for k, v in self.dataset.items()}
            yield batch

    def as_dict(self) -> Dict[str, np.ndarray]:
        return self.dataset

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

            if method == "standard":
                offset = np.mean(values, axis=reduce_axes, keepdims=True)
                scale = np.std(values, axis=reduce_axes, keepdims=True)
                constant = scale <= np.finfo(np.float32).eps
                safe_scale = np.where(constant, 1.0, scale)
                normalized = (values - offset) / safe_scale
            else:
                minimum = np.min(values, axis=reduce_axes, keepdims=True)
                maximum = np.max(values, axis=reduce_axes, keepdims=True)
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

    # ------------------------------------------------------------------
    # Build dataset
    # ------------------------------------------------------------------

    def _build_from_dict(self, source: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Build dataset from an in-memory dict.

        Expected input:
            {
                "x": np.ndarray,
                "y": np.ndarray,
                ...
            }

        Each value can be:
        - list
        - numpy array
        - pandas Series
        - pandas DataFrame (single column or multi-column)

        Rolling windows are built exactly like in _build_from_dataframe(),
        using self.input_specs[name] as the required window length.

        Output format:
            {
                "x": np.ndarray of shape (N, W_x, ...),
                "y": np.ndarray of shape (N, W_y, ...),
                ...
            }
        """
        missing = [name for name in self.input_specs if name not in source]
        if missing:
            raise ValueError(f"Source dict is missing required inputs: {missing}")

        # Normalize all provided arrays
        arrays: Dict[str, np.ndarray] = {}
        for name in self.input_specs:
            value = source[name]

            if isinstance(value, pd.Series):
                arr = value.to_numpy(dtype=self.dtype)
            elif isinstance(value, pd.DataFrame):
                arr = value.to_numpy(dtype=self.dtype)
            else:
                arr = np.asarray(value, dtype=self.dtype)

            if arr.ndim == 0:
                raise ValueError(f"Input '{name}' must be at least 1D, got scalar")

            # Normalize shape:
            #   (T,)       -> scalar feature over time
            #   (T, D)     -> D features over time
            #   (T, ...)   -> generic trailing feature dims
            arrays[name] = arr

        lengths = {name: arr.shape[0] for name, arr in arrays.items()}
        if not lengths:
            return {}

        min_rows = min(lengths.values())

        if min_rows == 0:
            raise ValueError("At least one provided input array is empty.")

        if len(set(lengths.values())) > 1:
            if self.trim:
                for name in arrays:
                    arrays[name] = arrays[name][:min_rows]
            else:
                raise ValueError(
                    f"Input arrays are not aligned in length: {lengths}. "
                    f"Use trim=True to align automatically."
                )

        n_rows = next(iter(arrays.values())).shape[0]
        max_past_window = max(spec[0] for spec in self.input_specs.values())
        max_future_window = max(spec[1] for spec in self.input_specs.values())
        max_window = max_past_window + max_future_window

        if n_rows < max_window:
            raise ValueError(
                f"Input arrays have only {n_rows} rows, but the largest required window is {max_window}."
            )

        t_start = max_past_window - 1 if max_past_window > 0 else 0
        t_end = n_rows - max_future_window - 1

        raw_data: Dict[str, List[np.ndarray]] = {name: [] for name in self.input_specs}

        for t in range(t_start, t_end + 1):
            for name, (past, future) in self.input_specs.items():
                start = t - past + 1
                end = t + future + 1

                values = (
                    arrays[name][start:end]
                    if past + future > 0
                    else arrays[name][t : t + 1]
                )

                if values.shape[0] != past + future and past + future > 0:
                    raise RuntimeError(
                        f"Internal error while building window for '{name}': "
                        f"expected {past + future}, got {values.shape[0]}"
                    )

                raw_data[name].append(self._format_temporal_window(name, values))

        dataset = {
            name: np.stack(windows, axis=0).astype(self.dtype)
            for name, windows in raw_data.items()
        }
        dataset = self._apply_sequence_windows(dataset)
        self._check_alignment(dataset)
        return dataset

    def _build_from_folder(self) -> Dict[str, np.ndarray]:
        csv_files = sorted(self.source.glob(self.csv_glob))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files matching '{self.csv_glob}' found in {self.source}"
            )
        chunks: Dict[str, List[np.ndarray]] = {name: [] for name in self.input_specs}
        for csv_path in csv_files:
            df = pd.read_csv(csv_path)
            missing = []
            for name in self.input_specs:
                col = (
                    self.format[name] if (self.format and name in self.format) else name
                )
                if isinstance(col, int):
                    if col < 0 or col >= df.shape[1]:
                        missing.append(f"{name} -> index {col}")
                else:
                    if col not in df.columns:
                        missing.append(f"{name} -> '{col}'")
            if missing:
                raise ValueError(
                    f"File '{csv_path.name}' is missing required columns or indices: {missing}"
                )

            file_dataset = self._build_from_dataframe(df)
            for name, arr in file_dataset.items():
                chunks[name].append(arr)

        dataset = {}

        for name, arr_list in chunks.items():
            arr = np.concatenate(arr_list, axis=0).astype(self.dtype)
            dataset[name] = arr
        return dataset

    def _build_from_dataframe(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        For each input with window W, build rolling windows:

            column = [1,2,3,4,5,6]
            W = 3

            -> [[1,2,3],
                [2,3,4],
                [3,4,5],
                [4,5,6]]

        All inputs are aligned by trimming to the common valid range.
        """
        n_rows = len(df)
        if n_rows == 0:
            raise ValueError("Encountered an empty CSV file.")

        max_past_window = max(spec[0] for spec in self.input_specs.values())
        max_future_window = max(spec[1] for spec in self.input_specs.values())
        max_window = max_past_window + max_future_window
        if n_rows < max_window:
            raise ValueError(
                f"CSV has only {n_rows} rows, but the largest required window is {max_window}."
            )

        # Common aligned sample end indices
        # If max_window = 5, valid end indices are 4,5,6,...
        t_start = max_past_window - 1 if max_past_window > 0 else 0
        t_end = n_rows - max_future_window - 1

        raw_data: Dict[str, List[np.ndarray]] = {name: [] for name in self.input_specs}
        for t in range(t_start, t_end + 1):
            for name, (past, future) in self.input_specs.items():
                start = t - past + 1
                end = t + future + 1
                col = (
                    self.format[name] if (self.format and name in self.format) else name
                )
                if isinstance(col, int):
                    try:
                        values = (
                            df.iloc[start:end, col].to_numpy(dtype=self.dtype)
                            if past + future > 0
                            else np.array([df.iloc[t, col]], dtype=self.dtype)
                        )
                    except IndexError:
                        raise ValueError(
                            f"Column index {col} out of range for input '{name}'"
                        )
                else:
                    values = (
                        df[col].iloc[start:end].to_numpy(dtype=self.dtype)
                        if past + future > 0
                        else np.array([df[col].iloc[t]], dtype=self.dtype)
                    )

                if past + future > 0 and values.shape[0] != past + future:
                    raise RuntimeError(
                        f"Internal error while building window for '{name}': "
                        f"expected {past + future}, got {values.shape[0]}"
                    )
                raw_data[name].append(self._format_temporal_window(name, values))

        dataset = {
            name: np.stack(windows, axis=0) for name, windows in raw_data.items()
        }
        dataset = self._apply_sequence_windows(dataset)
        self._check_alignment(dataset)
        return dataset

    def _format_temporal_window(self, name: str, values: np.ndarray) -> np.ndarray:
        """Convert raw [time, features...] data to nnodely [dim..., time]."""
        values = np.asarray(values, dtype=self.dtype)
        window_length = values.shape[0]
        dim = tuple(self.input_nodes[name].dim)
        feature_size = int(np.prod(values.shape[1:], dtype=int))
        expected_size = int(np.prod(dim, dtype=int))

        if feature_size != expected_size:
            raise ValueError(
                f"Input '{name}' provides {feature_size} values per timestep, "
                f"but its dim={dim} requires {expected_size}."
            )

        values = np.reshape(values, (window_length, *dim))
        return np.moveaxis(values, 0, -1)

    def _apply_sequence_windows(
        self, dataset: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Build every seq axis on top of the already-created time windows."""
        spans = {
            name: sum(length - 1 for length in sequence)
            for name, sequence in self.sequence_specs.items()
        }
        max_span = max(spans.values(), default=0)
        result = {}

        for name, values in dataset.items():
            required = spans[name] + 1
            if len(values) < required:
                raise ValueError(
                    f"Input '{name}' has only {len(values)} temporal windows, "
                    f"but seq={self.sequence_specs[name]} requires at least {required}."
                )

            windows = values
            for length in self.sequence_specs[name]:
                windows = np.lib.stride_tricks.sliding_window_view(
                    windows,
                    window_shape=length,
                    axis=0,
                )

            # A sequence window is aligned to its final temporal sample. Inputs
            # with shorter/no seq dimensions therefore skip earlier samples so
            # every model input refers to the same endpoint.
            offset = max_span - spans[name]
            if offset:
                windows = windows[offset:]
            result[name] = np.ascontiguousarray(windows, dtype=self.dtype)

        return result

    def _check_alignment(self, dataset: Dict[str, np.ndarray]) -> None:
        lengths = {k: len(v) for k, v in dataset.items()}
        unique_lengths = set(lengths.values())

        if len(unique_lengths) <= 1:
            return

        if self.trim:
            min_len = min(lengths.values())
            for k in dataset:
                dataset[k] = dataset[k][:min_len]

    def _infer_num_steps(self) -> int:
        if not self.dataset:
            return 0
        return min(len(v) for v in self.dataset.values())
