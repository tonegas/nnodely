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
