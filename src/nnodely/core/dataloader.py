from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal

import numpy as np
import pandas as pd

import keras


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
        format: Dict[str, str | int] | None = None,
        trim: bool = False,
        csv_glob: str = "*.csv",
        dtype: Any = np.float32,
    ):
        self.model = model
        self.format = format
        self.trim = trim
        self.csv_glob = csv_glob
        self.dtype = dtype

        if model._train_model is None:
            raise ValueError(
                f"Model {model.name} is not built. Make sure to call {model.name}.build() first."
            )
        self.input_specs = {
            node.name: [node.past, node.future] for node in model.train_inputs
        }
        if not self.input_specs:
            raise ValueError("Could not infer any inputs from model.inputs")

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

    # def get_step(self, idx: int) -> Dict[str, Any]:
    #     if idx < 0:
    #         idx = self._num_steps + idx
    #     if idx < 0 or idx >= self._num_steps:
    #         raise IndexError(f"idx out of range: {idx} (len={self._num_steps})")
    #     return {k: v[idx] for k, v in self.dataset.items()}

    def get_step(self, idx: int) -> Dict[str, Any]:
        if idx < 0:
            idx = self._num_steps + idx

        if idx < 0 or idx >= self._num_steps:
            raise IndexError(f"idx out of range: {idx} (len={self._num_steps})")

        return {
            k: keras.ops.convert_to_tensor(v[idx][None, ...])
            for k, v in self.dataset.items()
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.get_step(idx)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for i in range(self._num_steps):
            yield self.get_step(i)

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

        t_start = max_window - 1
        t_end = n_rows - 1

        raw_data: Dict[str, List[np.ndarray]] = {name: [] for name in self.input_specs}

        for t in range(t_start, t_end + 1):
            for name, (past, future) in self.input_specs.items():
                start = t - past + 1
                end = t + future + 1

                values = arrays[name][start:end]

                if values.shape[0] != past + future:
                    raise RuntimeError(
                        f"Internal error while building window for '{name}': "
                        f"expected {past + future}, got {values.shape[0]}"
                    )

                raw_data[name].append(values)

        dataset = {
            name: np.stack(windows, axis=0).astype(self.dtype)
            for name, windows in raw_data.items()
        }
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
            time_size = sum(self.input_specs[name])

            if not arr_list:
                dataset[name] = np.empty(
                    (0, 1, time_size),
                    dtype=self.dtype,
                )
            else:
                arr = np.concatenate(arr_list, axis=0).astype(self.dtype)

                # Old shape:
                #   (N, time)
                #
                # New shape:
                #   (N, 1, time)
                if arr.ndim == 2:
                    arr = arr[:, None, :]

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
        t_start = max_past_window - 1
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
                        values = df.iloc[start:end, col].to_numpy(dtype=self.dtype)
                    except IndexError:
                        raise ValueError(
                            f"Column index {col} out of range for input '{name}'"
                        )
                else:
                    values = df[col].iloc[start:end].to_numpy(dtype=self.dtype)

                if values.shape[0] != past + future:
                    raise RuntimeError(
                        f"Internal error while building window for '{name}': "
                        f"expected {past + future}, got {values.shape[0]}"
                    )

                raw_data[name].append(values)

        dataset = {
            name: np.stack(windows, axis=0) for name, windows in raw_data.items()
        }
        self._check_alignment(dataset)
        return dataset

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


# from __future__ import annotations

# from dataclasses import dataclass, field
# from pathlib import Path
# from typing import Any, Dict, Iterator, List, Literal

# import keras
# import numpy as np
# import pandas as pd


# @dataclass
# class DataLoader:
#     dataset: Dict[str, np.ndarray] = field(default_factory=dict)
#     align: Literal["trim", "error"] = "trim"

#     def __init__(
#         self,
#         model: Any,
#         source: str | dict,
#         format: Dict[str, str | int] | None = None,
#         trim: bool = False,
#         csv_glob: str = "*.csv",
#         dtype: Any = np.float32,
#     ):
#         self.model = model
#         self.source = source
#         self.format = format
#         self.trim = trim
#         self.csv_glob = csv_glob
#         self.dtype = dtype

#         train_model = getattr(model, "_train_model", None) or getattr(model, "model", None)
#         if train_model is None:
#             raise ValueError(
#                 f"Model {model.name} is not built. Make sure to call {model.name}.build() first."
#             )

#         train_inputs = getattr(model, "train_inputs", None) or getattr(model, "inputs", None)
#         if not train_inputs:
#             raise ValueError("Could not infer any inputs from model.inputs")

#         self.input_nodes = {node.name: node for node in train_inputs}

#         # Each spec contains symbolic shape information.
#         # Expected node attributes:
#         # - dim
#         # - time OR past/future
#         # - seq
#         self.input_specs = {
#             node.name: {
#                 "dim": tuple(getattr(node, "dim", (1,))),
#                 "seq": tuple(getattr(node, "seq", ())),
#                 "past": int(getattr(node, "past", getattr(node, "time", 1))),
#                 "future": int(getattr(node, "future", 0)),
#             }
#             for node in train_inputs
#         }

#         if isinstance(source, str):
#             self.source = Path(source)
#             if not self.source.exists():
#                 raise FileNotFoundError(f"Source does not exist: {self.source}")
#             if not self.source.is_dir():
#                 raise NotADirectoryError(f"Not a folder: {self.source}")
#             if self.format is not None and not isinstance(self.format, dict):
#                 raise TypeError("format must be a dict mapping input name to column name or index")
#             self.dataset = self._build_from_folder()

#         elif isinstance(source, dict):
#             self.dataset = self._build_from_dict(source)

#         else:
#             raise TypeError("source must be either a folder path or a dictionary")

#         self._num_steps = self._infer_num_steps()

#     @property
#     def inputs(self) -> List[str]:
#         return list(self.dataset.keys())

#     def __len__(self) -> int:
#         return self._num_steps

#     def get_input(self, name: str) -> np.ndarray:
#         return self.dataset[name]

#     def get_step(self, idx: int) -> Dict[str, Any]:
#         if idx < 0:
#             idx = self._num_steps + idx

#         if idx < 0 or idx >= self._num_steps:
#             raise IndexError(f"idx out of range: {idx} (len={self._num_steps})")

#         return {
#             k: keras.ops.convert_to_tensor(v[idx][None, ...])
#             for k, v in self.dataset.items()
#         }

#     def __getitem__(self, idx: int) -> Dict[str, Any]:
#         return self.get_step(idx)

#     def __iter__(self) -> Iterator[Dict[str, Any]]:
#         for i in range(self._num_steps):
#             yield self.get_step(i)

#     def as_dict(self) -> Dict[str, np.ndarray]:
#         return self.dataset

#     # ------------------------------------------------------------------
#     # Build dataset
#     # ------------------------------------------------------------------

#     def _build_from_dict(self, source: Dict[str, Any]) -> Dict[str, np.ndarray]:
#         missing = [name for name in self.input_specs if name not in source]
#         if missing:
#             raise ValueError(f"Source dict is missing required inputs: {missing}")

#         arrays: Dict[str, np.ndarray] = {}

#         for name in self.input_specs:
#             value = source[name]

#             if isinstance(value, pd.Series):
#                 arr = value.to_numpy(dtype=self.dtype)
#             elif isinstance(value, pd.DataFrame):
#                 arr = value.to_numpy(dtype=self.dtype)
#             else:
#                 arr = np.asarray(value, dtype=self.dtype)

#             if arr.ndim == 0:
#                 raise ValueError(f"Input '{name}' must be at least 1D, got scalar.")

#             arrays[name] = arr

#         lengths = {name: arr.shape[0] for name, arr in arrays.items()}
#         min_rows = min(lengths.values())

#         if min_rows == 0:
#             raise ValueError("At least one provided input array is empty.")

#         if len(set(lengths.values())) > 1:
#             if self.trim:
#                 for name in arrays:
#                     arrays[name] = arrays[name][:min_rows]
#             else:
#                 raise ValueError(
#                     f"Input arrays are not aligned in length: {lengths}. "
#                     f"Use trim=True to align automatically."
#                 )

#         n_rows = next(iter(arrays.values())).shape[0]
#         self._check_enough_rows(n_rows)

#         raw_data: Dict[str, List[np.ndarray]] = {name: [] for name in self.input_specs}

#         t_start, t_end = self._aligned_time_bounds(n_rows)

#         for t in range(t_start, t_end + 1):
#             for name, arr in arrays.items():
#                 spec = self.input_specs[name]
#                 values = self._slice_raw_array(arr, t, spec)
#                 values = self._to_model_sample(values, spec)
#                 raw_data[name].append(values)

#         dataset = {
#             name: np.stack(windows, axis=0).astype(self.dtype)
#             for name, windows in raw_data.items()
#         }

#         self._check_alignment(dataset)
#         return dataset

#     def _build_from_folder(self) -> Dict[str, np.ndarray]:
#         csv_files = sorted(self.source.glob(self.csv_glob))
#         if not csv_files:
#             raise FileNotFoundError(
#                 f"No CSV files matching '{self.csv_glob}' found in {self.source}"
#             )

#         chunks: Dict[str, List[np.ndarray]] = {name: [] for name in self.input_specs}

#         for csv_path in csv_files:
#             df = pd.read_csv(csv_path)
#             self._check_dataframe_columns(df, csv_path.name)

#             file_dataset = self._build_from_dataframe(df)
#             for name, arr in file_dataset.items():
#                 chunks[name].append(arr)

#         dataset = {}
#         for name, arr_list in chunks.items():
#             if not arr_list:
#                 spec = self.input_specs[name]
#                 sample_shape = self._sample_shape(spec)
#                 dataset[name] = np.empty((0,) + sample_shape, dtype=self.dtype)
#             else:
#                 dataset[name] = np.concatenate(arr_list, axis=0).astype(self.dtype)

#         return dataset

#     def _build_from_dataframe(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
#         n_rows = len(df)
#         if n_rows == 0:
#             raise ValueError("Encountered an empty CSV file.")

#         self._check_enough_rows(n_rows)

#         raw_data: Dict[str, List[np.ndarray]] = {name: [] for name in self.input_specs}

#         t_start, t_end = self._aligned_time_bounds(n_rows)

#         for t in range(t_start, t_end + 1):
#             for name, spec in self.input_specs.items():
#                 values = self._slice_dataframe_column(df, name, t, spec)
#                 values = self._to_model_sample(values, spec)
#                 raw_data[name].append(values)

#         dataset = {
#             name: np.stack(windows, axis=0).astype(self.dtype)
#             for name, windows in raw_data.items()
#         }

#         self._check_alignment(dataset)
#         return dataset

#     # ------------------------------------------------------------------
#     # Shape conversion
#     # ------------------------------------------------------------------

#     def _to_model_sample(self, values: np.ndarray, spec: dict) -> np.ndarray:
#         """
#         Convert raw window values to framework convention without batch:

#             raw values: [time, dim...]
#             model sample: [dim..., time, seq...]

#         Examples:
#             values shape (W,)       -> (1, W)
#             values shape (W, D)     -> (D, W)
#             values shape (W, D1,D2) -> (D1, D2, W)

#         If seq is present, append seq axes at the end.
#         """
#         values = np.asarray(values, dtype=self.dtype)

#         if values.ndim == 1:
#             # [time] -> [time, 1]
#             values = values[:, None]

#         expected_time = spec["past"] + spec["future"]
#         if values.shape[0] != expected_time:
#             raise RuntimeError(
#                 f"Expected time window length {expected_time}, got {values.shape[0]}"
#             )

#         # [time, dim...] -> [dim..., time]
#         axes = list(range(1, values.ndim)) + [0]
#         sample = np.transpose(values, axes)

#         expected_dim = tuple(spec["dim"])
#         if sample.shape[: len(expected_dim)] != expected_dim:
#             try:
#                 sample = np.reshape(sample, expected_dim + (expected_time,))
#             except Exception as e:
#                 raise ValueError(
#                     f"Cannot reshape sample with shape {sample.shape} to "
#                     f"dim/time shape {expected_dim + (expected_time,)}"
#                 ) from e

#         seq = tuple(spec["seq"])
#         if seq:
#             sample = np.reshape(sample, expected_dim + (expected_time,) + seq)

#         return sample.astype(self.dtype)

#     def _sample_shape(self, spec: dict) -> tuple:
#         return tuple(spec["dim"]) + (spec["past"] + spec["future"],) + tuple(spec["seq"])

#     # ------------------------------------------------------------------
#     # Time slicing
#     # ------------------------------------------------------------------

#     def _aligned_time_bounds(self, n_rows: int) -> tuple[int, int]:
#         max_past = max(spec["past"] for spec in self.input_specs.values())
#         max_future = max(spec["future"] for spec in self.input_specs.values())

#         t_start = max_past - 1
#         t_end = n_rows - max_future - 1
#         return t_start, t_end

#     def _check_enough_rows(self, n_rows: int) -> None:
#         max_past = max(spec["past"] for spec in self.input_specs.values())
#         max_future = max(spec["future"] for spec in self.input_specs.values())
#         required = max_past + max_future

#         if n_rows < required:
#             raise ValueError(
#                 f"Data has only {n_rows} rows, but the largest required window is {required}."
#             )

#     def _slice_raw_array(self, arr: np.ndarray, t: int, spec: dict) -> np.ndarray:
#         past = spec["past"]
#         future = spec["future"]

#         start = t - past + 1
#         end = t + future + 1

#         values = arr[start:end]

#         expected = past + future
#         if values.shape[0] != expected:
#             raise RuntimeError(
#                 f"Internal error while building window: expected {expected}, got {values.shape[0]}"
#             )

#         return values

#     def _slice_dataframe_column(
#         self,
#         df: pd.DataFrame,
#         name: str,
#         t: int,
#         spec: dict,
#     ) -> np.ndarray:
#         past = spec["past"]
#         future = spec["future"]

#         start = t - past + 1
#         end = t + future + 1

#         col = self.format[name] if (self.format and name in self.format) else name

#         if isinstance(col, int):
#             if col < 0 or col >= df.shape[1]:
#                 raise ValueError(f"Column index {col} out of range for input '{name}'")
#             values = df.iloc[start:end, col].to_numpy(dtype=self.dtype)
#         else:
#             values = df[col].iloc[start:end].to_numpy(dtype=self.dtype)

#         expected = past + future
#         if values.shape[0] != expected:
#             raise RuntimeError(
#                 f"Internal error while building window for '{name}': "
#                 f"expected {expected}, got {values.shape[0]}"
#             )

#         return values

#     # ------------------------------------------------------------------
#     # Validation
#     # ------------------------------------------------------------------

#     def _check_dataframe_columns(self, df: pd.DataFrame, csv_name: str) -> None:
#         missing = []

#         for name in self.input_specs:
#             col = self.format[name] if (self.format and name in self.format) else name

#             if isinstance(col, int):
#                 if col < 0 or col >= df.shape[1]:
#                     missing.append(f"{name} -> index {col}")
#             else:
#                 if col not in df.columns:
#                     missing.append(f"{name} -> '{col}'")

#         if missing:
#             raise ValueError(
#                 f"File '{csv_name}' is missing required columns or indices: {missing}"
#             )

#     def _check_alignment(self, dataset: Dict[str, np.ndarray]) -> None:
#         lengths = {k: len(v) for k, v in dataset.items()}
#         unique_lengths = set(lengths.values())

#         if len(unique_lengths) <= 1:
#             return

#         if self.trim:
#             min_len = min(lengths.values())
#             for k in dataset:
#                 dataset[k] = dataset[k][:min_len]
#             return

#         raise ValueError(f"Dataset arrays are not aligned: {lengths}")

#     def _infer_num_steps(self) -> int:
#         if not self.dataset:
#             return 0
#         return min(len(v) for v in self.dataset.values())
