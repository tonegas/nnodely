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
        folder: str,
        format: Dict[str, str|int] | None = None,
        trim: bool = False,
        csv_glob: str = "*.csv",
        dtype: Any = np.float32,
    ):
        self.model = model
        self.folder = Path(folder)
        self.format = format
        if self.format is not None and not isinstance(self.format, dict):
            raise TypeError("format must be a dict mapping input name to column name or index")
        self.trim = trim
        self.csv_glob = csv_glob
        self.dtype = dtype

        if not self.folder.exists():
            raise FileNotFoundError(f"Folder does not exist: {self.folder}")
        if not self.folder.is_dir():
            raise NotADirectoryError(f"Not a folder: {self.folder}")

        self.input_specs = self._extract_input_specs(model)
        if not self.input_specs:
            raise ValueError("Could not infer any inputs from model.inputs")

        for minimizer in model._minimizers: ## TODO: Finish the data loader 
            if minimizer[''] in self.input_specs:
                raise ValueError(f"Input '{minimizer['source']}' is used as a minimizer source, which is not supported.")
        self.dataset = self._build_from_folder()
        self._num_steps = self._infer_num_steps()

    @property
    def inputs(self) -> List[str]:
        return list(self.dataset.keys())

    def __len__(self) -> int:
        return self._num_steps

    def get_input(self, name: str) -> np.ndarray:
        return self.dataset[name]

    def get_step(self, idx: int) -> Dict[str, Any]:
        if idx < 0:
            idx = self._num_steps + idx
        if idx < 0 or idx >= self._num_steps:
            raise IndexError(f"idx out of range: {idx} (len={self._num_steps})")
        return {k: v[idx] for k, v in self.dataset.items()}

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

    def _build_from_folder(self) -> Dict[str, np.ndarray]:
        csv_files = sorted(self.folder.glob(self.csv_glob))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files matching '{self.csv_glob}' found in {self.folder}"
            )
        chunks: Dict[str, List[np.ndarray]] = {name: [] for name in self.input_specs}
        for csv_path in csv_files:
            df = pd.read_csv(csv_path)
            missing = []
            for name in self.input_specs:
                col = self.format[name] if (self.format and name in self.format) else name
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
            if not arr_list:
                dataset[name] = np.empty((0, self.input_specs[name]), dtype=self.dtype)
            else:
                dataset[name] = np.concatenate(arr_list, axis=0).astype(self.dtype)
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

        max_window = max(self.input_specs.values())
        if n_rows < max_window:
            raise ValueError(
                f"CSV has only {n_rows} rows, but the largest required window is {max_window}."
            )

        # Common aligned sample end indices
        # If max_window = 5, valid end indices are 4,5,6,...
        t_start = max_window - 1
        t_end = n_rows - 1

        raw_data: Dict[str, List[np.ndarray]] = {name: [] for name in self.input_specs}

        for t in range(t_start, t_end + 1):
            for name, w in self.input_specs.items():
                start = t - w + 1
                end = t + 1

                col = self.format[name] if (self.format and name in self.format) else name
                if isinstance(col, int):
                    try:
                        values = df.iloc[start:end, col].to_numpy(dtype=self.dtype)
                    except IndexError:
                        raise ValueError(f"Column index {col} out of range for input '{name}'")
                else:
                    values = df[col].iloc[start:end].to_numpy(dtype=self.dtype)

                if values.shape[0] != w:
                    raise RuntimeError(
                        f"Internal error while building window for '{name}': "
                        f"expected {w}, got {values.shape[0]}"
                    )

                raw_data[name].append(values)

        dataset = {name: np.stack(windows, axis=0) for name, windows in raw_data.items()}
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

    # ------------------------------------------------------------------
    # Model introspection
    # ------------------------------------------------------------------

    def _extract_input_specs(self, model: Any) -> Dict[str, Dict[str, int]]:
        """
        Extract input names and window sizes from model.inputs.

        Expected output:
            {
                "data_1": 5,
                "data_2": 1,
                ...
            }

        This assumes each input object has:
        - a name: input.name
        - possibly a window/sample-window attribute if sw(...) was used
        """
        #raw_inputs = getattr(model, "inputs", None)
        input_shapes = getattr(model, "_input_shapes", None)

        specs: Dict[str, int] = {}
        for inp, shapes  in input_shapes.items():
            specs[inp] = shapes[1]
        return specs