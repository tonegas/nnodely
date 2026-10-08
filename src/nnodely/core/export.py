"""Exporting a built Modely to a Keras file or to ONNX, and reading them back.

A model is exported as it was declared, without its minimizers: only its
outputs, and only the inputs they read.
"""

import os
from pathlib import Path
from typing import Any

import keras
import numpy as np


def _traces_backward_pass(model) -> bool:
    """True if any layer of the graph evaluates a backward pass of its own.

    Layers declare this themselves, so the export path does not have to know
    which ones they are. The walk is recursive because such a layer can sit
    inside a nested model - a recurrent body, or the sub-graph a Derivative
    differentiates.
    """
    seen: set[int] = set()
    stack = [model]
    while stack:
        layer = stack.pop()
        if id(layer) in seen:
            continue
        seen.add(id(layer))
        if getattr(layer, "_traces_backward_pass", False):
            return True
        stack.extend(getattr(layer, "_layers", None) or [])
    return False


def _pair_onnx_names(values, expected):
    """Map each exported tensor to the Modely name it stands for.

    An exporter that renames the graph also reorders it, so position alone is
    not evidence of identity. A tensor whose shape matches exactly one expected
    name is that one whatever its position; only tensors left ambiguous - two
    outputs of the same shape - fall back to pairing in order.
    """

    def graph_shape(value):
        dims = value.type.tensor_type.shape.dim
        return tuple(int(axis.dim_value) for axis in dims[1:] if axis.dim_value)

    shapes = [graph_shape(value) for value in values]
    pending_graph = list(range(len(values)))
    pending_expected = list(range(len(expected)))

    pairs: dict[int, int] = {}
    for index in list(pending_graph):
        matches = [
            other for other in pending_expected if expected[other][1] == shapes[index]
        ]
        same_shape = [
            other for other in pending_graph if shapes[other] == shapes[index]
        ]
        if len(matches) == 1 and len(same_shape) == 1:
            pairs[index] = matches[0]
            pending_graph.remove(index)
            pending_expected.remove(matches[0])
    for index, other in zip(pending_graph, pending_expected):
        pairs[index] = other

    return {
        values[index].name: expected[other][0]
        for index, other in pairs.items()
        if values[index].name != expected[other][0]
    }


def _static_input_signature(model, batch_size: int):
    """The model's own input structure, with the batch axis fixed."""

    def spec(tensor):
        shape = (batch_size,) + tuple(int(axis) for axis in tensor.shape[1:])
        return keras.InputSpec(shape=shape, dtype=tensor.dtype, name=tensor.name)

    return [keras.tree.map_structure(spec, model._inputs_struct)]


def _export_file(path, filename: str | None, model_name: str, suffix: str) -> Path:
    """The file an exporter writes: ``filename`` - the model name by default -
    in the folder ``path``, with ``suffix`` added unless it already ends so."""
    name = model_name if filename is None else str(filename)
    if not name.lower().endswith(suffix):
        name += suffix
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    return folder / name


def export_keras(model, path: str | os.PathLike, filename: str | None = None) -> None:
    """The implementation of :meth:`Modely.export_keras`."""
    if model.inference_model is None:
        raise ValueError("Model is not built. Call build() before export_keras().")

    if not isinstance(model.inference_model, keras.Model):
        raise TypeError(f"Expected keras.Model, got {type(model.inference_model)}.")

    model.inference_model.save(_export_file(path, filename, model.name, ".keras"))


def import_keras(filename: str, safe_mode: bool = True):
    """The implementation of :meth:`Modely.import_keras`."""
    path = Path(filename)
    if path.suffix.lower() != ".keras":
        path = path.with_suffix(".keras")
    return keras.models.load_model(
        path,
        safe_mode=safe_mode,
    )


def export_onnx(
    model,
    path: str | os.PathLike,
    filename: str | None = None,
    *,
    input_signature=None,
    batch_size: int | None = None,
    opset_version: int | None = None,
    verbose: bool = False,
) -> None:
    """The implementation of :meth:`Modely.export_onnx`."""
    if model.inference_model is None:
        raise ValueError("Model is not built. Call build() before export_onnx().")

    if keras.backend.backend() == "jax":
        # Keras reaches ONNX from jax through jax2tf, and jax 0.4.36 removed
        # graph serialization, so jax2tf now emits the whole model as one
        # opaque XlaCallModule node that tf2onnx has no converter for.
        raise NotImplementedError(
            "ONNX export is not available on the jax backend: jax>=0.4.36 "
            "makes jax2tf emit an XlaCallModule node that tf2onnx cannot "
            "convert. Export from the tensorflow or torch backend instead "
            "(set KERAS_BACKEND before importing keras)."
        )

    from nnodely.layers.torch_module import TorchModuleImpl

    if keras.backend.backend() == "tensorflow" and any(
        isinstance(layer, TorchModuleImpl)
        for layer in model.inference_model._flatten_layers()
    ):
        # There a TorchModule is a host callback, which tf2onnx writes as a
        # PyFunc node that no ONNX runtime can execute.
        raise NotImplementedError(
            "ONNX export of a model with a TorchModule is not available on the "
            "tensorflow backend, where the module runs as a Python callback. "
            "Export from the torch backend instead (set KERAS_BACKEND before "
            "importing keras)."
        )

    file = _export_file(path, filename, model.name, ".onnx")

    inference_model = model.inference_model
    export_model = inference_model
    if keras.backend.backend() == "torch" and isinstance(inference_model.input, dict):
        # Keras' Torch ONNX exporter does not currently accept dictionary
        # signatures. Trace an equivalent positional wrapper instead.
        export_inputs = [
            keras.Input(
                shape=tuple(tensor.shape[1:]),
                dtype=tensor.dtype,
                name=tensor.name,
            )
            for tensor in inference_model.inputs
        ]
        input_map = {
            tensor.name: export_input
            for tensor, export_input in zip(inference_model.inputs, export_inputs)
        }
        export_model = keras.Model(
            export_inputs,
            inference_model(input_map, training=False),
            name=f"{inference_model.name}_onnx",
        )

    # Keras requires a model to have been called before export. Modely.build
    # creates a Functional model, but does not necessarily execute it.
    if not getattr(export_model, "_called", False):
        warmup_inputs = {}
        for tensor in export_model.inputs:
            shape = tuple(1 if dim is None else int(dim) for dim in tensor.shape)
            warmup_inputs[tensor.name] = np.zeros(shape, dtype=np.float32)
        export_model(warmup_inputs, training=False)

    if _traces_backward_pass(export_model) and keras.backend.backend() == "torch":
        raise NotImplementedError(
            "ONNX export of a model that differentiates a sub-graph - a "
            "Derivative with respect to an Input - is not supported on the "
            "'torch' backend: its exporter traces a forward pass only and "
            "cannot record the backward pass such a layer evaluates. The "
            "'tensorflow' and 'jax' backends export it, because there the "
            "backward pass becomes ordinary graph operations."
        )

    if input_signature is None:
        if batch_size is None and _traces_backward_pass(export_model):
            batch_size = 1
        if batch_size is not None:
            input_signature = _static_input_signature(export_model, batch_size)

    export_kwargs: dict[str, Any] = {"verbose": verbose}
    if input_signature is not None:
        export_kwargs["input_signature"] = input_signature
    if opset_version is not None:
        export_kwargs["opset_version"] = opset_version

    export_model.export(file, format="onnx", **export_kwargs)
    _set_onnx_io_names(model, file)


def _set_onnx_io_names(model, path: Path) -> None:
    """Restore Modely input/output names if an exporter replaced them."""
    try:
        import onnx
    except ImportError:
        return

    if model.inference_model is None or not model.built:
        raise ValueError("Model is not built. Call build() before export_onnx().")
    onnx_model = onnx.load(str(path))
    graph = onnx_model.graph
    expected_inputs = [
        (tensor.name, tuple(int(axis) for axis in tensor.shape[1:] if axis))
        for tensor in model.inference_model.inputs
    ]
    expected_outputs = [(node.name, tuple(node.shape.tuple)) for node in model.outputs]
    rename = {}

    # Only an exporter that dropped the names has to be corrected. One that
    # kept them may still have reordered them - both tf2onnx and the Torch
    # exporter sort their outputs - and pairing those off by position would
    # rename each tensor after a different one, silently swapping two
    # outputs' values.
    for values, expected in (
        (graph.input, expected_inputs),
        (graph.output, expected_outputs),
    ):
        names = [value.name for value in values]
        if len(names) != len(expected) or set(names) == {name for name, _ in expected}:
            continue
        rename.update(_pair_onnx_names(values, expected))
    if not rename:
        return

    for collection in (
        graph.input,
        graph.output,
        graph.value_info,
        graph.initializer,
    ):
        for value in collection:
            value.name = rename.get(value.name, value.name)
    for node in graph.node:
        for idx, name in enumerate(node.input):
            node.input[idx] = rename.get(name, name)
        for idx, name in enumerate(node.output):
            node.output[idx] = rename.get(name, name)

    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, str(path))


def validate_onnx(
    filename: str | os.PathLike,
    inputs: dict,
    *,
    return_dict: bool = False,
    providers: list[str] | None = None,
):
    """The implementation of :meth:`Modely.validate_onnx`."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "validate_onnx() requires the optional 'onnxruntime' package."
        ) from exc

    path = Path(filename)
    if path.suffix.lower() != ".onnx":
        path = path.with_suffix(".onnx")
    if not path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {path}")

    session_kwargs = {} if providers is None else {"providers": providers}
    session = ort.InferenceSession(str(path), **session_kwargs)  # type: ignore
    input_names = [item.name for item in session.get_inputs()]
    missing = [name for name in input_names if name not in inputs]
    if missing:
        raise ValueError(f"Missing ONNX inputs: {missing}")

    feed = {name: np.asarray(inputs[name], dtype=np.float32) for name in input_names}
    outputs = session.run(None, feed)
    if return_dict:
        return {item.name: value for item, value in zip(session.get_outputs(), outputs)}
    return outputs
