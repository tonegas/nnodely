"""Exporting a built Modely to a Keras file or to ONNX, and reading them back.

A model is exported as it was declared, without its minimizers: only its
outputs, and only the inputs they read.
"""

import importlib.util
import os
from pathlib import Path

import keras
import numpy as np


def _traces_backward_pass(model) -> bool:
    """True if any layer of the graph evaluates a backward pass of its own.

    Layers declare this themselves, so the export path does not have to know
    which ones they are. The walk is recursive because such a layer can sit
    inside a nested model, such as a recurrent body or the sub-graph a
    Gradient differentiates.
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


def _pair_onnx_names(values, expected, by_name: bool = False):
    """Map each exported tensor to the Modely name it stands for.

    An exporter that renames the graph also reorders it, so position alone is
    not evidence of identity. A tensor whose shape matches exactly one expected
    name is that one whatever its position. Only tensors left ambiguous, two
    outputs of the same shape, fall back to pairing in order: the order of
    the names with ``by_name``, as Keras flattens a dict of outputs, else the
    order given.
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
    if by_name:
        pending_expected.sort(key=lambda other: expected[other][0])
    for index, other in zip(pending_graph, pending_expected):
        pairs[index] = other

    return {
        values[index].name: expected[other][0]
        for index, other in pairs.items()
        if values[index].name != expected[other][0]
    }


#: The package Keras converts a model to ONNX with, on each backend.
_CONVERTERS = {"tensorflow": "tf2onnx", "torch": "onnxscript", "jax": "jax2onnx"}


def _static_input_signature(keras_model, batch_size: int, seq: int | None, name: str):
    """The input structure of ``keras_model``, every axis fixed: the batch to
    ``batch_size``, a sequence of dynamic length to ``seq``. ``name``
    is the Modely's, for the error."""

    def spec(tensor):
        sizes = []
        for axis in tensor.shape[1:]:
            if axis is None:
                if seq is None:
                    raise ValueError(
                        f"Model {name!r} reads {tensor.name!r} as a "
                        "sequence of dynamic length, and an ONNX file has "
                        "fixed shapes: give its length, export_onnx(..., "
                        "seq=n). The exported file computes sequences "
                        "of that length."
                    )
                axis = seq
            sizes.append(int(axis))
        return keras.InputSpec(
            shape=(batch_size, *sizes), dtype=tensor.dtype, name=tensor.name
        )

    return [keras.tree.map_structure(spec, keras_model._inputs_struct)]


def _positive_int(value, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"export_onnx(): {name} is a positive int, got {value!r}.")


def _export_file(path, filename: str | None, model_name: str, suffix: str) -> Path:
    """The file an exporter writes: ``filename`` (the model name by default)
    in the folder ``path``, with ``suffix`` added unless it already ends so."""
    name = model_name if filename is None else str(filename)
    if not name.lower().endswith(suffix):
        name += suffix
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    return folder / name


def export_keras(model, path: str | os.PathLike, filename: str | None = None) -> None:
    """The implementation of :meth:`Modely.export_keras`."""
    if model._keras_inference_model is None:
        raise ValueError("Model is not built. Call build() before export_keras().")
    model._check_names_built()

    if not isinstance(model._keras_inference_model, keras.Model):
        raise TypeError(
            f"Expected keras.Model, got {type(model._keras_inference_model)}."
        )

    _standalone_copy(model._keras_inference_model).save(
        _export_file(path, filename, model.name, ".keras")
    )


def _standalone_copy(keras_model: keras.Model) -> keras.Model:
    """A copy of ``keras_model`` that a ``.keras`` file can hold and give back.

    nnodely shares layers between Keras graphs: a body rolled out by two
    Loops, a Parameter read inside a Loop body and outside it, the sub-graphs
    a Gradient evaluates again. Keras saves such a layer once, but
    cannot rebuild a layer called by two graphs: it replays the calls of
    each graph by their position. Serialized outside a sharing scope, every
    nested model and layer is written in full wherever it is used, so the
    copy calls each layer from one graph only; the values of the model are
    then copied in, layer by layer.
    """
    config = keras.saving.serialize_keras_object(keras_model)
    copy = keras.saving.deserialize_keras_object(config)
    for source, target in zip(keras_model.layers, copy.layers):
        if len(source.weights) != len(target.weights):
            raise RuntimeError(
                f"The Keras copy of layer {source.name!r} has "
                f"{len(target.weights)} weights where it has {len(source.weights)}."
            )
        for value, variable in zip(source.weights, target.weights):
            variable.assign(value)
    return copy


def import_keras(filename: str):
    """The implementation of :meth:`Modely.import_keras`."""
    path = Path(filename)
    if path.suffix.lower() != ".keras":
        path = path.with_suffix(".keras")
    return keras.models.load_model(path)


def export_onnx(
    model,
    path: str | os.PathLike,
    filename: str | None = None,
    *,
    batch_size: int = 1,
    seq: int | None = None,
    opset_version: int | None = None,
    verbose: bool = False,
) -> None:
    """The implementation of :meth:`Modely.export_onnx`."""
    if model._keras_inference_model is None:
        raise ValueError("Model is not built. Call build() before export_onnx().")
    model._check_names_built()
    _positive_int(batch_size, "batch_size")
    if seq is not None:
        _positive_int(seq, "seq")

    backend = keras.backend.backend()
    converter = _CONVERTERS.get(backend)
    if converter is not None and importlib.util.find_spec(converter) is None:
        raise ImportError(
            f"ONNX export on the {backend} backend needs the {converter} "
            f'package, which comes with the backend: pip install "nnodely[{backend}]".'
        )

    file = _export_file(path, filename, model.name, ".onnx")

    inference_model = model._keras_inference_model
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
            "ONNX export of a model that differentiates a sub-graph (a "
            "Gradient with respect to an Input) is not supported on the "
            "'torch' backend: its exporter traces a forward pass only and "
            "cannot record the backward pass such a layer evaluates. Export "
            "it from the 'jax' backend, or from 'tensorflow' when tf2onnx "
            "converts the gradients of its layers."
        )

    # Every axis fixed: the torch exporter keeps no dynamic axis right, and
    # jax2onnx none through nnodely's layers.
    export_model.export(
        file,
        format="onnx",
        verbose=verbose,
        input_signature=_static_input_signature(
            export_model, batch_size, seq, model.name
        ),
        opset_version=opset_version,
    )
    _refuse_unknown_operators(file, model.name)
    _set_onnx_io_names(model, file, seq)


def _refuse_unknown_operators(path: Path, name: str) -> None:
    """Remove the file and raise when it holds operators ONNX does not
    define, which tf2onnx copies as they are when it cannot convert them,
    rather than leave a file no runtime loads."""
    import onnx

    unknown: set[str] = set()

    def walk(graph) -> None:
        for node in graph.node:
            for attribute in node.attribute:
                for body in (attribute.g, *attribute.graphs):
                    walk(body)
            if node.domain in ("", "ai.onnx") and not onnx.defs.has(node.op_type):
                unknown.add(node.op_type)

    walk(onnx.load(str(path)).graph)
    if unknown:
        path.unlink()
        raise NotImplementedError(
            f"Model {name!r} cannot be exported to ONNX on the "
            f"{keras.backend.backend()!r} backend: its exporter has no ONNX "
            f"operator for {sorted(unknown)}. A Gradient records the gradient "
            "of every layer it differentiates, such as TanhGrad for a Tanh: "
            "export such a model from the 'jax' backend."
        )


def _set_onnx_io_names(model, path: Path, seq: int | None = None) -> None:
    """Restore Modely input/output names if an exporter replaced them; a
    dynamic sequence axis was exported ``seq`` long."""
    import onnx

    onnx_model = onnx.load(str(path))
    graph = onnx_model.graph

    def exported(shape) -> tuple:
        sizes = (seq if axis is None else axis for axis in shape)
        return tuple(int(size) for size in sizes if size is not None)

    expected_inputs = [
        (tensor.name, exported(tensor.shape[1:]))
        for tensor in model._keras_inference_model.inputs
    ]
    expected_outputs = [
        (node.name, exported(node.shape.tuple)) for node in model._outputs
    ]
    rename = {}

    # Only an exporter that dropped the names has to be corrected. One that
    # kept them may still have reordered them, since both tf2onnx and the
    # Torch exporter sort their outputs. Pairing those off by position would
    # rename each tensor after a different one, silently swapping two
    # outputs' values. The order is restored below, by name.
    for values, expected, by_name in (
        (graph.input, expected_inputs, False),
        (graph.output, expected_outputs, True),
    ):
        names = [value.name for value in values]
        if len(names) != len(expected) or set(names) == {name for name, _ in expected}:
            continue
        rename.update(_pair_onnx_names(values, expected, by_name=by_name))

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

    # The exporters sort the outputs: listed again as model.outputs lists
    # them, a positional reading of the results follows the model.
    order = [name for name, _ in expected_outputs]
    listed = [value.name for value in graph.output]
    reorder = listed != order and sorted(listed) == sorted(order)
    if reorder:
        by_name = {value.name: value for value in graph.output}
        outputs = []
        for name in order:
            value = onnx.ValueInfoProto()
            value.CopyFrom(by_name[name])
            outputs.append(value)
        del graph.output[:]
        graph.output.extend(outputs)
    if not rename and not reorder:
        return

    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, str(path))


def run_onnx(
    filename: str | os.PathLike,
    inputs: dict,
    *,
    return_dict: bool = False,
):
    """The implementation of :meth:`Modely.run_onnx`."""
    import onnxruntime as ort

    path = Path(filename)
    if path.suffix.lower() != ".onnx":
        path = path.with_suffix(".onnx")
    if not path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {path}")

    session = ort.InferenceSession(str(path))
    input_names = [item.name for item in session.get_inputs()]
    missing = [name for name in input_names if name not in inputs]
    if missing:
        raise ValueError(f"Missing ONNX inputs: {missing}")

    feed = {name: np.asarray(inputs[name], dtype=np.float32) for name in input_names}
    outputs = session.run(None, feed)
    if return_dict:
        return {item.name: value for item, value in zip(session.get_outputs(), outputs)}
    return outputs
