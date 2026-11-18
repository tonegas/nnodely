# Adding or changing a layer

Read [`architecture.md`](architecture.md) first. Copy the closest existing
layer rather than starting from scratch:

| The new layer | Start from |
| --- | --- |
| applies a function to every value, no settings | `Sin` in `layers/trigonometric.py` (`UnaryOp`) |
| combines two streams elementwise | `Atan2` in `layers/trigonometric.py` (`BinaryOp`) |
| has weights | `Fir` in `layers/fir.py`, `Linear` in `layers/linear.py` |
| selects or reshapes axes | `layers/partitions.py` |
| acts over time (state, integration) | `layers/calculus.py`, `layers/buffer.py` |

## Checklist

1. **Keras layer** `<Name>Impl`: registered with
   `@keras.saving.register_keras_serializable(package="nnodely")`, computes
   with `keras.ops` (a backend branch only as the hard rules of `AGENTS.md`
   allow, with a clear error on a backend that cannot do it),
   `get_config()` returns every constructor argument.
2. **nnodely layer** `<Name>(Layer)`: typed constructor that passes its
   settings to `super().__init__(name, **settings)` (they become
   `_properties`, which `get_config()` saves); `output_shape()` overridden
   when the shape cannot be probed with a dummy tensor (dynamic axes, a
   changed number of sequence axes); `build_layer()` returns the Impl.
3. **Validation at call time**: refuse wrong shapes in `output_shape()` or
   `_weight_shapes()` with a message naming the layer and the fix
   (`self._needs_window(x)` for layers that need a window).
4. **Public API**: import it in `src/nnodely/__init__.py` and add it to
   `__all__`.
5. **Reference page**: list it in the right `.. autosummary::` block of
   `docs/api/layers.rst`. Write its docstring as described in
   [`documentation.md`](documentation.md); end it with
   `.. versionadded:: <next version>`.
6. **Tests** (see [`testing.md`](testing.md)): output shape for every calling
   form, values against NumPy, gradient when it is trainable, `save` /
   `load` round-trip, `export_keras`, and ONNX export.
7. **Changelog**: an entry under `### Added`.
8. Run the fast suite on all three backends:
   `KERAS_BACKEND=<b> uv run pytest tests -m "not slow"` for each.

## Changing an existing layer

- Its constructor arguments are saved in model files: renaming or removing
  one breaks loading. Bump `ModelSerializer.VERSION` and note it in the
  changelog.
- Its output shape is a contract: every doctest that prints it, and the
  guide pages, must still pass (`uv run pytest tests/test_docs.py`).
