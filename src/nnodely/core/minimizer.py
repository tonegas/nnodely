"""The objectives of a Modely: how a minimizer is checked, resolved against
the built graph, and turned into the loss the training model minimizes."""

import warnings

import keras
import numpy as np

from nnodely.core.dag import _streams_from
from nnodely.core.layer import Layer
from nnodely.core.stream import Node, Stream
from nnodely.layers.input import Input
from nnodely.utils.utils import MaskedLoss, _resolve_loss, _weighted_sequence_loss


def _depends_on_dynamic_input(node) -> bool:
    """True if a stream reads an Input whose sequence length is left dynamic.

    Those are the inputs a DataLoader pads when simulations differ in length,
    so they are the streams whose padded rollout steps the mask removes.
    """
    seen: set[int] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, Input) and any(n is None for n in current.seq):
            return True
        stack.extend(current.preds)
    return False


def _reads_input(node) -> bool:
    """False for a stream computed from Constants and Parameters alone."""
    seen: set[int] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, Input):
            return True
        stack.extend(current.preds)
    return False


def _reaches_trainable_weight(node) -> bool:
    """False only for a stream known to be computed without trainable weights.

    The walk enters the body of a called model as well. A layer not built yet
    cannot tell, and is taken to train: the check must never flag a minimizer
    that does.
    """
    from nnodely.core.modely import Modely

    seen: set[int] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, Layer):
            layer = current._layer
            if layer is None or layer.trainable_weights:
                return True
        inner = getattr(current, "model", None)
        if isinstance(inner, Modely):
            stack.extend(inner.outputs)
        stack.extend(current.preds)
    return False


def _validate_gain(gain, minimizer: str) -> float:
    """A minimizer's weight in the total loss: a finite number, not negative.

    A negative gain would maximize the error it weighs.
    """
    if isinstance(gain, (bool, np.bool_)) or not isinstance(
        gain, (int, float, np.integer, np.floating)
    ):
        raise TypeError(
            f"The gain of minimizer {minimizer!r} must be a number, got "
            f"{type(gain).__name__}."
        )
    gain = float(gain)
    if not np.isfinite(gain) or gain < 0:
        raise ValueError(
            f"The gain of minimizer {minimizer!r} must be finite and not "
            f"negative, got {gain}."
        )
    return gain


def _seq_weights_for(seq_weights, steps: int | None, minimizer: str):
    """One weight per step of a last axis ``steps`` long, normalized to mean one.

    A callable is given the number of steps. Weights are not negative, and not
    all zero: a negative weight would maximize the error of its step, and all
    zeros would train nothing. ``steps=None`` leaves the length unchecked.
    """
    weights = seq_weights(steps) if callable(seq_weights) else seq_weights
    weights = np.asarray(weights, dtype=np.float32)
    if weights.ndim != 1 or steps is not None and weights.shape != (steps,):
        raise ValueError(
            f"The seq_weights of minimizer {minimizer!r} must be one weight per "
            f"step of the last axis, {steps} of them, got shape {weights.shape}."
        )
    if not np.isfinite(weights).all() or (weights < 0).any() or not weights.any():
        raise ValueError(
            f"The seq_weights of minimizer {minimizer!r} must be finite, not "
            f"negative and not all zero, got {weights.tolist()}."
        )
    return weights / weights.mean()


def _validate_seq_weights(seq_weights, source: Stream, minimizer: str):
    """The seq_weights of a minimizer, checked as far as its source allows.

    A dynamic sequence follows the data, whatever length its stream declares,
    so its weights are checked once the data sets the length.
    """
    if seq_weights is None:
        return None
    steps = None if _depends_on_dynamic_input(source) else source.shape.tuple[-1]
    if callable(seq_weights):
        if steps is not None:
            _seq_weights_for(seq_weights, steps, minimizer)
        return seq_weights
    return _seq_weights_for(seq_weights, steps, minimizer)


class _MinimizerTerm:
    """One minimizer, resolved against the outputs of the built graph."""

    def __init__(
        self,
        name,
        source,
        target,
        target_constant,
        loss,
        masked,
        log_name,
        gain=1.0,
        seq_weights=None,
    ):
        self.name = name
        self.gain = gain  # weight of this term in the total loss
        # weight of each step of the last axis, or a callable of its length
        self.seq_weights = seq_weights
        self.source = source  # name of the graph output holding the prediction
        self.target = target  # name of the graph output holding the reference
        # a target computed from constants alone, which broadcasts to its source
        self.target_constant = target_constant
        self.masked = masked
        self.loss = MaskedLoss(loss) if masked else loss
        # Logged under the key compile() gave a per-output loss, which is the
        # one the printers and existing training scripts read.
        self.tracker = keras.metrics.Mean(name=f"{log_name}_loss")


class _MinimizerTerms:
    """Plain holder, so Keras does not track the terms as model state.

    The minimizers can change between two calls to train() without a new
    build(), and state tracked by the model could only ever grow.
    """

    def __init__(self, terms):
        self.terms = list(terms)


@keras.saving.register_keras_serializable(package="nnodely")
class MinimizerModel(keras.Model):
    """The built graph, trained on the minimizers of its Modely.

    Keras compiles one loss per output against a label fed from the dataset,
    but a minimizer compares two streams of the graph: its target can be an
    Output, a computed stream or a window narrower than the dataset column.
    Both sides are therefore read from the same forward pass, and the loss is
    computed here. The labels carry only the padding mask, when simulations of
    different lengths had to be padded to one rollout width.
    """

    def set_minimizers(self, terms):
        self._nnodely_minimizers = _MinimizerTerms(terms)

    def _minimizer_terms(self):
        holder = getattr(self, "_nnodely_minimizers", None)
        return holder.terms if holder is not None else []

    @property
    def metrics(self):
        return super().metrics + [term.tracker for term in self._minimizer_terms()]

    @staticmethod
    def _align_target(term, target, source):
        """Give a target built from constants the axes of its source.

        Only such a target is broadcast. Two streams read from the data that
        differ in shape are an error: a loss would broadcast one over the other
        and silently compare, say, one sample with a whole window.
        """
        broadcast = term.target_constant
        if broadcast:
            while len(target.shape) < len(source.shape):
                target = keras.ops.expand_dims(target, axis=-1)

        source_axes, target_axes = tuple(source.shape[1:]), tuple(target.shape[1:])
        compatible = len(source_axes) == len(target_axes) and all(
            s is None or t is None or s == t or (broadcast and t == 1)
            for s, t in zip(source_axes, target_axes)
        )
        if not compatible:
            raise ValueError(
                f"Minimizer {term.name!r} compares {term.source!r} of shape "
                f"{source_axes} with {term.target!r} of shape {target_axes}. "
                "They must have the same shape."
            )
        if broadcast:
            target = keras.ops.broadcast_to(target, keras.ops.shape(source))
        return target

    def compute_loss(
        self, x=None, y=None, y_pred=None, sample_weight=None, training=True
    ):
        terms = self._minimizer_terms()
        if not terms:
            return super().compute_loss(x, y, y_pred, sample_weight, training=training)

        total = None
        for term in terms:
            if y_pred is None:
                raise ValueError(
                    f"Minimizer {term.name!r} compares {term.source!r} with "
                    f"{term.target!r}, but the forward pass did not return "
                    "the graph outputs it needs. Call build() after adding a "
                    "minimizer, before train()."
                )
            source = y_pred[term.source]
            target = self._align_target(term, y_pred[term.target], source)
            # The mask is one row per sample, one column per rollout step, and
            # the rollout is the last axis of every padded stream.
            rollout = (
                target.shape[-1] in (None, y.shape[-1]) if y is not None else False
            )
            if term.masked and rollout:
                valid = keras.ops.cast(y, "bool")
                for _ in range(len(target.shape) - 2):
                    valid = keras.ops.expand_dims(valid, axis=1)
                target = keras.ops.where(
                    valid, target, keras.ops.full_like(target, np.nan)
                )
            if term.seq_weights is not None:
                # The rollout is unrolled, so its length is static here.
                weights = _seq_weights_for(
                    term.seq_weights, source.shape[-1], term.name
                )
                value = _weighted_sequence_loss(term.loss, target, source, weights)
            else:
                # A loss function returns one value per sample, a Loss instance
                # the reduced value: the mean is how compile() reduces either.
                value = keras.ops.mean(term.loss(target, source))
            # Logged unweighted by the gain, as Keras logs a per-output loss
            # under loss_weights; only the total the optimizer sees is weighted.
            # The seq_weights are part of the loss itself, so they are logged.
            term.tracker.update_state(value)
            weighted = value if term.gain == 1.0 else value * term.gain
            total = weighted if total is None else total + weighted
        if self.losses:
            total = total + keras.ops.sum(self.losses)
        return total


# ----------------------------------------------------------------------
# The minimizers of a model
# ----------------------------------------------------------------------


def _minimizer_stream(model, value, side: str, minimizer: str) -> Stream:
    """A minimizer side as a Stream, looking a name up in the model.

    A name can be any stream of the model's graph, one of its declared inputs,
    or a stream the registered minimizers already compare.
    """
    if isinstance(value, Stream):
        return value
    if not isinstance(value, str):
        raise TypeError(
            f"The {side} of minimizer {minimizer!r} must be a Stream or the "
            f"name of one, got {type(value).__name__}."
        )
    roots: list[Node] = [*model.outputs, *model.inputs]
    for registered in model.minimizers:
        roots.extend((registered["source"], registered["target"]))
    matches = [stream for stream in _streams_from(roots) if stream.name == value]
    if not matches:
        outputs = sorted(output.name for output in model.outputs)
        raise ValueError(
            f"The {side} {value!r} of minimizer {minimizer!r} is not the name "
            f"of a stream of model {model.name!r}. Its outputs are {outputs}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"The {side} {value!r} of minimizer {minimizer!r} is ambiguous: "
            f"{len(matches)} streams of model {model.name!r} carry that name. "
            "Pass the stream itself."
        )
    return matches[0]


def _resolve_minimizer_terms(model) -> list[_MinimizerTerm]:
    """Resolve every minimizer of ``model`` to the graph outputs of its build."""
    assert model.model is not None
    # The forward pass returns a dict keyed by the names of these nodes.
    output_names = {node.name for node in model.train_outputs}

    def output_name(minimizer, side):
        node = minimizer[side]
        name = model._minimizer_outputs.get(node)
        if name is None and not isinstance(node, Input):
            name = node.name
        if name not in output_names:
            raise ValueError(
                f"The {side} {node.name!r} of minimizer {minimizer['name']!r} "
                "is not an output of the built model. Call build() after "
                "adding a minimizer, before train()."
            )
        return name

    return [
        _MinimizerTerm(
            name=minimizer["name"],
            source=output_name(minimizer, "source"),
            target=output_name(minimizer, "target"),
            target_constant=not _reads_input(minimizer["target"]),
            loss=_resolve_loss(minimizer["loss"]),
            masked=_depends_on_dynamic_input(minimizer["source"])
            or _depends_on_dynamic_input(minimizer["target"]),
            log_name=minimizer["source"].name,
            gain=minimizer.get("gain", 1.0),
            seq_weights=minimizer.get("seq_weights"),
        )
        for minimizer in model.minimizers
    ]


def _check_minimizers_train_weights(model, km: keras.Model) -> None:
    """Every minimizer should compare something the network computes.

    A minimizer whose source and target reach no trainable weight adds a
    constant to the loss and trains nothing. Alongside minimizers that do
    train, it is only worth a warning; when none of them trains the
    network's weights, the training could not change anything.
    """
    if not km.trainable_weights:
        return  # Keras itself warns that the model has nothing to train
    weightless = [
        minimizer["name"]
        for minimizer in model.minimizers
        if not _reaches_trainable_weight(minimizer["source"])
        and not _reaches_trainable_weight(minimizer["target"])
    ]
    if len(weightless) == len(model.minimizers):
        raise ValueError(
            f"No minimizer of model {model.name!r} depends on a trainable "
            f"weight: {weightless} compare streams computed without any, so "
            "training could not change the network. Compare an output of "
            "the network with its reference."
        )
    for name in weightless:
        warnings.warn(
            f"Minimizer {name!r} of model {model.name!r} depends on no "
            "trainable weight: it adds a constant to the loss and trains "
            "nothing.",
            UserWarning,
            stacklevel=3,
        )
