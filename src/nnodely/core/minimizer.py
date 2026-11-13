"""The objectives of a Modely: how a minimizer is checked, resolved against
the built graph, and turned into the loss the training model minimizes."""

import warnings
from typing import TYPE_CHECKING, TypeGuard

import keras
import numpy as np

from nnodely.core.dag import _nodes_by_model, resolve, toposort
from nnodely.core.layer import Layer
from nnodely.core.shape import OUTER_AXIS, expand_to_rank, outer_view
from nnodely.core.stream import Node, Stream
from nnodely.layers.input import Input
from nnodely.utils.utils import MaskedLoss, _resolve_loss

if TYPE_CHECKING:
    from nnodely.layers.loop import Loop


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
        stack.extend(current._preds)
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
        stack.extend(current._preds)
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
        inner = current._model
        if isinstance(inner, Modely):
            stack.extend(inner._outputs)
        stack.extend(current._preds)
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
        gain=1.0,
    ):
        self.name = name
        self.gain = gain  # weight of this term in the total loss
        self.source = source  # name of the graph output holding the prediction
        self.target = target  # name of the graph output holding the reference
        # a target computed from constants alone, which broadcasts to its source
        self.target_constant = target_constant
        self.masked = masked
        self.loss = MaskedLoss(loss) if masked else loss
        # Logged under the minimizer's name: two minimizers can share a source.
        self.tracker = keras.metrics.Mean(name=f"{name}_loss")


def _rollout_mask(term, target, y):
    """The padding mask ``y`` laid out to broadcast against ``target``, or
    None when it does not apply to the minimizer.

    The mask is one row per sample, one column per rollout step, and the
    rollout is the outer step axis of every padded stream.
    """
    if not term.masked or y is None:
        return None
    if target.shape[OUTER_AXIS] not in (None, y.shape[-1]):
        return None
    return outer_view(keras.ops.cast(y, "bool"), len(target.shape))


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
            target = expand_to_rank(target, len(source.shape))

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

    @staticmethod
    def _term_loss(term, source, target, y):
        """One minimizer's loss: ``source`` against its aligned ``target``,
        with the padded rollout steps the mask ``y`` marks left out.

        Validation computes its loss here too, so a model is scored on its
        validation data exactly as it was trained.
        """
        valid = _rollout_mask(term, target, y)
        if valid is not None:
            target = keras.ops.where(valid, target, keras.ops.full_like(target, np.nan))
        # A loss function returns one value per sample, a Loss instance the
        # reduced value: the mean is how compile() reduces either.
        return keras.ops.mean(term.loss(target, source))

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
            value = self._term_loss(term, source, target, y)
            # Logged unweighted by the gain, as Keras logs a per-output loss
            # under loss_weights; only the total the optimizer sees is weighted.
            # Each batch weighs as many samples as it holds, as in Keras's own
            # loss: a short last batch would otherwise count as a full one.
            term.tracker.update_state(value, sample_weight=keras.ops.shape(source)[0])
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
    or a stream the registered minimizers already compare. A name several
    streams carry raises, listing them: the stream itself is passed then.
    """
    if isinstance(value, Stream):
        return value
    if not isinstance(value, str):
        raise TypeError(
            f"The {side} of minimizer {minimizer!r} must be a Stream or the "
            f"name of one, got {type(value).__name__}."
        )
    roots: list[Node] = [*model._outputs, *model._inputs]
    for registered in model._minimizers:
        roots.extend((registered["source"], registered["target"]))
    candidates = [
        (stream, None, _where(stream))
        for stream in toposort(roots)
        if isinstance(stream, Stream)
    ]
    return resolve(
        value,
        candidates,
        kind="stream",
        owner=f"The {side} of minimizer {minimizer!r} in model {model.name!r}",
        hint="Pass the stream itself, or give it a name of its own, "
        "stream.name = 'new_name'.",
        missing=lambda value: (
            f"The {side} {value!r} of minimizer {minimizer!r} "
            f"is not the name of a stream of model {model.name!r}. Its outputs are "
            f"{sorted(output.name for output in model._outputs)}."
        ),
    )


def _resolve_sides(model, source, target, minimizer: str) -> tuple:
    """The streams of ``model`` a minimizer compares, its ``source`` and
    ``target`` given as streams.

    A stream of the body of a Loop the model rolls out - an output of the
    body, or the stream it names, or an Input the Loop feeds - stands for
    what the Loop makes of it: the trajectory it returns for an output, the
    stream it feeds the Input from. That is how a model declared one step
    at a time, as a ModelyLoop is, is minimized on the streams of its step.
    A bare Input the model does not read, compared with a trajectory, that
    has the shape of one step of it is a target declared one step at a
    time: it reads one sample per step, an Input of its name with the
    horizon of the trajectory.

    Every other side is computed from the graph of the model, from values,
    and from Inputs only minimizers read. One that reads the graph of
    another model - the step of a ModelyLoop, the body of a Loop, a block
    called - is refused, and so is a target of another shape than its
    source.
    """
    nodes, known = _known_nodes(model)
    loops = [node for node in nodes if _is_loop(node)]
    source = _of_loops(model, source, loops, known, "source", minimizer)
    target = _of_loops(model, target, loops, known, "target", minimizer)
    source = _one_step_input(nodes, known, source, target)
    target = _one_step_input(nodes, known, target, source)
    for side, value in (("source", source), ("target", target)):
        _refuse_foreign(model, nodes, known, value, side, minimizer)
    _check_shapes(source, target, minimizer)
    return source, target


def _is_loop(node) -> TypeGuard["Loop"]:
    # loop.py imports the module of Modely, which imports this one.
    from nnodely.layers.loop import Loop

    return isinstance(node, Loop)


def _known_nodes(model) -> tuple[list[Node], set[int]]:
    """Every node of the graph of ``model`` - its outputs, the inputs it
    declares and the minimizers it has - and their ids."""
    roots: list[Node] = [*model._outputs, *model._inputs]
    for registered in model._minimizers:
        roots.extend((registered["source"], registered["target"]))
    nodes = toposort(roots)
    return nodes, {id(node) for node in nodes}


def _of_loops(model, value, loops, known: set[int], side: str, minimizer: str):
    """What ``value``, a stream of the body of one of ``loops``, stands for
    in ``model``: the trajectory the Loop returns for an output of its
    body, or the stream it feeds an Input of its body from; ``value`` itself
    when it is no stream of a body. A body the model rolls out more than
    once is ambiguous, and a state has no trajectory of its own."""
    if id(value) in known or value._is_value:
        return value
    # Each stream it stands for, with the name of the Loop that returns it.
    found: dict[int, tuple[Stream, str]] = {}
    for loop in loops:
        step = loop._step
        # The outputs the Loop returns, as they are in its body and in the
        # graph it was declared over - the step of a ModelyLoop, which the
        # body copies.
        for returned in step.returns:
            declared = step.copied_from(returned)
            if any(
                value is node
                for node in (returned, declared, returned._preds[0], declared._preds[0])
            ):
                stream = _returned_by(loop, returned)
                found[id(stream)] = (stream, loop.name)
        if not isinstance(value, Input):
            continue
        key = step.key(value)
        for state, output in step.pairs:
            if step.key(state) == key:
                raise ValueError(
                    f"Minimizer {minimizer!r}: its {side} {value.name!r} is fed "
                    f"back by Loop {loop.name!r} with {output.name!r}: within a "
                    "step it holds the value of the step before, and it has no "
                    f"trajectory of its own. Minimize {output.name!r}, its "
                    "trajectory one step ahead - an output of the step, or "
                    "declared among them."
                )
        for node, stream in zip(loop._static_inputs, loop._static_sources):
            if step.key(node) == key:
                found[id(stream)] = (stream, loop.name)
        # An Input the body starts a state from: the initial value the
        # Loop reads for it.
        for state, start in step.starts.items():
            if isinstance(start, Input) and step.key(start) == key:
                stream = loop._initial_values[loop._feedback_inputs.index(state)]
                found[id(stream)] = (stream, loop.name)
    if len(found) > 1:
        names = sorted({name for _, name in found.values()})
        raise ValueError(
            f"Minimizer {minimizer!r}: its {side} {value.name!r} is a stream of "
            f"a body model {model.name!r} rolls out {len(found)} times, by the "
            f"Loops {names}: which trajectory it stands for is ambiguous. "
            "Pass the stream to minimize itself, the one the Loop returns."
        )
    if not found:
        return value
    ((stream, _),) = found.values()
    # A declared output names it: the minimizer reads it as that output.
    return next(
        (output for output in model._outputs if output._preds[0] is stream), stream
    )


def _returned_by(loop, returned):
    """The stream ``loop`` returns for ``returned``, an output of its body:
    the loop itself, when it has one output."""
    if not loop._return_all_outputs:
        return loop
    index = next(
        index for index, node in enumerate(loop._model._outputs) if node is returned
    )
    return next(output for output in loop._loop_outputs if output._index == index)


def _one_step_input(nodes: list[Node], known: set[int], value, other):
    """``value`` read one sample per step of ``other``: a bare Input the
    model does not read, with the shape of one step of ``other``, a
    trajectory a Loop collects, is an Input of its name with the horizon of
    the trajectory - the model's own, when it reads one of that shape. Any
    other ``value`` is itself."""
    from nnodely.core.modely import _same_sizes

    if (
        not isinstance(value, Input)
        or id(value) in known
        or value.update is not None
        or value.tw is None
    ):
        return value
    shape = getattr(other, "shape", None)
    if (
        shape is None
        or shape.seq_rank != value.shape.seq_rank + 1
        or not _same_sizes(shape.without_outer_seq().tuple, value.shape.tuple)
        or not any(_is_loop(node) and node._collect for node in toposort([other]))
    ):
        return value
    horizon = shape.seq[0]
    rolled = value.shape.with_seq(horizon)
    named = [
        node
        for node in nodes
        if isinstance(node, Input) and not node._hidden and node.name == value.name
    ]
    window = (value.past, value.future)
    for node in named:
        if _same_sizes(node.shape.tuple, rolled.tuple) and (
            (node.past, node.future) == window
        ):
            return node
    if named:
        return value  # an Input of its name, of another shape: refused next
    return Input(
        value.name,
        dim=value.dim,
        seq=tuple(-1 if size is None else size for size in rolled.seq),
        tw=window,
    )


def _refuse_foreign(model, nodes, known: set[int], value, side: str, minimizer: str):
    """Refuse ``value`` if it reads the graph of another model: a stream
    another model computes, a fed-back Input the model does not read, or an
    Input of the name of one the model reads - the data is fed by name."""
    if id(value) in known:
        return
    inputs = {
        node.name: node
        for node in nodes
        if isinstance(node, Input) and not node._hidden
    }
    owners: dict[int, str] | None = None
    stack, seen = [value], set()
    while stack:
        node = stack.pop()
        # A call to a block is a node, but no stream.
        if id(node) in seen or id(node) in known or getattr(node, "_is_value", False):
            continue
        seen.add(id(node))
        reads = (
            f"its {side} {value.name!r}"
            if node is value
            else f"its {side} {value.name!r} reads {node.name!r}, which"
        )
        if isinstance(node, Input):
            # A layer's own state - Integrate, Derivative, Buffer - is no
            # Input of the user's: it starts from its init.
            if node.update is not None and not node._hidden:
                raise ValueError(
                    f"Minimizer {minimizer!r}: {reads} is fed back, and model "
                    f"{model.name!r} does not read it: a state of another graph "
                    "- the step of a ModelyLoop, the body of a Loop - has no "
                    "value in it. Minimize a stream of the model: an output, "
                    "or one it computes."
                )
            twin = inputs.get(node.name)
            if twin is not None and twin is not node:
                raise ValueError(
                    f"Minimizer {minimizer!r}: {reads} is not the Input "
                    f"{node.name!r} model {model.name!r} reads, but another of "
                    "its name - of another graph, such as the step of a "
                    "ModelyLoop - and the data is fed by input name. Use the "
                    f"model's own, model.inputs[{node.name!r}], or give the "
                    "Input a name of its own."
                )
            continue
        if owners is None:
            owners = _graph_owners(model)
        owner = owners.get(id(node))
        if owner is not None:
            raise ValueError(
                f"Minimizer {minimizer!r}: {reads} is a stream of the graph of "
                f"{owner!r} - the step of a ModelyLoop, the body of a Loop, a "
                f"model called as a block - which model {model.name!r} does not "
                "compute. Minimize a stream of the model: an output, which for "
                "an output of a body is its trajectory, or a stream computed "
                "from them; a stream inside a step is minimized declared among "
                "the outputs."
            )
        stack.extend(node._preds)


def _graph_owners(model) -> dict[int, str]:
    """The name of the model whose graph holds each node, by id, for every
    model ``model`` holds - the blocks it calls, the bodies of its Loops,
    theirs too - and every graph a Loop was declared over, such as the step
    of a ModelyLoop, which its body copies."""
    owners: dict[int, str] = {}
    groups = _nodes_by_model(model)
    for held, nodes in groups[1:]:
        for node in nodes:
            owners.setdefault(id(node), held.name)
    for _, nodes in groups:
        for node in nodes:
            if _is_loop(node) and node._step.f is not node._model:
                declared = node._step.f
                for inner in toposort(declared._outputs):
                    owners.setdefault(id(inner), declared.name)
    return owners


def _check_shapes(source, target, minimizer: str) -> None:
    """Refuse a target of another shape than its source, but for a value,
    which broadcasts: a loss would broadcast one over the other and silently
    compare, say, one sample with a whole window."""
    from nnodely.core.modely import _same_sizes

    if not _reads_input(target) or _same_sizes(source.shape.tuple, target.shape.tuple):
        return
    hint = ""
    if isinstance(target, Input) and target.shape.seq_rank < source.shape.seq_rank:
        seq = [-1 if size is None else size for size in source.shape.seq]
        hint = (
            f" Declare {target.name!r} with the sequence of {source.name!r}, "
            f"seq={tuple(seq) if len(seq) > 1 else seq[0]}."
        )
    raise ValueError(
        f"Minimizer {minimizer!r} compares {source.name!r} of shape "
        f"{source.shape} with {target.name!r} of shape {target.shape}. They "
        f"must have the same shape.{hint}"
    )


def _where(stream: Stream) -> str:
    """What ``stream`` is, for the listing of a name several carry: its
    kind, and the call that returns it, for the output of a block."""
    call = getattr(stream, "_pred", None)
    returned = (
        f" returned by {call.name!r} on {[pred.name for pred in call._preds]}"
        if call is not None
        else ""
    )
    kind = type(stream).__name__
    return f"({'an' if kind[0] in 'AEIOU' else 'a'} {kind}{returned})"


def _resolve_minimizer_terms(model) -> list[_MinimizerTerm]:
    """Resolve every minimizer of ``model`` to the graph outputs of its build."""
    assert model.model is not None

    def output_name(minimizer, side):
        # The key the forward pass returns the side under, given by build().
        node = minimizer[side]
        name = model._minimizer_outputs.get(node)
        if name is None:
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
            gain=minimizer.get("gain", 1.0),
        )
        for minimizer in model._minimizers
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
        for minimizer in model._minimizers
        if not _reaches_trainable_weight(minimizer["source"])
        and not _reaches_trainable_weight(minimizer["target"])
    ]
    if len(weightless) == len(model._minimizers):
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
