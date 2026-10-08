"""Input-state-output port-Hamiltonian field, as a block of one model.

The form is the one shared by the port-Hamiltonian literature::

    dx/dt = [J(x) - R(x)] dH/dx + G(x) u
    y     = G(x)^T dH/dx

with `J = -J^T` and `R = R^T >= 0`. What makes the block worth having is that
neither constraint is a penalty the optimizer can trade away: `J` is assembled
from a skew basis and `R` as `L L^T`, so for *every* value of the parameters the
power balance

    dH/dt = -dH/dx^T R dH/dx + y^T u <= y^T u

holds, and the learned system can only take energy in through the port. `y` is
collocated with `u` by construction for the same reason - it is what makes
`y^T u` a power rather than an arbitrary readout.

Every matrix is carried as the flat vector of its free entries and expanded
against a constant basis, `M = sum_k c_k B_k`, so the structure lives in the
basis rather than in a constraint that has to be re-imposed. The same vector is
produced either by a `Parameter` (a constant matrix) or by an MLP of the state
(a state-dependent one), and the algebra downstream does not know the difference.

`dH/dx` is a `Derivative` with respect to the state `Input`, so the field is
evaluated at that input's current sample. The block computes the field and the
output, nothing else: `Ode` integrates it and `Loop` rolls it out.
"""

from __future__ import annotations

from collections.abc import Sequence

import keras
import numpy as np

from nnodely.core.dag import next_name
from nnodely.core.layer import Layer
from nnodely.core.stream import Stream
from nnodely.layers.activations import Tanh
from nnodely.layers.derivative import Derivative
from nnodely.layers.input import Input
from nnodely.layers.linear import Linear
from nnodely.layers.parameter import Parameter
from nnodely.layers.time_ops import SampleWindow


def _basis(kind: str, rows: int, columns: int) -> np.ndarray:
    """Constant [k, rows, columns] tensor whose k slices span the matrix family.

    A matrix is then the contraction of its flat entries with this tensor, which
    is why the structure cannot be lost: an antisymmetric basis can only sum to
    an antisymmetric matrix.
    """
    if kind == "skew":
        pairs = [(i, j) for i in range(rows) for j in range(i + 1, rows)]
    elif kind == "lower":
        pairs = [(i, j) for i in range(rows) for j in range(i + 1)]
    else:
        pairs = [(i, j) for i in range(rows) for j in range(columns)]

    tensor = np.zeros((len(pairs), rows, columns), dtype=np.float32)
    for index, (i, j) in enumerate(pairs):
        tensor[index, i, j] = 1.0
        if kind == "skew":
            tensor[index, j, i] = -1.0
    return tensor


def _entry_count(kind: str, rows: int, columns: int) -> int:
    if kind == "skew":
        return rows * (rows - 1) // 2
    if kind == "lower":
        return rows * (rows + 1) // 2
    return rows * columns


# Streams are laid out [batch, dim, time, *seq], so the contractions run over
# the axis after the batch and carry time and sequence along in the ellipsis.


@keras.saving.register_keras_serializable(package="nnodely")
class StructuredFieldImpl(keras.layers.Layer):
    """[J - R] dH + G u, with J skew and R = L L^T for any parameter value."""

    def __init__(self, state_dim: int, input_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.state_dim = int(state_dim)
        self.input_dim = int(input_dim)
        self.skew = _basis("skew", self.state_dim, self.state_dim)
        self.lower = _basis("lower", self.state_dim, self.state_dim)
        self.port = _basis("full", self.state_dim, self.input_dim)

    def call(self, xs):
        gradient, drive, skew_flat, lower_flat, port_flat = xs

        conservative = keras.ops.einsum(
            "nk...,kab,nb...->na...", skew_flat, self.skew, gradient
        )
        # R dH is L (L^T dH): forming R explicitly would be the same number of
        # contractions and one more chance to lose the symmetry.
        scaled = keras.ops.einsum(
            "nk...,kba,nb...->na...", lower_flat, self.lower, gradient
        )
        dissipative = keras.ops.einsum(
            "nk...,kab,nb...->na...", lower_flat, self.lower, scaled
        )
        driven = keras.ops.einsum(
            "nk...,kab,nb...->na...", port_flat, self.port, drive
        )
        return conservative - dissipative + driven

    def get_config(self):
        config = super().get_config()
        config.update({"state_dim": self.state_dim, "input_dim": self.input_dim})
        return config


@keras.saving.register_keras_serializable(package="nnodely")
class CollocatedOutputImpl(keras.layers.Layer):
    """G^T dH - the output collocated with the port, so y^T u is a power."""

    def __init__(self, state_dim: int, input_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.state_dim = int(state_dim)
        self.input_dim = int(input_dim)
        self.port = _basis("full", self.state_dim, self.input_dim)

    def call(self, xs):
        gradient, port_flat = xs
        return keras.ops.einsum(
            "nk...,kab,na...->nb...", port_flat, self.port, gradient
        )

    def get_config(self):
        config = super().get_config()
        config.update({"state_dim": self.state_dim, "input_dim": self.input_dim})
        return config


class _Structured(Layer):
    """Common plumbing for the two equations, configured then called on streams."""

    impl = None

    def __init__(self, state_dim, input_dim, name=None):
        self.state_dim = int(state_dim)
        self.input_dim = int(input_dim)
        super().__init__(
            name=name, state_dim=self.state_dim, input_dim=self.input_dim
        )

    def build_layer(self):
        return self.impl(  # type: ignore[misc]
            state_dim=self.state_dim, input_dim=self.input_dim, name=self.name
        )


class StructuredField(_Structured):
    impl = StructuredFieldImpl


class CollocatedOutput(_Structured):
    impl = CollocatedOutputImpl


class PortHamiltonian:
    """Port-Hamiltonian field and collocated output, built from `x` and `u`::

        field = PortHamiltonian(state_dim=4, input_dim=1, hamiltonian=[64, 64])
        dx, y = field(x.last(), u.last())

    `hamiltonian` is the list of hidden widths of the MLP holding `H(x)`, which
    is the only part of the dynamics learned without structure. `J`, `R` and `G`
    each take either ``"constant"`` - one matrix for the whole state space - or
    a list of hidden widths, in which case that matrix is an MLP of the state.
    A constant `J`, `R` and `G` over a nonlinear `H` is already a nonlinear
    system, and is the right default: state-dependent matrices add freedom that
    a single trajectory usually cannot identify.

    `x` has to be the current sample of the state `Input`, because `dH/dx` is a
    `Derivative` with respect to that input. A stream computed from the state is
    not one: an intermediate stage of a multi-stage `Ode` method, say. Integrate
    the block with ``Ode(..., method="euler")`` (rolled out by a `Loop`), or put
    it in the body of an `OdeNet` for a higher-order or adaptive method, with
    `u` held over the step as a state of zero derivative.

    The block may be called as often as needed and every call shares one set of
    weights. :meth:`energy` gives `H(x)` itself, with the same weights.
    """

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        hamiltonian: Sequence[int] = (64, 64),
        activation: type[Layer] = Tanh,
        J: str | Sequence[int] = "constant",
        R: str | Sequence[int] = "constant",
        G: str | Sequence[int] = "constant",
        name: str | None = None,
    ):
        if state_dim < 2:
            raise ValueError(
                f"PortHamiltonian needs at least two states for J to be non-trivial, "
                f"got {state_dim}."
            )
        if not hamiltonian:
            raise ValueError(
                "PortHamiltonian needs at least one hidden layer for H: a linear "
                "Hamiltonian has a constant gradient and no dynamics."
            )

        self.name = name or next_name("PortHamiltonian")
        self.state_dim = int(state_dim)
        self.input_dim = int(input_dim)
        self.activation = activation

        # A bias on H would not reach dH/dx, and so would never be trained.
        self.hamiltonian = self._mlp(hamiltonian, 1, "H", output_bias=False)
        self.J = self._build_source(J, "skew", "J")
        self.R = self._build_source(R, "lower", "R")
        self.G = self._build_source(G, "full", "G")

    def _mlp(self, widths: Sequence[int], entries: int, tag: str, output_bias=True):
        layers = []
        for index, width in enumerate(widths):
            layers.append(Linear(out_features=width, name=f"{self.name}_{tag}_{index}"))
            layers.append(self.activation())
        layers.append(
            Linear(
                out_features=entries,
                use_bias=output_bias,
                name=f"{self.name}_{tag}_out",
            )
        )
        return layers

    def _build_source(self, spec, kind: str, tag: str):
        """A Parameter for a constant matrix, or the layers of an MLP of the state.

        Either way the result is the flat vector of the matrix's free entries,
        so the two cases are indistinguishable downstream.
        """
        entries = _entry_count(kind, self.state_dim, self.input_dim)
        if isinstance(spec, str):
            if spec != "constant":
                raise ValueError(
                    f"PortHamiltonian {tag} must be 'constant' or a list of hidden "
                    f"widths, got {spec!r}."
                )
            return Parameter(f"{self.name}_{tag}", dim=entries)
        return self._mlp(spec, entries, tag)

    def _apply(self, source, state: Stream) -> Stream:
        if isinstance(source, Parameter):
            return source
        stream = state
        for layer in source:
            stream = layer(stream)
        return stream

    def energy(self, x: Stream) -> Stream:
        """H(x), the energy the field is the gradient flow of."""
        return self._apply(self.hamiltonian, x)

    def __call__(self, x: Stream, u: Stream):
        for stream, expected, label in ((x, self.state_dim, "x"), (u, self.input_dim, "u")):
            if not isinstance(stream, Stream):
                raise TypeError(
                    f"PortHamiltonian {label} must be a Stream, got {type(stream).__name__}."
                )
            if tuple(stream.dim) != (expected,):
                raise ValueError(
                    f"PortHamiltonian {label} must have dim ({expected},), got "
                    f"{tuple(stream.dim)}."
                )
        state = x.preds[0] if isinstance(x, SampleWindow) else None
        if not (isinstance(state, Input) and (x.past, x.future) == (1, 0)):
            raise TypeError(
                f"PortHamiltonian differentiates H with respect to the state Input, "
                f"so x must be that input's current sample, x.last(); got "
                f"{x.name!r}. A stream computed from the state, such as a stage of "
                f"a multi-stage Ode method, has no input to differentiate against: "
                f"integrate with Ode(method='euler'), or with a Loop or OdeNet body."
            )
        if state.shape.time != 1:
            raise ValueError(
                f"PortHamiltonian: the state input {state.name!r} has a window of "
                f"{state.shape.time} samples, but dH/dx is read at its current "
                f"sample alone, so the input must carry no other window."
            )

        gradient = Derivative(order=1, respect_to=state)(self.energy(x))
        G = self._apply(self.G, x)
        field = StructuredField(self.state_dim, self.input_dim)(
            [gradient, u, self._apply(self.J, x), self._apply(self.R, x), G]
        )
        measurement = CollocatedOutput(self.state_dim, self.input_dim)(
            [gradient, G]
        )
        return field, measurement
