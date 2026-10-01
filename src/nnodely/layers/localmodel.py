import inspect

from collections.abc import Callable

from nnodely.basic.relation import NeuObj, Stream
from nnodely.layers.arithmetic import add_relation_name
from nnodely.layers.part import Select
from nnodely.support.jsonutils import merge
from nnodely.support.utils import check, enforce_types

localmodel_relation_name = "LocalModel"


def _signature_len(fn) -> int:
    return len(inspect.signature(fn).parameters)


def _apply(fn, x):
    """Invoke ``fn`` on a Stream or on the unpacked elements of a tuple."""
    return fn(*x) if type(x) is tuple else fn(x)


class LocalModel(NeuObj):
    """
    Represents a Local Model relation in the neural network model.

    Parameters
    ----------
    input_function : Callable, optional
        A callable function to process the inputs.
    output_function : Callable, optional
        A callable function to process the outputs.
    pass_indexes : bool, optional
        A boolean indicating whether to pass indexes to the functions. Default is False.

    Attributes
    ----------
    relation_name : str
        The name of the relation.
    pass_indexes : bool
        A boolean indicating whether to pass indexes to the functions.
    input_function : Callable
        The function to process the inputs.
    output_function : Callable
        The function to process the outputs.

    Examples
    --------

    .. include:: /examples_basics/layer_module_ex/localmodel.rst
    """

    @enforce_types
    def __init__(
        self,
        input_function: Callable | None = None,
        output_function: Callable | None = None,
        *,
        pass_indexes: bool = False,
    ):
        self.relation_name = localmodel_relation_name
        self.pass_indexes = pass_indexes
        self.input_function = input_function
        self.output_function = output_function
        super().__init__(localmodel_relation_name + str(NeuObj.count))
        self.json["Functions"][self.name] = {}

    @enforce_types
    def __call__(self, inputs: Stream | tuple, activations: Stream | tuple = None):
        if type(activations) is not tuple:
            activations = (activations,)

        in_func = self.input_function
        check(
            in_func is not None or type(inputs) is not tuple,
            TypeError,
            "The input cannot be a tuple without input_function",
        )

        # ``input_function`` output is reusable across cells iff the same
        # callable is invoked with the same arguments for every cell:
        # ``pass_indexes`` False and not a zero-arg factory.
        shared_out_in = None
        if (
            in_func is not None
            and not self.pass_indexes
            and _signature_len(in_func) > 0
        ):
            shared_out_in = _apply(in_func, inputs)

        select_cache: dict[tuple[int, int], Stream] = {}

        def cached_select(act_idx: int, i: int) -> Stream:
            cached = select_cache.get((act_idx, i))
            if cached is None:
                cached = Select(activations[act_idx], i)
                select_cache[(act_idx, i)] = cached
            return cached

        cells: list[Stream] = []
        self._build_cells(
            activations,
            inputs,
            cells,
            cached_select,
            shared_out_in,
            prefix=None,
            idx_list=[],
            depth=0,
        )
        return self._nary_add(cells)

    def _build_cells(
        self,
        activations,
        inputs,
        cells,
        cached_select,
        shared_out_in,
        *,
        prefix,
        idx_list,
        depth,
    ):
        # ``prefix`` is the cached product of Selects for indices [0..depth);
        # sibling subtrees reuse the same Stream, turning the per-cell K-1
        # chain of activation muls into an incremental tree build.
        if depth == len(activations):
            out_in = (
                shared_out_in
                if shared_out_in is not None
                else self._apply_fn(
                    self.input_function,
                    inputs,
                    idx_list,
                )
            )
            cells.append(
                self._apply_fn(
                    self.output_function,
                    out_in * prefix,
                    idx_list,
                )
            )
            return

        for i in range(activations[depth].dim["dim"]):
            sel = cached_select(depth, i)
            new_prefix = sel if prefix is None else prefix * sel
            self._build_cells(
                activations,
                inputs,
                cells,
                cached_select,
                shared_out_in,
                prefix=new_prefix,
                idx_list=idx_list + [i],
                depth=depth + 1,
            )

    def _apply_fn(self, fn, x, idx_list):
        # Dispatch on the user function's signature:
        # zero-arg ``fn`` is a factory producing a fresh cell callable;
        # ``pass_indexes`` makes the factory idx-dependent; otherwise ``fn``
        # is already the cell callable (shared params across cells).
        if fn is None:
            return x
        if _signature_len(fn) == 0:
            cell_fn = fn()
        elif self.pass_indexes:
            cell_fn = fn(idx_list)
        else:
            cell_fn = fn
        return _apply(cell_fn, x)

    @staticmethod
    def _nary_add(cells: list[Stream]) -> Stream:
        # Equivalent to ``cells[0] + cells[1] + ... + cells[-1]``, but folded
        # into a single ``Add`` relation. ``Add_Layer.forward(*inputs)`` does
        # the same left-to-right fold at runtime, so the float result is
        # bit-exact to the chained binary version while the build avoids
        # ``N-1`` intermediate Streams (each of which would deep-copy a
        # growing JSON).
        if len(cells) == 1:
            return cells[0]

        combined = cells[0].json
        for c in cells[1:]:
            combined = merge(combined, c.json)

        name = add_relation_name + str(Stream.count)
        new_stream = Stream(name, combined, cells[0].dim)
        new_stream.json["Relations"][name] = [
            add_relation_name,
            [c.name for c in cells],
        ]
        return new_stream
