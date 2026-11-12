import sys

from nnodely import Input, Output, Fir, Modely, Parameter, Constant
from nnodely.core.dag import toposort
from nnodely.core.layer import Multiply, Add
from nnodely.core.stream import Node


def test_flatten_1():
    x = Input("x", dim=1, tw=5)
    param = Parameter("param1", dim=1)
    x_param = x * param
    x_out = Output("x_out", x_param)
    model1 = Modely("model1", inputs=[x], outputs=[x_out])
    model1.build()

    y = Input("y", dim=1, tw=5)
    y_fir = Fir(features=1)([model1([y])])
    y_out = Output("y_out", y_fir)
    model2 = Modely("model2", inputs=[y], outputs=[y_out])
    model2.build()

    z = Input("z", dim=1, tw=5)
    const_z = Constant("const_z", value=[1.0, 2.0, 3.0, 4.0, 5.0], tw=5)
    z_fir = Fir(features=1)([model2([z + const_z])])
    z_out = Output("z_out", z_fir)
    model3 = Modely("model3", inputs=[z], outputs=[z_out])
    model3.build()


def test_flatten_2():
    x1 = Input("x1", tw=1)
    x2 = Input("x2", tw=1)
    mul1 = Multiply("mul1")([x1, x2])
    mul2 = Multiply("mul2")([x2, x1])
    add = Add("add")([mul1, mul2])
    y = Output("y", add)
    m1 = Modely(name="m1", inputs=[x1, x2], outputs=[y])

    m1.flatten()

    a1 = Input("a1", tw=1)
    a2 = Input("a2", tw=1)
    m1c = m1([a1, a2])
    m2 = Modely("m2", [a1, a2], [m1c, m1c])

    m2.flatten()

    c1 = Input("c1", tw=1)
    c2 = Input("c2", tw=1)
    m2c1, m2c2 = m2([c1, c2])
    d1 = Output("d1", m2c1)
    d2 = Output("d2", m2c2)
    m3 = Modely("m3", [c1, c2], [d1, d2])

    m3.flatten()

    e1 = Input("e1", tw=1)
    e2 = Input("e2", tw=1)
    m2e1, m2e2 = m2([e1, e2])
    m3e1, m3e2 = m3([m2e1, m2e2])
    f1 = Output("f1", m3e1)
    f2 = Output("f2", m3e2)
    m4 = Modely("m4", [e1, e2], [f1, f2])

    m4.flatten()

    g1 = Input("g1", tw=1)
    m4g1, m4g2 = m4([g1, g1])
    h1 = Output("h1", m4g1)
    h2 = Output("h2", m4g2)
    m5 = Modely("m5", [g1], [h1, h2])

    m5.flatten()


def test_toposort_is_depth_first_each_node_after_its_predecessors():
    # A node shared by two readers is sorted once, before both; each node's
    # predecessors are visited in their order, and a root already reached
    # through another adds nothing.
    a, b, c, d = (Node(f"topo_{n}") for n in "abcd")
    b._preds = [a]
    c._preds = [a, b]
    d._preds = [c, b]
    assert toposort([d, b]) == [a, b, c, d]
    assert toposort([b, d]) == [a, b, c, d]
    e = Node("topo_e", preds=[b, a])
    assert toposort([e, d]) == [a, b, e, c, d]


def test_a_graph_deeper_than_the_recursion_limit_is_sorted_and_flattened():
    # Sorting and inlining a graph walk it without recursion: a chain deeper
    # than the recursion limit is declared, sorted and flattened.
    depth = 3 * sys.getrecursionlimit()
    x = Input("deep_x", tw=1)
    stream = x
    for _ in range(depth):
        stream = stream + 0.0
    model = Modely("deep", inputs=[x], outputs=[Output("deep_y", stream)])
    assert model._order[0] is x and model._order[-1] is model._outputs[0]
    assert len(model._order) == depth + 3  # the Input, the 0.0, the sums, the Output
    flat = model.flatten()
    assert len(flat._order) == len(model._order)
