from nnodely import Input, Output, Fir, Modely, Parameter, Constant
from nnodely.core.layer import Multiply, Add


def test_flatten_1():
    x = Input("x", dim=1)
    param = Parameter("param1", dim=1)
    x_param = x.sw(5) * param
    x_out = Output("x_out", x_param)
    model1 = Modely("model1", inputs=[x], outputs=[x_out])
    model1.build()

    y = Input("y", dim=1)
    y_fir = Fir(out_features=1)([model1([y.sw(5)])])
    y_out = Output("y_out", y_fir)
    model2 = Modely("model2", inputs=[y], outputs=[y_out])
    model2.build()

    z = Input("z", dim=1)
    const_z = Constant("const_z", value=[1.0, 2.0, 3.0, 4.0, 5.0])
    z_fir = Fir(out_features=1)([model2([z.sw(5) + const_z])])
    z_out = Output("z_out", z_fir)
    model3 = Modely("model3", inputs=[z], outputs=[z_out])
    model3.build()


def test_flatten_2():
    x1 = Input("x1")
    x2 = Input("x2")
    x3 = Input("x3")
    mul1 = Multiply("mul1")([x1, x2])
    mul2 = Multiply("mul2")([x2, x3])
    add = Add("add")([mul1, mul2])
    y = Output("y", add)
    m1 = Modely(name="m1", inputs=[x1, x2], outputs=[y])

    m1.flatten()

    a1 = Input("a1")
    a2 = Input("a2")
    m1c = m1([a1, a2])
    m2 = Modely("m2", [a1, a2], [m1c, m1c])

    m2.flatten()

    c1 = Input("c1")
    c2 = Input("c2")
    m2c1, m2c2 = m2([c1, c2])
    d1 = Output("d1", m2c1)
    d2 = Output("d2", m2c2)
    m3 = Modely("m3", [c1, c2], [d1, d2])

    m3.flatten()

    e1 = Input("e1")
    e2 = Input("e2")
    m2e1, m2e2 = m2([e1, e2])
    m3e1, m3e2 = m3([m2e1, m2e2])
    f1 = Output("f1", m3e1)
    f2 = Output("f2", m3e2)
    m4 = Modely("m4", [e1, e2], [f1, f2])

    m4.flatten()

    g1 = Input("g1")
    m4g1, m4g2 = m4([g1, g1])
    h1 = Output("h1", m4g1)
    h2 = Output("h2", m4g2)
    m5 = Modely("m5", [g1], [h1, h2])

    m5.flatten()
