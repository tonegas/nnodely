import os

from nnodely import Input, Output, Fir, Modely, Parameter, Constant, Cos, Sin


def test_plot_and_export_html(tmp_path):
    x = Input("x", dim=1)
    out = Output("out", Fir(out_features=1)([x.sw(3)]))
    model = Modely("plot_model", inputs=[x], outputs=[out])
    model.build()

    html_path = model.export_html(out_dir=tmp_path, filename="plot_model")
    assert html_path.endswith(".html")
    assert (tmp_path / "plot_model.html").exists()


def test_model_composed_visualization(tmp_path):
    # ------- Model definition and building -------
    x = Input("x", dim=1)
    y = Input("y", dim=1)

    x_stream = x.sw(10)
    y_stream = y.sw(10)

    fir = Fir(out_features=2)
    result_fir = fir([x_stream + y_stream])

    x_out = Output("x_pred", result_fir)
    model1 = Modely("model1", inputs=[x, y], outputs=[x_out])
    model1.build()

    # ------- Model composition -------
    z = Input("z", dim=1)
    z_stream = z.sw(10)
    z_fir = Fir(out_features=1)([model1([z_stream, z_stream])])
    z_out = Output("z_pred", z_fir)
    model2 = Modely("composed_model", inputs=[z], outputs=[z_out])
    model2.build()

    # ------- Model visualization -------
    model1.plot(to_file=os.path.join(tmp_path, "model1.png"))
    model2.plot(to_file=os.path.join(tmp_path, "model2.png"))

    # ------- Model export to HTML -------
    model1.export_html(out_dir=tmp_path, filename="model1")
    model2.export_html(out_dir=tmp_path, filename="model2")


def test_multiple_model_composed_visualization(tmp_path):
    # ------- Model Flatten and Visualization -------
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

    # ------- Visualize the Flatten model -------
    model3.export_html(out_dir=tmp_path, filename="model3")
    model3.plot(to_file=os.path.join(tmp_path, "model3_standard.png"))
    model3.plot(to_file=os.path.join(tmp_path, "model3_flattened.png"), flatten=True)


# def test_visualize_loop(tmp_path):
#     # ------- Model with closed loop connections -------
#     x = Input(name="x", dim=1)
#     y = Input(name="y", dim=1)
#     r1 = x.sw(1) + y.sw(1)
#     out1 = Output("out1", r1)
#     model1 = Modely(name="model1", inputs=[x, y], outputs=[out1])
#     model1.build()

#     z = Input(name="z", dim=1, seq=5)
#     const = Constant("const", value=2.0)
#     r2 = z.sw(1) * const
#     loop_fn = Loop(f=model1, closed_loop={out1: z})
#     out = Output("out", loop_fn([z, r2]))
#     model = Modely(name="model", inputs=[z], outputs=[out])
#     model.build()

#     # ------- Model visualization -------
#     model1.plot(to_file=tmp_path / "model1.png")
#     model.plot(to_file=tmp_path / "model2.png")

#     # ------- Model export to HTML -------
#     model1.export_html(out_dir=tmp_path, filename="model1")
#     model.export_html(out_dir=tmp_path, filename="model2")


def test_visualize_high_level_blocks(tmp_path):
    class Tangent:
        def __init__(self, name: str | None = "Tangent"):
            self.name = name

        def __call__(self, inputs):
            ret = Sin()(inputs) / Cos()(inputs)
            out = Output(name=f"{self.name}_out", stream=ret)
            return Modely(name=f"{self.name}", inputs=inputs, outputs=[out])(inputs)

    input = Input("input")
    tan = Tangent(name="tangent")([input])
    output = Output("output", tan)

    model = Modely(name="model", inputs=[input], outputs=[output])
    model_flat = model.flatten()

    model.plot(to_file=os.path.join(tmp_path, "model_tangent.png"))
    model_flat.plot(to_file=os.path.join(tmp_path, "model_tangent_flat.png"))

    model.export_html(out_dir=tmp_path, filename="model_tangent")
