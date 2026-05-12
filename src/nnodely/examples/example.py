import numpy as np

from nnodely.core.dag import flatten

from nnodely.core.modely import Modely
from nnodely.layers.input import Input
from nnodely.layers.output import Output
from nnodely.layers.fir import Fir
from nnodely.layers.fuzzify import Fuzzify
from nnodely.layers.localmodel import LocalModel
from nnodely.core.dataloader import DataLoader
from nnodely.layers.parameter import Parameter
from nnodely.layers.constant import Constant
from nnodely.layers.concatenate import Concatenate, TimeConcatenate
from nnodely.layers.activations import (
    ReLU,
    ELU,
    LeakyReLU,
    PReLU,
    Sigmoid,
    Tanh,
    Softmax,
    Swish,
    GELU,
    Softplus,
)
from nnodely.layers.trigonometric import Sin, Cos, Tan, Asin, Acos, Atan
from nnodely.core.layer import Add, Multiply

import os

# os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ["KERAS_BACKEND"] = "torch"


def main(example=1):
    if example == 1:
        # ------- Model definition and building -------
        x = Input("x", dim=1)
        y = Input("y", dim=1)

        window_size = 10
        x_stream = x.sw(window_size)
        y_stream = y.sw(window_size)

        fir = Fir(out_features=2)
        result_fir = fir([x_stream + y_stream])

        x_out = Output("x_pred", result_fir)
        model1 = Modely("model1", inputs=[x, y], outputs=[x_out])
        print(model1)
        model1.build()

        # ------- Model composition -------
        z = Input("z", dim=1)
        z_stream = z.sw(window_size)
        z_fir = Fir(out_features=1)(model1([z_stream, z_stream]))
        z_out = Output("z_pred", z_fir)
        model2 = Modely("composed_model", inputs=[z], outputs=[z_out])
        print(model2)
        model2.build()

        # ------- Model inference -------
        batch_size = 4
        dummy_input_x = np.ones((batch_size, window_size, 1), dtype=np.float32)
        dummy_input_y = np.ones((batch_size, window_size, 1), dtype=np.float32)
        dummy_input_z = np.ones((batch_size, window_size, 1), dtype=np.float32)

        result1 = model1([dummy_input_x, dummy_input_y])
        print("Model1 - Input shape:", dummy_input_x.shape)
        print("Model1 - Output shape:", result1["x_pred"].shape)
        print("Model1 - Output:", result1)

        result2 = model2([dummy_input_z])
        print("Model2 - Input shape:", dummy_input_z.shape)
        print("Model2 - Output shape:", result2["z_pred"].shape)
        print("Model2 - Output:", result2)

        # ------- Save/load model inference -------
        model1.save("results/model1")
        model2.save("results/model2")
        loaded_model1 = Modely.load("results/model1")
        loaded_model2 = Modely.load("results/model2")
        loaded_model1.build()
        loaded_model2.build()
        loaded_result1 = loaded_model1([dummy_input_x, dummy_input_y])
        loaded_result2 = loaded_model2([dummy_input_z])
        print("Loaded Model1 - Output:", loaded_result1)
        print("Loaded Model2 - Output:", loaded_result2)

        # ------- Model visualization -------
        model1.plot(to_file="html/model1.png")
        model2.plot(to_file="html/model2.png")

        # ------- Model export to HTML -------
        model1.export_html(out_dir="html", filename="model1")
        model2.export_html(out_dir="html", filename="model2")

    if example == 2:
        # ------- Model definition and training -------
        x = Input("x", dim=1)
        y = Input("y", dim=1)
        x_fir = Fir(out_features=1)([x.sw(5)])
        y_fir = Fir(out_features=1)([y.sw(5)])

        x_out = Output("fir_pred", x_fir + y_fir)
        model1 = Modely("linear_fit", inputs=[x, y], outputs=[x_out])

        # ------- Define loss and minimizer -------
        model1.minimize(
            "error", source=x_out, target=Input("x_target", dim=1).sw(1), loss="mse"
        )
        model1.build()

        # ------ Load dataset -------
        data_train = DataLoader(
            model1,
            format={"x": "data_1", "y": "data_2", "x_target": "data_3"},
            source=os.path.join("src", "nnodely", "examples", "data"),
        )

        # ------ Train the model -------
        model1.train(train_data=data_train, epochs=60, batch_size=4)

        # ------ Remove minimizer and retrain with multi loss -------
        model1.remove_minimizer("error")
        model1.minimize(
            "error_fir_x",
            source=x_fir,
            target=Input("x_target", dim=1).sw(1),
            loss="mse",
        )
        model1.minimize(
            "error_fir_y", source=y_fir, target=None, loss="mse"
        )  ## with target=None, the loss will be minimized to zero

        model1.build()  # Rebuild the model after removing the minimizer otherwise the training loop will still try to compute the loss and update the model based on it, even if it's not used for training anymore
        model1.train(train_data=data_train, epochs=60, batch_size=4)

        # ------ Visualize the model -------
        model1.plot(to_file="html/model1_trained.png")

        # ------ Remove one minimizer and retrain with a constant value -------
        model1.remove_minimizer("error_fir_y")
        model1.minimize(
            "error_fir_y", source=y_fir, target=3.0, loss="mse"
        )  ## this will minimize the difference between y_fir and the constant value 3.0, effectively training the model to make y_fir close to 3.0
        model1.build()
        model1.train(train_data=data_train, epochs=60, batch_size=4)

    if example == 3:
        # ------- Model definition with Parameters and Constants -------
        x = Input("x", dim=1)
        param = Parameter("param1", value=[1.0])
        const = Constant("const1", value=[1.0])
        x_param = x.sw(1) * param + const
        x_out = Output("x_out", x_param)
        model = Modely("model", inputs=[x], outputs=[x_out])
        model.minimize(
            "error", source=x_out, target=Input("x_target", dim=1).sw(1), loss="mse"
        )
        model.build()

        # ------ Model inference -------
        batch_size = 4
        dummy_input_x = np.ones((batch_size, 1, 1), dtype=np.float32)
        result = model({"x": dummy_input_x})
        print("Output before training:", result)

        # ------ Create a simple dataset and train the model -------

        true_param = 3.5
        dataframe = {
            "x": np.ones((100, 1, 1), dtype=np.float32),
            "x_target": np.ones((100, 1, 1), dtype=np.float32) * true_param,
        }
        data_train = DataLoader(model, source=dataframe)

        model.train(train_data=data_train, epochs=50, batch_size=16)
        print(
            f"Learned parameter value: {param.value_numpy}, True parameter value: {true_param - const.value_numpy}"
        )
        print(
            f"Constant value after training: {const.value_numpy}, True constant value: {1.0}"
        )

        # ------ Inference after training -------
        result_after_training = model({"x": dummy_input_x})
        print("Output after training:", result_after_training)

        # ------ Inference after Save/Load -------
        model.save("results/model_with_param")
        model_loaded = Modely.load("results/model_with_param")
        model_loaded.build()
        result_loaded = model_loaded({"x": dummy_input_x})
        print("Output after loading the model:", result_loaded)

        # ------ Visualize the model -------
        model.plot(to_file="html/model_with_param.png")
        model.export_html(out_dir="html", filename="model_with_param")

    if example == 4:
        # ------- Model Flatten and Visualization -------
        x = Input("x", dim=1)
        param = Parameter("param1", dim=1)
        x_param = x.sw(5) * param
        x_out = Output("x_out", x_param)
        model1 = Modely("model1", inputs=[x], outputs=[x_out])
        model1.build()

        y = Input("y", dim=1)
        y_fir = Fir(out_features=1)(model1([y.sw(5)]))
        y_out = Output("y_out", y_fir)
        model2 = Modely("model2", inputs=[y], outputs=[y_out])
        model2.build()

        z = Input("z", dim=1)
        const_z = Constant("const_z", value=[1.0, 2.0, 3.0, 4.0, 5.0])
        z_fir = Fir(out_features=1)(model2([z.sw(5) + const_z]))
        z_out = Output("z_out", z_fir)
        model3 = Modely("model3", inputs=[z], outputs=[z_out])
        model3.build()

        # ------- Visualize the Flatten model -------
        model3.export_html(out_dir="html", filename="model3")
        model3.plot(to_file="html/model3_standard.png")
        model3.plot(to_file="html/model3_flattened.png", flatten=True)

    if example == 5:
        # ------- Model with closed loop connections -------
        from nnodely.layers.loop import Loop

        x = Input(name="x", dim=1)
        y = Input(name="y", dim=1)
        r1 = x.sw(1) + y.sw(1)
        out1 = Output("out1", r1)
        model1 = Modely(name="model1", inputs=[x, y], outputs=[out1])
        model1.build()

        z = Input(name="z", dim=1, seq=5)
        const = Constant("const", value=2.0)
        r2 = z.sw(1) * const
        loop_fn = Loop(f=model1, closed_loop={out1: z})
        out = Output("out", loop_fn([z, r2]))
        model = Modely(name="model", inputs=[z], outputs=[out])
        model.build()

        # ------- Model inference -------
        batch_size = 1
        dummy_input_x = np.ones((batch_size, 1, 1), dtype=np.float32)
        dummy_input_y = np.ones((batch_size, 1, 1), dtype=np.float32)
        dummy_input_z = np.ones((batch_size, 5, 1, 1), dtype=np.float32)

        result1 = model1({"x": dummy_input_x, "y": dummy_input_y})
        print("Model1 - Input shape:", dummy_input_x.shape)
        print("Model1 - Output shape:", result1["out1"].shape)
        print("Model1 - Output:", result1)

        result2 = model({"z": dummy_input_z})
        print("Model2 - Input shape:", dummy_input_z.shape)
        print("Model2 - Output shape:", result2["out"].shape)
        print("Model2 - Output:", result2)

        # ------- Model visualization -------
        model1.plot(to_file="html/model1.png")
        model.plot(to_file="html/model2.png")

        # ------- Model export to HTML -------
        model1.export_html(out_dir="html", filename="model1")
        model.export_html(out_dir="html", filename="model2")

    if example == 6:
        pass
        # # ------- Model with time window partitioning, Past and Future -------
        # x = Input("x", dim=1)
        # y = Input("y", dim=1)
        # z = Input("z", dim=1)
        # r1 = x.sw(1) + y.sw(1)
        # out1 = Output("out1", r1)
        # model1 = Modely("model1", outputs=out1)
        # model1.build()

        # r2 = Fir(out_features=1)(z.sw(3))
        # out2 = Output("out2", model1({"x": r2, "y": z.sw(1)}))
        # model2 = Modely("model2", outputs=out2)
        # model2.build()

        # k = Input("k", dim=1)
        # f = Input("f", dim=1)
        # r3 = model2({"z": k.sw(3)}) + model2({"z": f.sw(3)})
        # out3 = Output("out3", r3)
        # model3 = Modely("model3", outputs=out3)
        # model3.build()
        # pass

    if example == 7:
        # ------- Model Export and Import -------
        from nnodely.layers.loop import Loop

        x = Input(name="x", dim=1)
        y = Input(name="y", dim=1)
        r1 = x.sw(1) + y.sw(1)
        out1 = Output("out1", r1)
        model1 = Modely(name="model1", inputs=[x, y], outputs=[out1])
        model1.build()

        z = Input(name="z", dim=1, seq=5)
        const = Constant("const", value=2.0)
        r2 = z.sw(1) * const
        loop_fn = Loop(f=model1, closed_loop={out1: z})
        out = Output("out", loop_fn([z, r2]))
        model = Modely(name="model", inputs=[z], outputs=[out])
        model.build()

        model.save(filename="results/model_k")

        new_model = Modely.load(filename="results/model_k")
        new_model.build()

        # ------- New model inference -------
        batch_size = 1
        dummy_input_x = np.ones((batch_size, 1, 1), dtype=np.float32)
        dummy_input_y = np.ones((batch_size, 1, 1), dtype=np.float32)
        dummy_input_z = np.ones((batch_size, 5, 1, 1), dtype=np.float32)

        result2 = new_model({"z": dummy_input_z})
        print("Model2 - Input shape:", dummy_input_z.shape)
        print("Model2 - Output shape:", result2["out"].shape)
        print("Model2 - Output:", result2)

    if example == 8:
        # ------- Fuzzify Block -------
        x = Input("x", dim=1)

        x_stream = x.sw(1)

        fuzzy = Fuzzify(centers=[0.0, 0.5, 1.0], function="rectangular")

        x_out = Output("x_pred", fuzzy([x_stream]))
        model1 = Modely("model1", inputs=[x], outputs=[x_out])
        print(model1)
        model1.build()

        # ------- Model inference -------
        dummy_input_x = np.array([[[-7]], [[0.5]], [[0.8]]], dtype=np.float32)
        result1 = model1({"x": dummy_input_x})
        print("Model1 - Input shape:", dummy_input_x.shape)
        print("Model1 - Output shape:", result1["x_pred"].shape)
        print("Model1 - Output:", result1)

        # ------- Save/load model inference -------
        model1.save("results/model_fuzzify")
        loaded_model1 = Modely.load("results/model_fuzzify")
        loaded_model1.build()
        loaded_result1 = loaded_model1({"x": dummy_input_x})
        print("Loaded Model1 - Output shape:", loaded_result1["x_pred"].shape)
        print("Loaded Model1 - Output:", loaded_result1)

    if example == 9:
        # ------- High-level Blocks (Local Models) with Multi-inputs -------
        x = Input("x", dim=1)
        # y = Input("y", dim=5)
        # z = Input("z", dim=5)
        k = Input("k", dim=1)
        g = Constant("gravity", value=[9.81])

        # fuzzy_k = Fuzzify(centers=[0.0, 0.5, 1.0], function="rectangular")([k.sw(1)])
        # local_model = LocalModel(name="local_model", function="Fir")
        # output_local = local_model([[x.sw(5), y.sw(5), z.sw(5)], fuzzy_k])
        # out = Output("out", output_local)
        # model = Modely("model_with_local", inputs=[x, y, z, k], outputs=[out])
        # model.build()

        fuzzy_k = Fuzzify(centers=[0.0, 0.5, 1.0], function="rectangular")([k.sw(1)])
        local_model = LocalModel(
            input_function=Fir(out_features=1),
            output_function=ReLU(),
            name="local_model",
        )(activation=fuzzy_k)
        print("Local model ", local_model)
        out = Output("out", local_model([x]) + g)
        model = Modely("model_with_local", inputs=[x, k], outputs=[out])
        print("model:", model)
        model.build()

        # ------- Model inference -------
        # batch_size = 3
        # dummy_input_x = np.ones((batch_size, 5, 5), dtype=np.float32) * 1.0
        # dummy_input_y = np.ones((batch_size, 5, 5), dtype=np.float32) * 2.0
        # dummy_input_z = np.ones((batch_size, 5, 5), dtype=np.float32) * 3.0
        # dummy_input_k = np.array([[[0.0]], [[0.5]], [[1.0]]], dtype=np.float32)

        # batch_size = 3
        # dummy_input_x = np.ones((batch_size, 1, 1), dtype=np.float32) * 1.0
        # dummy_input_k = np.array([[[0.0]], [[0.5]], [[1.0]]], dtype=np.float32)

        # result = model({"x": dummy_input_x,"k": dummy_input_k})

        # print("Model - Input shape x:", dummy_input_x.shape)
        # print("Model - Input shape k:", dummy_input_k.shape)
        # print("Model - Output shape:", result["out"].shape)
        # print("Model - Output:", result)

        model.plot(to_file="html/local_model.png")
        model.plot(to_file="html/local_model_flattened.png", flatten=True)
        model.export_html(out_dir="html", filename="model_with_local")

        # ------- Save/load model inference -------
        # model.save("results/model_local")
        # loaded_model = Modely.load("results/model_local")
        # loaded_model.build()
        # loaded_result = loaded_model(
        #     {
        #         "x": dummy_input_x,
        #         "k": dummy_input_k,
        #     }
        # )
        # print("Loaded Model - Output shape:", loaded_result["out"].shape)
        # print("Loaded Model - Output:", loaded_result)

    if example == 10:
        # ------- Test all nnodely blocks -------
        x = Input("x", dim=1)
        param = Parameter("param1", dim=1)
        const = Constant("const1", value=[1.0])

        add = param + const
        mul = param * const
        sub = param - const
        div = param / const

        Fir_out = Fir(out_features=1)([x.sw(2)])
        out_add = Output("out_add", add)
        out_mul = Output("out_mul", mul)
        out_sub = Output("out_sub", sub)
        out_div = Output("out_div", div)
        out_fir = Output("out_fir", Fir_out)
        out_concat = Output(
            "out_concat", TimeConcatenate(name="time_concat")([x.sw(2), Fir_out])
        )
        param2 = Parameter("param2", dim=(3, 2))
        param3 = Parameter("param3", dim=(1, 2))
        param4 = Parameter("param4", dim=(1, 2))
        out_concat2 = Output(
            "out_concat2", Concatenate(name="concat", axis=0)([param2, param3])
        )
        out_concat3 = Output(
            "out_concat3", Concatenate(name="concat2", axis=1)([param3, param4])
        )
        out_concat4 = Output(
            "out_concat4", Concatenate(name="concat3", axis=0)([param2, param3, param4])
        )
        model = Modely(
            "model",
            inputs=[x],
            outputs=[
                out_add,
                out_mul,
                out_sub,
                out_div,
                out_fir,
                out_concat,
                out_concat2,
                out_concat3,
                out_concat4,
            ],
        )
        model.build()
        model.plot(to_file="html/model_all_blocks.png")

    if example == 11:
        # ------- Test activation layers -------
        x = Input("x", dim=1)
        relu_out = Output("out_relu", ReLU(max_value=1.0)([x.sw(1)]))
        elu_out = Output("out_elu", ELU(alpha=1.0)([x.sw(2)]))
        leaky_relu_out = Output(
            "out_leaky_relu", LeakyReLU(negative_slope=0.3)([x.sw(1)])
        )
        prelu_out = Output("out_prelu", PReLU()([x.sw(1)]))
        sigmoid_out = Output("out_sigmoid", Sigmoid()([x.sw(5)]))
        tanh_out = Output("out_tanh", Tanh()([x.sw(1)]))
        softmax_out = Output("out_softmax", Softmax(axis=-1)([x.sw(3)]))
        swish_out = Output("out_swish", Swish()([x.sw(1)]))
        gelu_out = Output("out_gelu", GELU()([x.sw(1)]))
        softplus_out = Output("out_softplus", Softplus()([x.sw(1)]))
        model = Modely(
            "activation_model",
            inputs=[x],
            outputs=[
                relu_out,
                elu_out,
                leaky_relu_out,
                prelu_out,
                sigmoid_out,
                tanh_out,
                softmax_out,
                swish_out,
                gelu_out,
                softplus_out,
            ],
        )
        model.build()
        model.plot(to_file="html/model_activation_blocks.png")

    if example == 12:
        # ------- Test trigonometric layers -------
        x = Input("x", dim=1)

        sin_out = Output("out_sin", Sin()([x.sw(1)]))
        cos_out = Output("out_cos", Cos()([x.sw(3)]))
        tan_out = Output("out_tan", Tan()([x.sw(1)]))
        asin_out = Output("out_asin", Asin()([x.sw(4)]))
        acos_out = Output("out_acos", Acos()([x.sw(1)]))
        atan_out = Output("out_atan", Atan()([x.sw(2)]))
        model = Modely(
            "trig_model",
            inputs=[x],
            outputs=[sin_out, cos_out, tan_out, asin_out, acos_out, atan_out],
        )
        model.build()
        model.plot(to_file="html/model_trig.png")

    if example == 13:
        x1 = Input("x1")
        x2 = Input("x2")
        x3 = Input("x3")
        mul1 = Multiply("mul1")([x1, x2])
        mul2 = Multiply("mul2")([x2, x3])
        add = Add("add")([mul1, mul2])
        y = Output("y", add)
        m1 = Modely(name="m1", inputs=[x1, x2], outputs=[y])

        m1_flat = flatten(m1)

        a1 = Input("a1")
        a2 = Input("a2")
        m1c = m1([a1, a2])
        m2 = Modely("m2", [a1, a2], [m1c[0], m1c[0]])

        m2_flat = flatten(m2)

        c1 = Input("c1")
        c2 = Input("c2")
        m2c1, m2c2 = m2([c1, c2])
        d1 = Output("d1", m2c1)
        d2 = Output("d2", m2c2)
        m3 = Modely("m3", [c1, c2], [d1, d2])

        m3_flat = flatten(m3)

        e1 = Input("e1")
        e2 = Input("e2")
        m2e1, m2e2 = m2([e1, e2])
        m3e1, m3e2 = m3([m2e1, m2e2])
        f1 = Output("f1", m3e1)
        f2 = Output("f2", m3e2)
        m4 = Modely("m4", [e1, e2], [f1, f2])

        m4_flat = flatten(m4)

        g1 = Input("g1")
        m4g1, m4g2 = m4([g1, g1])
        h1 = Output("h1", m4g1)
        h2 = Output("h2", m4g2)
        m5 = Modely("m5", [g1], [h1, h2])

        m5_flat = flatten(m5)

        print(m1)
        print(m1_flat)
        print(m2)
        print(m2_flat)
        print(m3)
        print(m3_flat)
        print(m4)
        print(m4_flat)
        print(m5)
        print(m5_flat)

    if example == 14:

        class Tangent:
            def __init__(self, name: str | None = "Tangent"):
                self.name = name

            def __call__(self, inputs):
                ret = Sin()(inputs) / Cos()(inputs)
                out = Output(name=f"{self.name}_out", stream=ret)
                return Modely(name=f"{self.name}", inputs=inputs, outputs=[out])(inputs)

        input = Input("input")
        tan = Tangent(name="tangent")([input])
        output = Output("output", tan[0])

        model = Modely(name="model", inputs=[input], outputs=[output])
        model_flat = flatten(model)

        print(model)
        print(model_flat)

        model.plot(to_file="html/model_tangent.png")
        model_flat.plot(to_file="html/model_tangent_flat.png")

        model.export_html(out_dir="html", filename="model_tangent")
        # model_flat.export_html(out_dir="html", filename="model_tangent_flat")

    if example == 15:
        # ------- Test Model Longitudinal Vehicle Dynamics -------
        n = 25
        na = 21

        # Create neural model inputs
        velocity = Input("vel")
        brake = Input("brk")
        gear = Input("gear")
        torque = Input("trq")
        altitude = Input("alt", dim=na)
        acc = Input("acc")

        # Create neural network relations
        air_drag_force = Fir(out_features=1, use_bias=True)([velocity.sw(1)])
        breaking_force = Fir(out_features=1)([brake.sw(n)])
        breaking_force = ReLU()([breaking_force])
        breaking_force = -1 * breaking_force
        # gravity_force = Linear(W_init='init_constant', W_init_params={'value':0}, dropout=0.1, W='gravity')(altitude.sw(1))
        gravity_force = Fir(out_features=1, use_bias=True)([altitude.sw(1)])
        fuzzi_gear = Fuzzify(6, range=[2, 7], functions="Rectangular")([gear.sw(1)])
        local_model = LocalModel(input_function=Fir(out_features=1), name="local_model")
        engine_force = local_model(fuzzi_gear)

        # Create neural network output
        out = Output(
            "accelleration",
            air_drag_force
            + breaking_force
            + gravity_force
            + engine_force([torque.sw(n)]),
        )

        # Add the neural model to the nnodely structure and neuralization of the model
        vehicle = Modely(
            "vehicle",
            inputs=[velocity, brake, gear, torque, altitude, acc],
            outputs=[out],
        )
        vehicle.minimize("acc_error", acc.sw(1), out, loss="rmse")
        vehicle.build()
        vehicle.plot(to_file="html/vehicle_model.png")

        # Load the training and the validation dataset
        data_folder = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "dataset", "trainingset"
        )
        DataLoader(
            model=vehicle,
            format={"vel": 0, "trq": 1, "brk": 2, "gear": 3, "alt": 4, "acc": 5},
            source=data_folder,
        )
        data_folder = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "dataset", "validationset"
        )
        DataLoader(
            model=vehicle,
            format={"vel": 0, "trq": 1, "brk": 2, "gear": 3, "alt": 4, "acc": 5},
            source=data_folder,
        )


if __name__ == "__main__":
    for i in range(1, 16):
        print(f"\n\n--- Running Example {i} ---\n\n")
        main(example=i)
