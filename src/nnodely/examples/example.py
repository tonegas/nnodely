import numpy as np
from typing import Any

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

import os

# os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ["KERAS_BACKEND"] = "torch"

def dummy_input(shape, method="ones"):
    if method == "ones":
        return np.ones(shape, dtype=np.float32)
    elif method == "zeros":
        return np.zeros(shape, dtype=np.float32)
    elif method == "random":
        return np.random.rand(*shape).astype(np.float32)
    elif method == "sequential":
        return np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

def main(example=1):
    if example == 1:
        # ------- Model definition and building -------
        x = Input("x", dim=1)
        y = Input("y", dim=1)

        window_size = 10
        x_stream = x.sw(window_size)
        y_stream = y.sw(window_size)

        fir = Fir(out_features=2)
        result_fir = fir(x_stream + y_stream)

        x_out = Output("x_pred", result_fir)
        model1 = Modely("model1", outputs=x_out)
        print(f"order: {model1._order}")
        model1.build()
        print(
            f"Model1 - Output stream: {x_out}, shape: {x_out.shape} \n Input streams: {model1.inputs}, shape: {list(model1._input_shapes.values())}"
        )

        # ------- Model composition -------
        z = Input("z", dim=1)
        z_stream = z.sw(window_size)
        z_fir = Fir(out_features=1)(model1({"x": z_stream, "y": z_stream}))
        z_out = Output("z_pred", z_fir)
        model2 = Modely("composed_model", outputs=z_out)
        print(f"order: {model2._order}")
        model2.build()
        print(
            f"Model2 - Output stream: {z_out}, shape: {z_out.shape} \n Input streams: {model2.inputs}, shape: {list(model2._input_shapes.values())}"
        )

        # ------- Model inference -------
        batch_size = 4
        dummy_input_x = dummy_input((batch_size, window_size, 1), method="ones")
        dummy_input_y = dummy_input((batch_size, window_size, 1), method="ones")
        dummy_input_z = dummy_input((batch_size, window_size, 1), method="ones")

        result1 = model1({"x": dummy_input_x, "y": dummy_input_y})
        print("Model1 - Input shape:", dummy_input_x.shape)
        print("Model1 - Output shape:", result1["x_pred"].shape)
        print("Model1 - Output:", result1)

        result2 = model2({"z": dummy_input_z})
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
        loaded_result1 = loaded_model1({"x": dummy_input_x, "y": dummy_input_y})
        loaded_result2 = loaded_model2({"z": dummy_input_z})
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
        x_fir = Fir(out_features=1)(x.sw(5))
        y_fir = Fir(out_features=1)(y.sw(5))

        x_out = Output("fir_pred", x_fir + y_fir)
        model1 = Modely("linear_fit", outputs=[x_out])

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
        model = Modely("model", outputs=x_out)
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
        model1 = Modely("model1", outputs=x_out)
        model1.build()

        y = Input("y", dim=1)
        y_fir = Fir(out_features=1)(model1({"x": y.sw(5)}))
        y_out = Output("y_out", y_fir)
        model2 = Modely("model2", outputs=y_out)
        model2.build()

        z = Input("z", dim=1)
        const_z = Constant("const_z", value=[1.0, 2.0, 3.0, 4.0, 5.0])
        z_fir = Fir(out_features=1)(model2({"y": z.sw(5) + const_z}))
        z_out = Output("z_out", z_fir)
        model3 = Modely("model3", outputs=z_out)
        model3.build()

        # ------- Visualize the Flatten model -------
        model3.export_html(out_dir="html", filename="model3")
        model3.plot(to_file="html/model3_standard.png")
        model3.plot(to_file="html/model3_flattened.png", flatten=True)

    if example == 5:
        # ------- Model with closed loop connections -------
        from nnodely.layers.loop import Loop

        _x = Input(name="_x", dim=1)
        _y = Input(name="_y", dim=1)
        r1 = _x.sw(1) + _y.sw(1)
        out1 = Output("out1", r1)
        model_add = Modely(name="model1", outputs=[out1])
        model_add.build()

        x = Input(name="x", dim=1)
        y = Input(name="y", dim=1, seq=4)
        loop_fn = Loop(f=model_add, closed_loop={"x": "out1"}, name="loop_model_add")
        out = Output("out", loop_fn(x, y))
        model_in = Modely(name="model", outputs=[out])
        model_in.build()

        w = Input(name="w", dim=1)
        z = Input(name="z", dim=1, seq=(4, 2))
        loop_fn2 = Loop(f=model_in, closed_loop={"z": "out"}, name="loop_model_in")
        out_w = Output("out_w", loop_fn2(w, z))
        model_out = Modely(name="model_with_loop_w", outputs=[out_w])
        model_out.build()

        model_in.plot(to_file="html/model_with_loop.png")
        model_out.plot(to_file="html/model_with_loop_w.png")

        # ------- Model inference -------
        batch_size = 1
        dummy_input_x = dummy_input((batch_size, 1, 1), method="ones")
        dummy_input_y = dummy_input((batch_size, 1, 1, 4), method="sequential")

        result_in = model_in({"x": dummy_input_x, "y": dummy_input_y})
        print("Model with loop - Output shape:", result_in.shape)
        print("Model with loop - Output:", result_in)

        dummy_input_w = dummy_input((batch_size, 1, 1), method="ones")
        dummy_input_z = dummy_input((batch_size, 1, 1, 4, 2), method="ones")
        result_out = model_out({"z": dummy_input_z, "w": dummy_input_w})
        print("Model with loop w - Output shape:", result_out.shape)
        print("Model with loop w - Output:", result_out)

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
        model1 = Modely(name="model1", outputs=[out1])
        model1.build()

        z = Input(name="z", dim=1, seq=5)
        const = Constant("const", value=2.0)
        r2 = z.sw(1) * const
        loop_fn = Loop(f=model1, closed_loop={"out1": "z"})
        out = Output("out", loop_fn(z, r2))
        model = Modely(name="model", outputs=[out])
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

        x_out = Output("x_pred", fuzzy(x_stream))
        model1 = Modely("model1", outputs=x_out)
        print(f"order: {model1._order}")
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
        x = Input("x", dim=5)
        y = Input("y", dim=5)
        z = Input("z", dim=5)
        k = Input("k", dim=1)

        # LocalModel now auto-encapsulates in a ModelCall when called with Streams
        fuzzy_k = Fuzzify(centers=[0.0, 0.5, 1.0], function="rectangular")(k.sw(1))
        local_model = LocalModel(name="local_model", function="Fir")
        # Call with Stream inputs directly - returns a ModelCall automatically
        output_local = local_model({"x": x.sw(5), "y": y.sw(5), "z": z.sw(5)}, fuzzy_k)
        out = Output("out", output_local)
        model = Modely("model_with_local", outputs=out)
        model.build()

        # ------- Model inference -------
        batch_size = 3
        dummy_input_x = np.ones((batch_size, 5, 5), dtype=np.float32) * 1.0
        dummy_input_y = np.ones((batch_size, 5, 5), dtype=np.float32) * 2.0
        dummy_input_z = np.ones((batch_size, 5, 5), dtype=np.float32) * 3.0
        dummy_input_k = np.array([[[0.0]], [[0.5]], [[1.0]]], dtype=np.float32)

        result = model(
            {
                "x": dummy_input_x,
                "y": dummy_input_y,
                "z": dummy_input_z,
                "k": dummy_input_k,
            }
        )

        print("Model - Input shape x:", dummy_input_x.shape)
        print("Model - Input shape k:", dummy_input_k.shape)
        print("Model - Output shape:", result["out"].shape)
        print("Model - Output:", result)

        model.plot(to_file="html/local_model.png")
        model.export_html(out_dir="html", filename="model_with_local")

        # ------- Save/load model inference -------
        model.save("results/model_local")
        loaded_model = Modely.load("results/model_local")
        loaded_model.build()
        loaded_result = loaded_model(
            {
                "x": dummy_input_x,
                "y": dummy_input_y,
                "z": dummy_input_z,
                "k": dummy_input_k,
            }
        )
        print("Loaded Model - Output shape:", loaded_result["out"].shape)
        print("Loaded Model - Output:", loaded_result)

    if example == 10:
        # ------- Test all nnodely blocks -------
        x = Input("x", dim=1)
        param = Parameter("param1", dim=1)
        const = Constant("const1", value=[1.0])

        add = param + const
        mul = param * const
        sub = param - const
        div = param / const

        Fir_out = Fir(out_features=1)(x.sw(2))
        out_add = Output("out_add", add)
        out_mul = Output("out_mul", mul)
        out_sub = Output("out_sub", sub)
        out_div = Output("out_div", div)
        out_fir = Output("out_fir", Fir_out)
        out_concat = Output(
            "out_concat", TimeConcatenate(name="time_concat")(x.sw(2), Fir_out)
        )
        param2 = Parameter("param2", dim=(3, 2))
        param3 = Parameter("param3", dim=(1, 2))
        param4 = Parameter("param4", dim=(1, 2))
        out_concat2 = Output(
            "out_concat2", Concatenate(name="concat", axis=0)(param2, param3)
        )
        out_concat3 = Output(
            "out_concat3", Concatenate(name="concat2", axis=1)(param3, param4)
        )
        out_concat4 = Output(
            "out_concat4", Concatenate(name="concat3", axis=0)(param2, param3, param4)
        )
        model = Modely(
            "model",
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
        relu_out = Output("out_relu", ReLU(max_value=1.0)(x.sw(1)))
        elu_out = Output("out_elu", ELU(alpha=1.0)(x.sw(2)))
        leaky_relu_out = Output(
            "out_leaky_relu", LeakyReLU(negative_slope=0.3)(x.sw(1))
        )
        prelu_out = Output("out_prelu", PReLU()(x.sw(1)))
        sigmoid_out = Output("out_sigmoid", Sigmoid()(x.sw(5)))
        tanh_out = Output("out_tanh", Tanh()(x.sw(1)))
        softmax_out = Output("out_softmax", Softmax(axis=-1)(x.sw(3)))
        swish_out = Output("out_swish", Swish()(x.sw(1)))
        gelu_out = Output("out_gelu", GELU()(x.sw(1)))
        softplus_out = Output("out_softplus", Softplus()(x.sw(1)))
        model = Modely(
            "activation_model",
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

        sin_out = Output("out_sin", Sin()(x.sw(1)))
        cos_out = Output("out_cos", Cos()(x.sw(3)))
        tan_out = Output("out_tan", Tan()(x.sw(1)))
        asin_out = Output("out_asin", Asin()(x.sw(4)))
        acos_out = Output("out_acos", Acos()(x.sw(1)))
        atan_out = Output("out_atan", Atan()(x.sw(2)))
        model = Modely(
            "trig_model",
            outputs=[sin_out, cos_out, tan_out, asin_out, acos_out, atan_out],
        )
        model.build()
        model.plot(to_file="html/model_trig.png")

    if example == 13:
        # ------- Test CustomLayer -------
        from nnodely.core.custom_layer import custom_layer
        from inspect import signature
        import keras

        @custom_layer
        class MyFirImpl(keras.layers.Layer):
            """
            A simple FIR implementation as a custom Keras layer.

            Args:
                out_features: Number of output features.
                use_bias: Whether to include a bias term.
                name: Optional name for the layer.
                **kwargs: Additional keyword arguments.
            """

            def __init__(
                self,
                out_features: int,
                use_bias: bool = True,
                name: str | None = None,
                **kwargs: Any,
            ):
                super().__init__(name=name, **kwargs)
                self.out_features = int(out_features)
                self.use_bias = bool(use_bias)
                self.flatten = keras.layers.Flatten()
                self.proj = keras.layers.Dense(
                    self.out_features, use_bias=self.use_bias
                )
                self.reshape = keras.layers.Reshape((1, self.out_features))

            def get_config(self):
                config = super().get_config()
                config.update(
                    {
                        "out_features": self.out_features,
                        "use_bias": self.use_bias,
                    }
                )
                return config

            def output_shape(self, *inputs):
                print("Fir.output_shape called with inputs:", inputs)
                seq = inputs[0].seq
                in_dim = inputs[0].dim
                if len(seq) != 0:
                    raise NotImplementedError(
                        "This proposal handles seq=() streams first; extend here for nested sequence dims."
                    )
                if len(in_dim) != 1:
                    raise ValueError(
                        f"Fir currently expects a single feature axis, got dim={in_dim}"
                    )
                return seq, 1, (self.out_features,)

            def build(self, input_shape):
                flat_shape = self.flatten.compute_output_shape(input_shape)
                self.proj.build(flat_shape)
                proj_shape = self.proj.compute_output_shape(flat_shape)
                self.reshape.build(proj_shape)
                super().build(input_shape)

            def call(self, x):
                x = self.flatten(x)
                x = self.proj(x)
                return self.reshape(x)

        impl = MyFirImpl(out_features=1, use_bias=False)
        print(signature(MyFirImpl.__init__))
        print("type of impl:", type(impl))
        # Firr = custom_layer(MyFirImpl)#(out_features=1, use_bias=False)
        # print("type of impl after custom_layer:", type(Firr))
        # impl = Firr(out_features=1, use_bias=False)
        x = Input("x", dim=1)
        out = Output("out", impl(x.sw(5)))
        model = Modely("model_with_custom_layer", outputs=out)
        model.build()
        model.plot(to_file="html/model_with_custom_layer.png")
        model.export_html(out_dir="html", filename="model_with_custom_layer")

        # ------- Model inference -------
        batch_size = 3
        dummy_input_x = np.ones((batch_size, 5, 1), dtype=np.float32)
        result = model({"x": dummy_input_x})
        print("Model - Input shape x:", dummy_input_x.shape)
        print("Model - Output shape:", result["out"].shape)
        print("Model - Output:", result)

        # ------- Save/load model inference -------
        model.save("results/model_with_custom_layer")
        loaded_model = Modely.load("results/model_with_custom_layer")
        loaded_model.build()
        loaded_result = loaded_model({"x": dummy_input_x})
        print("Loaded Model - Output shape:", loaded_result["out"].shape)
        print("Loaded Model - Output:", loaded_result)

import keras
if __name__ == "__main__":
    main(example=5)
    exit(0)
    for i in range(1, 13):
        print(f"\n\n--- Running Example {i} ---\n\n")
        main(example=i)
