def test_partitioning_windows():
    pass
    # # ------- Model with time window partitioning, Past and Future -------
    # x = Input("x", dim=1)
    # y = Input("y", dim=1)
    # z = Input("z", dim=1)
    # r1 = x.sw(1) + y.sw(1)
    # out1 = Output("out1", r1)
    # model1 = Modely("model1", inputs=[x, y], outputs=[out1])
    # model1.build()

    # r2 = Fir(out_features=1)([z.sw(3)])
    # out2 = Output("out2", model1({"x": r2, "y": z.sw(1)}))
    # model2 = Modely("model2", inputs=[z], outputs=[out2])
    # model2.build()

    # k = Input("k", dim=1)
    # f = Input("f", dim=1)
    # r3 = model2({"z": k.sw(3)}) + model2({"z": f.sw(3)})
    # out3 = Output("out3", r3)
    # model3 = Modely("model3", inputs=[k, f], outputs=[out3])
    # model3.build()
