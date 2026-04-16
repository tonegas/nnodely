from nnodely.layers.input import Input
from nnodely.layers.output import Output
from nnodely.core.modely import Modely
from nnodely.layers.constant import Constant
from nnodely.layers.loop import Loop

x = Input(name="x", dim=1)
y = Input(name="y", dim=1)
r1 = x.sw(1) + y.sw(1)
out1 = Output("out1", r1)
model1 = Modely(name="model1", outputs=[out1])

z = Input(name="z", dim=1, seq=5)
const = Constant('const', value=2.0)
r2 = z.sw(1) * const
loop_fn = Loop(f=model1, closed_loop={out1: z})
out = Output("out", loop_fn(z,r2))
model = Modely(name="model", outputs=[out])

model.build()