from nnodely.layers.parameter import SampleTime


class FixedStepSolver:
    def __init__(self, int_name: str | None = None, der_name: str | None = None):
        self.dt = SampleTime()
        self.int_name = int_name
        self.der_name = der_name


class Euler(FixedStepSolver):
    def __init__(self, int_name: str | None = None, der_name: str | None = None):
        super().__init__(int_name, der_name)

    def integrate(self, obj):
        from nnodely.layers.input import Input

        integral = Input(self.int_name, dimensions=obj.dim["dim"])
        return (integral.last() + obj * self.dt).closedLoop(integral)

    def derivate(self, obj):
        from nnodely.layers.input import Input

        obj = Input(self.int_name, dimensions=obj.dim["dim"]).connect(obj)
        return (obj.last() - obj.sw([-2, -1])) / self.dt


class Trapezoidal(FixedStepSolver):
    def __init__(self, int_name: str | None = None, der_name: str | None = None):
        super().__init__(int_name, der_name)

    def integrate(self, obj):
        from nnodely.layers.input import Input

        integral = Input(self.int_name, dimensions=obj.dim["dim"])
        obj = Input(self.der_name, dimensions=obj.dim["dim"]).connect(obj)
        return (
            integral.last() + (obj.last() + obj.sw([-2, -1])) * 0.5 * self.dt
        ).closedLoop(integral)

    def derivate(self, obj):
        from nnodely.layers.input import Input

        obj = Input(self.int_name, dimensions=obj.dim["dim"]).connect(obj)
        derivative = Input(self.der_name, dimensions=obj.dim["dim"])
        return (
            ((obj.last() - obj.sw([-2, -1])) * 2.0) / self.dt - derivative.last()
        ).closedLoop(derivative)
