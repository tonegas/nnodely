from nnodely.support.odeint.solvers import FixedGridODESolver
from nnodely.support.odeint.rk_solvers import rk4_step_func


class Euler(FixedGridODESolver):
    order = 1

    def _step_func(self, func, t0, dt, t1, y0):
        f0 = func(t0, y0)
        return dt * f0, f0


class RK4(FixedGridODESolver):
    order = 4

    def _step_func(self, func, t0, dt, t1, y0):
        f0 = func(t0, y0)
        return rk4_step_func(func, t0, dt, t1, y0, f0=f0), f0
