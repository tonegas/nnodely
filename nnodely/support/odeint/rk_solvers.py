import collections
import warnings
import torch
from nnodely.support.odeint.solvers import _handle_unused_kwargs, find_event, AdaptiveStepsizeEventODESolver, FixedGridODESolver

_ButcherTableau = collections.namedtuple('_ButcherTableau', 'alpha, beta, c_sol, c_error')


_RungeKuttaState = collections.namedtuple('_RungeKuttaState', 'y1, f1, t0, t1, dt, interp_coeff')
# Saved state of the Runge Kutta solver.
#
# Attributes:
#     y1: Tensor giving the function value at the end of the last time step.
#     f1: Tensor giving derivative at the end of the last time step.
#     t0: scalar float32 Tensor giving start of the last time step.
#     t1: scalar float32 Tensor giving end of the last time step.
#     dt: scalar float32 Tensor giving the size for the next time step.
#     interp_coeff: list of Tensors giving coefficients for polynomial
#         interpolation between `t0` and `t1`.

def _interp_fit(y0, y1, y_mid, f0, f1, dt):
    """Fit coefficients for 4th order polynomial interpolation.

    Args:
        y0: function value at the start of the interval.
        y1: function value at the end of the interval.
        y_mid: function value at the mid-point of the interval.
        f0: derivative value at the start of the interval.
        f1: derivative value at the end of the interval.
        dt: width of the interval.

    Returns:
        List of coefficients `[a, b, c, d, e]` for interpolating with the polynomial
        `p = a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e` for values of `x`
        between 0 (start of interval) and 1 (end of interval).
    """
    a = 2 * dt * (f1 - f0) - 8 * (y1 + y0) + 16 * y_mid
    b = dt * (5 * f0 - 3 * f1) + 18 * y0 + 14 * y1 - 32 * y_mid
    c = dt * (f1 - 4 * f0) - 11 * y0 - 5 * y1 + 16 * y_mid
    d = dt * f0
    e = y0
    return [e, d, c, b, a]

def _interp_evaluate(coefficients, t0, t1, t):
    """Evaluate polynomial interpolation at the given time point.

    Args:
        coefficients: list of Tensor coefficients as created by `interp_fit`.
        t0: scalar float32 Tensor giving the start of the interval.
        t1: scalar float32 Tensor giving the end of the interval.
        t: scalar float32 Tensor giving the desired interpolation point.

    Returns:
        Polynomial interpolation of the coefficients at time `t`.
    """

    assert (t0 <= t) & (t <= t1), 'invalid interpolation, fails `t0 <= t <= t1`: {}, {}, {}'.format(t0, t, t1)
    x = (t - t0) / (t1 - t0)
    x = x.to(coefficients[0].dtype)

    total = coefficients[0] + x * coefficients[1]
    x_power = x
    for coefficient in coefficients[2:]:
        x_power = x_power * x
        total = total + x_power * coefficient

    return total

def _select_initial_step(func, t0, y0, order, rtol, atol, norm, f0=None):
    """Empirically select a good initial step.

    The algorithm is described in [1]_.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4, 2nd edition.
    """

    dtype = y0.dtype
    device = y0.device
    t_dtype = t0.dtype
    t0 = t0.to(t_dtype)

    if f0 is None:
        f0 = func(t0, y0)

    scale = atol + torch.abs(y0) * rtol

    d0 = norm(y0 / scale).abs()
    d1 = norm(f0 / scale).abs()

    if d0 < 1e-5 or d1 < 1e-5:
        h0 = torch.tensor(1e-6, dtype=dtype, device=device)
    else:
        h0 = 0.01 * d0 / d1
    h0 = h0.abs()

    y1 = y0 + h0 * f0
    f1 = func(t0 + h0, y1)

    d2 = torch.abs(norm((f1 - f0) / scale) / h0)

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = torch.max(torch.tensor(1e-6, dtype=dtype, device=device), h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1. / float(order + 1))
    h1 = h1.abs()

    return torch.min(100 * h0, h1).to(t_dtype)

def _compute_error_ratio(error_estimate, rtol, atol, y0, y1, norm):
    error_tol = atol + rtol * torch.max(y0.abs(), y1.abs())
    return norm(error_estimate / error_tol).abs()

@torch.no_grad()
def _optimal_step_size(last_step, error_ratio, safety, ifactor, dfactor, order):
    """Calculate the optimal size for the next step."""
    if error_ratio == 0:
        return last_step * ifactor
    if error_ratio < 1:
        dfactor = torch.ones((), dtype=last_step.dtype, device=last_step.device)
    error_ratio = error_ratio.type_as(last_step)
    exponent = torch.tensor(order, dtype=last_step.dtype, device=last_step.device).reciprocal()
    factor = torch.min(ifactor, torch.max(safety / error_ratio ** exponent, dfactor))
    return last_step * factor

class _UncheckedAssign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scratch, value, index):
        ctx.index = index
        scratch.data[index] = value  # sneak past the version checker
        return scratch

    @staticmethod
    def backward(ctx, grad_scratch):
        return grad_scratch, grad_scratch[ctx.index], None

def _runge_kutta_step(func, y0, f0, t0, dt, t1, tableau):
    """Take an arbitrary Runge-Kutta step and estimate error.
    Args:
        func: Function to evaluate like `func(t, y)` to compute the time derivative of `y`.
        y0: Tensor initial value for the state.
        f0: Tensor initial value for the derivative, computed from `func(t0, y0)`.
        t0: float32 scalar Tensor giving the initial time.
        dt: float32 scalar Tensor giving the size of the desired time step.
        t1: float32 scalar Tensor giving the end time; equal to t0 + dt. This is used (rather than t0 + dt) to ensure
            floating point accuracy when needed.
        tableau: _ButcherTableau describing how to take the Runge-Kutta step.
    Returns:
        Tuple `(y1, f1, y1_error, k)` giving the estimated function value after
        the Runge-Kutta step at `t1 = t0 + dt`, the derivative of the state at `t1`,
        estimated error at `t1`, and a list of Runge-Kutta coefficients `k` used for
        calculating these terms.
    """

    t_dtype = y0.abs().dtype

    t0 = t0.to(t_dtype)
    dt = dt.to(t_dtype)
    t1 = t1.to(t_dtype)

    # We use an unchecked assign to put data into k without incrementing its _version counter, so that the backward
    # doesn't throw an (overzealous) error about in-place correctness. We know that it's actually correct.
    k = torch.empty(*f0.shape, len(tableau.alpha) + 1, dtype=y0.dtype, device=y0.device)
    k = _UncheckedAssign.apply(k, f0, (..., 0))
    for i, (alpha_i, beta_i) in enumerate(zip(tableau.alpha, tableau.beta)):
        if alpha_i == 1.:
            # Always step to perturbing just before the end time, in case of discontinuities.
            ti = t1
        else:
            ti = t0 + alpha_i * dt
        yi = y0 + torch.sum(k[..., :i + 1] * (beta_i * dt), dim=-1).view_as(f0)
        f = func(ti, yi)
        k = _UncheckedAssign.apply(k, f, (..., i + 1))

    if not (tableau.c_sol[-1] == 0 and (tableau.c_sol[:-1] == tableau.beta[-1]).all()):
        # This property (true for Dormand-Prince) lets us save a few FLOPs.
        yi = y0 + torch.sum(k * (dt * tableau.c_sol), dim=-1).view_as(f0)

    y1 = yi
    f1 = k[..., -1]
    y1_error = torch.sum(k * (dt * tableau.c_error), dim=-1)
    return y1, f1, y1_error, k


def rk4_step_func(func, t0, dt, t1, y0, f0=None):
    k1 = f0
    if k1 is None:
        k1 = func(t0, y0)
    half_dt = dt * 0.5
    k2 = func(t0 + half_dt, y0 + half_dt * k1)
    k3 = func(t0 + half_dt, y0 + half_dt * k2)
    k4 = func(t1, y0 + dt * k3)
    return (k1 + 2 * (k2 + k3) + k4) * dt * (1.0 / 6.0)


class RKAdaptiveStepsizeODESolver(AdaptiveStepsizeEventODESolver):
    order: int
    tableau: _ButcherTableau
    mid: torch.Tensor

    def __init__(self, func, y0, rtol, atol,
                 min_step=1e-8,
                 max_step=float('inf'),
                 first_step=None,
                 step_t=None,
                 jump_t=None,
                 safety=0.9,
                 ifactor=10.0,
                 dfactor=0.2,
                 max_num_steps=2**20,
                 dtype=torch.float32,
                 **kwargs):
        super(RKAdaptiveStepsizeODESolver, self).__init__(dtype=dtype, y0=y0, **kwargs)

        # We use mixed precision. y has its original dtype (probably float32), whilst all 'time'-like objects use
        # `dtype` (defaulting to float32).
        dtype = torch.promote_types(dtype, y0.abs().dtype)
        device = y0.device

        self.func = func
        self.rtol = torch.as_tensor(rtol, dtype=dtype, device=device)
        self.atol = torch.as_tensor(atol, dtype=dtype, device=device)
        self.min_step = torch.as_tensor(min_step, dtype=dtype, device=device)
        self.max_step = torch.as_tensor(max_step, dtype=dtype, device=device)
        self.first_step = None if first_step is None else torch.as_tensor(first_step, dtype=dtype, device=device)
        self.safety = torch.as_tensor(safety, dtype=dtype, device=device)
        self.ifactor = torch.as_tensor(ifactor, dtype=dtype, device=device)
        self.dfactor = torch.as_tensor(dfactor, dtype=dtype, device=device)
        self.max_num_steps = torch.as_tensor(max_num_steps, dtype=torch.int32, device=device)
        self.dtype = dtype

        self.step_t = None if step_t is None else torch.as_tensor(step_t, dtype=dtype, device=device)
        self.jump_t = None if jump_t is None else torch.as_tensor(jump_t, dtype=dtype, device=device)

        # Copy from class to instance to set device
        self.tableau = _ButcherTableau(alpha=self.tableau.alpha.to(device=device, dtype=y0.dtype),
                                       beta=[b.to(device=device, dtype=y0.dtype) for b in self.tableau.beta],
                                       c_sol=self.tableau.c_sol.to(device=device, dtype=y0.dtype),
                                       c_error=self.tableau.c_error.to(device=device, dtype=y0.dtype))
        self.mid = self.mid.to(device=device, dtype=y0.dtype)

    @classmethod
    def valid_callbacks(cls):
        return super(RKAdaptiveStepsizeODESolver, cls).valid_callbacks() | {'callback_step',
                                                                            'callback_accept_step',
                                                                            'callback_reject_step'}

    def _before_integrate(self, t):
        t0 = t[0]
        f0 = self.func(t[0], self.y0)
        if self.first_step is None:
            first_step = _select_initial_step(self.func, t[0], self.y0, self.order - 1, self.rtol, self.atol,
                                              self.norm, f0=f0)
        else:
            first_step = self.first_step
        self.rk_state = _RungeKuttaState(self.y0, f0, t[0], t[0], first_step, [self.y0] * 5)

    def _advance(self, next_t):
        """Interpolate through the next time point, integrating as necessary."""
        n_steps = 0
        while next_t > self.rk_state.t1:
            assert n_steps < self.max_num_steps, 'max_num_steps exceeded ({}>={})'.format(n_steps, self.max_num_steps)
            self.rk_state = self._adaptive_step(self.rk_state)
            n_steps += 1
        return _interp_evaluate(self.rk_state.interp_coeff, self.rk_state.t0, self.rk_state.t1, next_t)

    def _advance_until_event(self, event_fn):
        """Returns t, state(t) such that event_fn(t, state(t)) == 0."""
        if event_fn(self.rk_state.t1, self.rk_state.y1) == 0:
            return (self.rk_state.t1, self.rk_state.y1)

        n_steps = 0
        sign0 = torch.sign(event_fn(self.rk_state.t1, self.rk_state.y1))
        while sign0 == torch.sign(event_fn(self.rk_state.t1, self.rk_state.y1)):
            assert n_steps < self.max_num_steps, 'max_num_steps exceeded ({}>={})'.format(n_steps, self.max_num_steps)
            self.rk_state = self._adaptive_step(self.rk_state)
            n_steps += 1
        interp_fn = lambda t: _interp_evaluate(self.rk_state.interp_coeff, self.rk_state.t0, self.rk_state.t1, t)
        return find_event(interp_fn, sign0, self.rk_state.t0, self.rk_state.t1, event_fn, self.atol)

    def _adaptive_step(self, rk_state):
        """Take an adaptive Runge-Kutta step to integrate the ODE."""
        y0, f0, _, t0, dt, interp_coeff = rk_state
        if not torch.isfinite(dt):
            dt = self.min_step
        dt = dt.clamp(self.min_step, self.max_step)
        self.func.callback_step(t0, y0, dt)
        t1 = t0 + dt
        # dtypes: self.y0.dtype (probably float32); self.dtype (probably float32)
        # used for state and timelike objects respectively.
        # Then:
        # y0.dtype == self.y0.dtype
        # f0.dtype == self.y0.dtype
        # t0.dtype == self.dtype
        # dt.dtype == self.dtype
        # for coeff in interp_coeff: coeff.dtype == self.y0.dtype

        ########################################################
        #                      Assertions                      #
        ########################################################
        assert t0 + dt > t0, 'underflow in dt {}'.format(dt.item())
        assert torch.isfinite(y0).all(), 'non-finite values in state `y`: {}'.format(y0)

        y1, f1, y1_error, k = _runge_kutta_step(self.func, y0, f0, t0, dt, t1, tableau=self.tableau)
        # dtypes:
        # y1.dtype == self.y0.dtype
        # f1.dtype == self.y0.dtype
        # y1_error.dtype == self.dtype
        # k.dtype == self.y0.dtype

        ########################################################
        #                     Error Ratio                      #
        ########################################################
        error_ratio = _compute_error_ratio(y1_error, self.rtol, self.atol, y0, y1, self.norm)
        accept_step = error_ratio <= 1

        # Handle min max stepping
        if dt > self.max_step:
            accept_step = False
        if dt <= self.min_step:
            accept_step = True

        # dtypes:
        # error_ratio.dtype == self.dtype

        ########################################################
        #                   Update RK State                    #
        ########################################################
        if accept_step:
            self.func.callback_accept_step(t0, y0, dt)
            t_next = t1
            y_next = y1
            interp_coeff = self._interp_fit(y0, y_next, k, dt)
            f_next = f1
        else:
            self.func.callback_reject_step(t0, y0, dt)
            t_next = t0
            y_next = y0
            f_next = f0
        dt_next = _optimal_step_size(dt, error_ratio, self.safety, self.ifactor, self.dfactor, self.order)
        dt_next = dt_next.clamp(self.min_step, self.max_step)
        rk_state = _RungeKuttaState(y_next, f_next, t0, t_next, dt_next, interp_coeff)
        return rk_state

    def _interp_fit(self, y0, y1, k, dt):
        """Fit an interpolating polynomial to the results of a Runge-Kutta step."""
        dt = dt.type_as(y0)
        y_mid = y0 + torch.sum(k * (dt * self.mid), dim=-1).view_as(y0)
        f0 = k[..., 0]
        f1 = k[..., -1]
        return _interp_fit(y0, y1, y_mid, f0, f1, dt)
    
class FixedGridFIRKODESolver(FixedGridODESolver):
    order: int
    tableau: _ButcherTableau

    def __init__(self, func, y0, step_size=None, grid_constructor=None, interp='linear', perturb=False, max_iters=100, **unused_kwargs):

        self.max_iters = max_iters
        self.atol = unused_kwargs.pop('atol')
        unused_kwargs.pop('rtol', None)
        unused_kwargs.pop('norm', None)
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.dtype = y0.dtype
        self.device = y0.device
        self.step_size = step_size
        self.interp = interp
        self.perturb = perturb

        if step_size is None:
            if grid_constructor is None:
                self.grid_constructor = lambda f, y0, t: t
            else:
                self.grid_constructor = grid_constructor
        else:
            if grid_constructor is None:
                self.grid_constructor = self._grid_constructor_from_step_size(step_size)
            else:
                raise ValueError("step_size and grid_constructor are mutually exclusive arguments.")
            
        self.tableau = _ButcherTableau(alpha=self.tableau.alpha.to(device=self.device, dtype=y0.dtype),
                                       beta=[b.to(device=self.device, dtype=y0.dtype) for b in self.tableau.beta],
                                       c_sol=self.tableau.c_sol.to(device=self.device, dtype=y0.dtype),
                                       c_error=self.tableau.c_error.to(device=self.device, dtype=y0.dtype))
        
    def _step_func(self, func, t0, dt, t1, y0):
        if not isinstance(t0, torch.Tensor):
            t0 = torch.tensor(t0)
        if not isinstance(dt, torch.Tensor):
            dt = torch.tensor(dt)
        if not isinstance(t1, torch.Tensor):
            t1 = torch.tensor(t1)
        f0 = func(t0, y0)
        
        t_dtype = y0.abs().dtype
        tol = 1e-8
        if t_dtype == torch.float32:
            tol = 1e-8
        if t_dtype == torch.float32:
            tol = 1e-6

        t0 = t0.to(t_dtype)
        dt = dt.to(t_dtype)
        t1 = t1.to(t_dtype)

        k = f0.clone().unsqueeze(-1).tile(len(self.tableau.alpha))
        beta = torch.stack(self.tableau.beta, -1)

        # Broyden's Method to solve the system of nonlinear equations
        y = torch.matmul(k, beta * dt).add(y0.unsqueeze(-1)).movedim(-1, 0)
        f = self._residual(func, k, y, t0, dt, t1)
        J = torch.ones_like(f).diag()
        converged = False
        for _ in range(self.max_iters):
            if torch.linalg.norm(f, 2) < tol:
                converged = True
                break

            # If the matrix becomes singular, just stop and return the last value
            try:
                s = -torch.linalg.solve(J, f)
            except torch._C._LinAlgError:
                break

            k = k + s.reshape_as(k)
            y = torch.matmul(k, beta * dt).add(y0.unsqueeze(-1)).movedim(-1, 0)
            newf = self._residual(func, k, y, t0, dt, t1)
            z = newf - f
            f = newf
            J = J + (torch.outer ((z - torch.linalg.vecdot(J,s)),s)) / (torch.dot(s,s))

        if not converged:
            warnings.warn('Functional iteration did not converge. Solution may be incorrect.')

        dy = torch.matmul(k, dt * self.tableau.c_sol)

        return dy, f0
    
    def _residual(self, func, K, y, t0, dt, t1):
        res = torch.zeros_like(K)
        for i, (y_i, alpha_i) in enumerate(zip(y, self.tableau.alpha)):
            if alpha_i == 1.:
                ti = t1
            elif alpha_i == 0.:
                if not torch.all(self.tableau.beta[i]):
                    # Same slope as stored so skip
                    continue
                ti = t0
            else:
                ti = t0 + alpha_i * dt
            res[...,i] = K[...,i] - func(ti, y_i)
        return res.flatten()