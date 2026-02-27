import torch
import warnings

_all_callback_names = ['callback_step', 'callback_accept_step', 'callback_reject_step']
_all_adjoint_callback_names = [name + '_adjoint' for name in _all_callback_names]
_null_callback = lambda *args, **kwargs: None

def _linf_norm(tensor):
    return tensor.abs().max()

def _rms_norm(tensor):
    return tensor.abs().pow(2).mean().sqrt()

def _zero_norm(tensor):
    return 0.

def _mixed_norm(tensor_tuple):
    if len(tensor_tuple) == 0:
        return 0.
    return max([_rms_norm(tensor) for tensor in tensor_tuple])


def combine_event_functions(event_fn, t0, y0):
    """
    We ensure all event functions are initially positive,
    so then we can combine them by taking a min.
    """
    with torch.no_grad():
        initial_signs = torch.sign(event_fn(t0, y0))

    def combined_event_fn(t, y):
        c = event_fn(t, y)
        return torch.min(c * initial_signs)

    return combined_event_fn


def _tuple_tol(name, tol, shapes):
    try:
        iter(tol)
    except TypeError:
        return tol
    tol = tuple(tol)
    assert len(tol) == len(shapes), "If using tupled {} it must have the same length as the tuple y0".format(name)
    tol = [torch.as_tensor(tol_).expand(shape.numel()) for tol_, shape in zip(tol, shapes)]
    return torch.cat(tol)


def _flat_to_shape(tensor, length, shapes):
    tensor_list = []
    total = 0
    for shape in shapes:
        next_total = total + shape.numel()
        # It's important that this be view((...)), not view(...). Else when length=(), shape=() it fails.
        tensor_list.append(tensor[..., total:next_total].view((*length, *shape)))
        total = next_total
    return tuple(tensor_list)


def _assert_floating(name, t):
    if not torch.is_floating_point(t):
        raise TypeError('`{}` must be a floating point Tensor but is a {}'.format(name, t.type()))


def _check_timelike(name, timelike, can_grad):
    assert isinstance(timelike, torch.Tensor), '{} must be a torch.Tensor'.format(name)
    _assert_floating(name, timelike)
    assert timelike.ndimension() == 1, "{} must be one dimensional".format(name)
    if not can_grad:
        assert not timelike.requires_grad, "{} cannot require gradient".format(name)
    diff = timelike[1:] > timelike[:-1]
    assert diff.all() or (~diff).all(), '{} must be strictly increasing or decreasing'.format(name)


class _TupleFunc(torch.nn.Module):
    def __init__(self, base_func, shapes):
        super(_TupleFunc, self).__init__()
        self.base_func = base_func
        self.shapes = shapes

    def forward(self, t, y):
        f = self.base_func(t, _flat_to_shape(y, (), self.shapes))
        return torch.cat([f_.reshape(-1) for f_ in f])


class _TupleInputOnlyFunc(torch.nn.Module):
    def __init__(self, base_func, shapes):
        super(_TupleInputOnlyFunc, self).__init__()
        self.base_func = base_func
        self.shapes = shapes

    def forward(self, t, y):
        return self.base_func(t, _flat_to_shape(y, (), self.shapes))


class _ReverseFunc(torch.nn.Module):
    def __init__(self, base_func, mul=1.0):
        super(_ReverseFunc, self).__init__()
        self.base_func = base_func
        self.mul = mul

    def forward(self, t, y):
        return self.mul * self.base_func(-t, y)


def _check_inputs(func, y0, t, rtol, atol, method, options, event_fn, SOLVERS):

    if event_fn is not None:
        if len(t) != 2:
            raise ValueError(f"We require len(t) == 2 when in event handling mode, but got len(t)={len(t)}.")

        # Combine event functions if the output is multivariate.
        event_fn = combine_event_functions(event_fn, t[0], y0)

    # Keep reference to original func as passed in
    original_func = func

    # Normalise to tensor (non-tupled) input
    shapes = None
    is_tuple = not isinstance(y0, torch.Tensor)
    if is_tuple:
        assert isinstance(y0, tuple), 'y0 must be either a torch.Tensor or a tuple'
        shapes = [y0_.shape for y0_ in y0]
        rtol = _tuple_tol('rtol', rtol, shapes)
        atol = _tuple_tol('atol', atol, shapes)
        y0 = torch.cat([y0_.reshape(-1) for y0_ in y0])
        func = _TupleFunc(func, shapes)
        if event_fn is not None:
            event_fn = _TupleInputOnlyFunc(event_fn, shapes)

    # Normalise method and options
    if options is None:
        options = {}
    else:
        options = options.copy()
    if method is None:
        method = 'dopri5'
    if method not in SOLVERS:
        raise ValueError('Invalid method "{}". Must be one of {}'.format(method,
                                                                         '{"' + '", "'.join(SOLVERS.keys()) + '"}.'))

    if is_tuple:
        # We accept tupled input. This is an abstraction that is hidden from the rest of odeint (exception when
        # returning values), so here we need to maintain the abstraction by wrapping norm functions.

        if 'norm' in options:
            # If the user passed a norm then get that...
            norm = options['norm']
        else:
            # ...otherwise we default to a mixed Linf/L2 norm over tupled input.
            norm = _mixed_norm

        # In either case, norm(...) is assumed to take a tuple of tensors as input. (As that's what the state looks
        # like from the point of view of the user.)
        # So here we take the tensor that the machinery of odeint has given us, and turn it in the tuple that the
        # norm function is expecting.
        def _norm(tensor):
            y = _flat_to_shape(tensor, (), shapes)
            return norm(y)
        options['norm'] = _norm

    else:
        if 'norm' in options:
            # No need to change the norm function.
            pass
        else:
            # Else just use the default norm.
            # Technically we don't need to set that here (RKAdaptiveStepsizeODESolver has it as a default), but it
            # makes it easier to reason about, in the adjoint norm logic, if we know that options['norm'] is
            # definitely set to something.
            options['norm'] = _rms_norm

    # Normalise time
    _check_timelike('t', t, True)
    t_is_reversed = False
    if len(t) > 1 and t[0] > t[1]:
        t_is_reversed = True

    if t_is_reversed:
        # Change the integration times to ascending order.
        # We do this by negating the time values and all associated arguments.
        t = -t

        # Ensure time values are un-negated when calling functions.
        func = _ReverseFunc(func, mul=-1.0)
        if event_fn is not None:
            event_fn = _ReverseFunc(event_fn)

        # For fixed step solvers.
        try:
            _grid_constructor = options['grid_constructor']
        except KeyError:
            pass
        else:
            options['grid_constructor'] = lambda func, y0, t: -_grid_constructor(func, y0, -t)

        # For RK solvers.
        #_flip_option(options, 'step_t')
        #_flip_option(options, 'jump_t')

    # Can only do after having normalised time
    assert (t[1:] > t[:-1]).all(), 't must be strictly increasing or decreasing'

    # Tol checking
    if torch.is_tensor(rtol):
        assert not rtol.requires_grad, "rtol cannot require gradient"
    if torch.is_tensor(atol):
        assert not atol.requires_grad, "atol cannot require gradient"

    # Backward compatibility: Allow t and y0 to be on different devices
    if t.device != y0.device:
        warnings.warn("t is not on the same device as y0. Coercing to y0.device.")
        t = t.to(y0.device)
    # ~Backward compatibility

    # Add perturb argument to func.
    #func = _PerturbFunc(func)

    # Add callbacks to wrapped_func
    callback_names = set()
    for callback_name in _all_callback_names:
        try:
            callback = getattr(original_func, callback_name)
        except AttributeError:
            setattr(func, callback_name, _null_callback)
        else:
            if callback is not _null_callback:
                callback_names.add(callback_name)
                # At the moment all callbacks have the arguments (t0, y0, dt).
                # These will need adjusting on a per-callback basis if that changes in the future.
                if is_tuple:
                    def callback(t0, y0, dt, _callback=callback):
                        y0 = _flat_to_shape(y0, (), shapes)
                        return _callback(t0, y0, dt)
                if t_is_reversed:
                    def callback(t0, y0, dt, _callback=callback):
                        return _callback(-t0, y0, dt)
            setattr(func, callback_name, callback)
    for callback_name in _all_adjoint_callback_names:
        try:
            callback = getattr(original_func, callback_name)
        except AttributeError:
            pass
        else:
            setattr(func, callback_name, callback)

    invalid_callbacks = callback_names - SOLVERS[method].valid_callbacks()
    if len(invalid_callbacks) > 0:
        warnings.warn("Solver '{}' does not support callbacks {}".format(method, invalid_callbacks))

    return shapes, func, y0, t, rtol, atol, method, options, event_fn, t_is_reversed    