"""This module contains utility functions, which fulfill common puposes
in different modules of this project."""

import inspect

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
from scipy.integrate import BDF
import prince_cr.config as config

def convert_to_namedtuple(dictionary, name='GenericNamedTuple'):
    """Converts a dictionary to a named tuple."""
    from collections import namedtuple
    return namedtuple(name, list(dictionary.keys()))(**dictionary)

def get_interp_object(xgrid, ygrid, **kwargs):
    """Returns simple standard interpolation object.

    Default type of interpolation is a spline of order
    one without extrapolation (extrapolation to zero).

    Args:
        xgrid (numpy.array): x values of function
        ygrid (numpy.array): y values of function
    """
    if xgrid.shape != ygrid.shape:
        raise Exception(
            'xgrid and ygrid args need identical shapes: {0} != {1}'.format(
                xgrid.shape, ygrid.shape))

    if 'k' not in kwargs:
        kwargs['k'] = 1
    if 'ext' not in kwargs:
        kwargs['ext'] = 'zeros'

    return InterpolatedUnivariateSpline(xgrid, ygrid, **kwargs)
    # if xwidths is not None:
    #     return np.tile(xwidths,len(ygrid)).reshape(len(xwidths),len(ygrid))*res
    # else:
    #     return res


def get_2Dinterp_object(xgrid, ygrid, zgrid, xbins=None, **kwargs):
    """Returns simple standard interpolation object for 2-dimentsional
    distribution.

    Default type of interpolation is a spline of order
    one without extrapolation (extrapolation to zero).

    Args:
        xgrid (numpy.array): x values of function
        ygrid (numpy.array): y values of function
    """
    if (xgrid.shape[0], ygrid.shape[0]) != zgrid.shape:
        raise Exception(
            'x and y grid do not match z grid shape: {0} != {1}'.format(
                (xgrid.shape, ygrid.shape), zgrid.shape))

    if 'kx' not in kwargs:
        kwargs['kx'] = 1
    if 'ky' not in kwargs:
        kwargs['ky'] = 1
    if 's' not in kwargs:
        kwargs['s'] = 0.
    return RectBivariateSplineNoExtrap(xgrid, ygrid, zgrid, xbins, **kwargs)


class RectBivariateSplineNoExtrap(RectBivariateSpline):
    """Same as RectBivariateSpline but makes sure, that extrapolated data is alway 0"""

    def __init__(self, xgrid, ygrid, zgrid, xbins=None, *args, **kwargs):
        self.xbins = xbins
        RectBivariateSpline.__init__(self, xgrid, ygrid, zgrid, *args,
                                     **kwargs)
        xknots, yknots = self.get_knots()
        self.xmin, self.xmax = np.min(xknots), np.max(xknots)
        self.ymin, self.ymax = np.min(yknots), np.max(yknots)

    def __call__(self, x, y, **kwargs):
        if 'grid' not in kwargs:
            x, y = np.meshgrid(x, y)
            kwargs['grid'] = False

            result = RectBivariateSpline.__call__(self, x, y, **kwargs)
            # result = np.where((x < self.xmax) & (x > self.xmin), result, 0.)
            # result[np.isnan(result)] = 0.
            # return result.T
            return np.where(np.isnan(result), 0., result).T
        else:
            result = RectBivariateSpline.__call__(self, x, y, **kwargs)
            # result = np.where((x <= xmax) & (x >= xmin), result, 0.)
            # result[np.isnan(result)] = 0.
            # return result
            return np.where(np.isnan(result), 0., result)


class RectBivariateSplineLogData(RectBivariateSplineNoExtrap):
    """Same as RectBivariateSpline but data is internally interpoled as log(data)"""

    def __init__(self, x, y, z, *args, **kwargs):
        x = np.log10(x)
        y = np.log10(y)

        info(2, 'Spline created')
        RectBivariateSplineNoExtrap.__init__(self, x, y, z, *args, **kwargs)

    def __call__(self, x, y, **kwargs):
        x = np.log10(x)
        y = np.log10(y)

        result = RectBivariateSplineNoExtrap.__call__(self, x, y, **kwargs)
        return result


def caller_name(skip=2):
    """Get a name of a caller in the format module.class.method

    `skip` specifies how many levels of stack to skip while getting caller
    name. skip=1 means "who calls me", skip=2 "who calls my caller" etc.
    An empty string is returned if skipped levels exceed stack height.abs

    From https://gist.github.com/techtonik/2151727
    """
    import inspect

    stack = inspect.stack()
    start = 0 + skip

    if len(stack) < start + 1:
        return ''

    parentframe = stack[start][0]

    name = []

    if config.print_module:
        module = inspect.getmodule(parentframe)
        # `modname` can be None when frame is executed directly in console
        if module:
            name.append(module.__name__ + '.')

    # detect classname
    if 'self' in parentframe.f_locals:
        # I don't know any way to detect call from the object method
        # there seems to be no way to detect static method call - it will
        # be just a function call

        name.append(parentframe.f_locals['self'].__class__.__name__ + '::')

    codename = parentframe.f_code.co_name
    if codename != '<module>':  # top level usually
        name.append(codename + '(): ')  # function or a method
    else:
        name.append(': ')  # If called from module scope

    del parentframe
    return "".join(name)


def info(min_dbg_level, *message, **kwargs):
    """Print to console if `min_debug_level <= config.debug_level`

    The fuction determines automatically the name of caller and appends
    the message to it. Message can be a tuple of strings or objects
    which can be converted to string using `str()`.

    Args:
        min_dbg_level (int): Minimum debug level in config for printing
        message (tuple): Any argument or list of arguments that casts to str
        condition (bool): Print only if condition is True
        blank_caller (bool): blank the caller name (for multiline output)
        no_caller (bool): don't print the name of the caller
    """
    condition = kwargs.pop('condition', min_dbg_level <= config.debug_level)
    # Dont' process the if the function if nothing will happen
    if not (condition or config.override_debug_fcn): 
        return

    blank_caller = kwargs.pop('blank_caller', False)
    no_caller = kwargs.pop('no_caller', False)
    if config.override_debug_fcn and min_dbg_level < config.override_max_level:
        fcn_name = caller_name(skip=2).split('::')[-1].split('():')[0]
        if fcn_name in config.override_debug_fcn:
            min_dbg_level = 0

    if condition and min_dbg_level <= config.debug_level:
        message = [str(m) for m in message]
        cname = caller_name() if not no_caller else ''
        if blank_caller:
            cname = len(cname) * ' '
        print(cname + " ".join(message))

def get_AZN(nco_id):
    """Returns mass number :math:`A`, charge :math:`Z` and neutron
    number :math:`N` of ``nco_id``.

    Args:
        nco_id (int): corsika id of nucleus/mass group
    Returns:
        (int,int,int): (Z,A) tuple
    """
    Z, A = 1, 1

    if nco_id >= 100:
        Z = nco_id % 100
        A = (nco_id - Z) // 100
    else:
        Z, A = 0, 0

    return A, Z, A - Z

def bin_widths(bin_edges):
    """Computes and returns bin widths from given edges."""
    edg = np.array(bin_edges)

    return np.abs(edg[1:, ...] - edg[:-1, ...])

class AdditiveDictionary(dict):
    """This dictionary subclass adds values if keys are
    are already present instead of overwriting. For value tuples
    only the second argument is added and the first kept to its
    original value."""

    def __setitem__(self, key, value):
        if key not in self:
            super(AdditiveDictionary, self).__setitem__(key, value)
        elif isinstance(value, tuple):
            super(AdditiveDictionary, self).__setitem__(
                key, (self[key][0], value[1] + self[key][1]))
        else:
            super(AdditiveDictionary, self).__setitem__(
                key, self[key] + value)
            
class PrinceBDF(BDF):
    """This is a modified version of :class:`scipy.integrate.BDF` solver,
    that avoids oscillations that triggers excessive Jacobian updates.
    This improves solutions for CR propagation for z>1."""
    
    def _step_impl(self):
        from scipy.integrate._ivp.bdf import (
            change_D, solve_bdf_system, NEWTON_MAXITER,
            MIN_FACTOR, MAX_FACTOR, MAX_ORDER)
        from scipy.integrate._ivp.common import norm            
        t = self.t
        D = self.D

        max_step = self.max_step
        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)
        if self.h_abs > max_step:
            h_abs = max_step
            change_D(D, self.order, max_step / self.h_abs)
            self.n_equal_steps = 0
        elif self.h_abs < min_step:
            h_abs = min_step
            change_D(D, self.order, min_step / self.h_abs)
            self.n_equal_steps = 0
        else:
            h_abs = self.h_abs

        atol = self.atol
        rtol = self.rtol
        order = self.order

        alpha = self.alpha
        gamma = self.gamma
        error_const = self.error_const

        J = self.J
        LU = self.LU
        current_jac = self.jac is None

        step_accepted = False
        while not step_accepted:
            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP

            h = h_abs * self.direction
            t_new = t + h

            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound
                change_D(D, order, np.abs(t_new - t) / h_abs)
                self.n_equal_steps = 0
                LU = None

            h = t_new - t
            h_abs = np.abs(h)

            y_predict = np.sum(D[:order + 1], axis=0)

            scale = atol + rtol * np.abs(y_predict)
            psi = np.dot(D[1: order + 1].T, gamma[1: order + 1]) / alpha[order]

            converged = False
            c = h / alpha[order]
            while not converged:
                if LU is None:
                    LU = self.lu(self.I - c * J)

                converged, n_iter, y_new, d = solve_bdf_system(
                    self.fun, t_new, y_predict, c, psi, LU, self.solve_lu,
                    scale, self.newton_tol)

                if not converged:
                    if current_jac:
                        break
                    J = self.jac(t_new, y_predict)
                    LU = None
                    current_jac = True

            if not converged:
                factor = 0.5
                h_abs *= factor
                change_D(D, order, factor)
                self.n_equal_steps = 0
                LU = None
                continue

            safety = round(0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER
                                                       + n_iter), ndigits=15)

            scale = atol + rtol * np.abs(y_new)
            error = error_const[order] * d
            error_norm = norm(error / scale)

            if error_norm > 1:
                factor = max(MIN_FACTOR,
                             safety * error_norm ** (-1 / (order + 1)))
                h_abs *= factor
                change_D(D, order, factor)
                self.n_equal_steps = 0
                # As we didn't have problems with convergence, we don't
                # reset LU here.
            else:
                step_accepted = True

        self.n_equal_steps += 1

        self.t = t_new
        self.y = y_new

        self.h_abs = h_abs
        self.J = J
        self.LU = LU

        # Update differences. The principal relation here is
        # D^{j + 1} y_n = D^{j} y_n - D^{j} y_{n - 1}. Keep in mind that D
        # contained difference for previous interpolating polynomial and
        # d = D^{k + 1} y_n. Thus this elegant code follows.
        D[order + 2] = d - D[order + 1]
        D[order + 1] = d
        for i in reversed(range(order + 1)):
            D[i] += D[i + 1]

        if self.n_equal_steps < order + 1:
            return True, None

        if order > 1:
            error_m = error_const[order - 1] * D[order]
            error_m_norm = norm(error_m / scale)
        else:
            error_m_norm = np.inf
        
        

        if order < MAX_ORDER:
            error_p = error_const[order + 1] * D[order + 2]
            error_p_norm = norm(error_p / scale)
        else:
            error_p_norm = np.inf

        error_norms = np.array([error_m_norm, error_norm, error_p_norm])
        with np.errstate(divide='ignore'): 
            factors = error_norms ** (-1 / np.arange(order, order + 3))

        delta_order = np.argmax(factors) - 1
        order += delta_order
        self.order = order

        factor = min(MAX_FACTOR, safety * np.max(factors))
        
        # # This is the custom modification for PriNCe
        if round(self.h_abs * factor, ndigits=15) > self.max_step:
            if round(self.h_abs,ndigits=15) != self.max_step:
                change_D(D, order, max_step / self.h_abs)
                self.h_abs = self.max_step
                self.n_equal_steps = 0
                self.LU = None
            self.n_equal_steps = 0
            return True, None
        # custom modications end
        self.h_abs *= factor
        change_D(D, order, factor)
        self.n_equal_steps = 0
        self.LU = None

        return True, None

class PrinceProgressBar(object):
    """This is a wrapper around tqdm to process some prince
    argument handling, making it optional, for notebooks and
    python scripts using the bar_type argument."""
    def __init__(self, bar_type=None, nsteps=None):
        if bar_type == None or bar_type == False:
            self.pbar = None
        elif bar_type == 'notebook':
            from tqdm import tqdm_notebook as tqdm
            self.pbar = tqdm(total=nsteps)
            self.pbar.update()
        else:
            from tqdm import tqdm
            self.pbar = tqdm(total=nsteps)
            self.pbar.update()

    def __enter__(self):
        return self
    
    def update(self):
        if self.pbar is not None:
            self.pbar.update()
    
    def __exit__(self, type, value, traceback):
        if self.pbar is not None:
            self.pbar.close()