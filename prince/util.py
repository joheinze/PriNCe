import inspect
from scipy.interpolate import InterpolatedUnivariateSpline
from prince_config import config

m_proton = 0.9382720813  # GeV


def e_nucleon(e_tot, p_id):
    return e_tot / A


def get_interp_object(xgrid, ygrid, **kwargs):
    """Returns simple standard interpolation object.

    Default type of interpolation is a spline of order
    one without extrapolation.

    Args:
        xgrid (numpy.array): x values of function
        ygrid (numpy.array): y values of function
    """
    if xgrid.shape != ygrid.shape:
        raise Exception('xgrid and ygrid args need identical shape.')

    if 'k' not in kwargs:
        kwargs['k'] = 1
    if 'ext' not in kwargs:
        kwargs['ext'] = 'zeros'

    return InterpolatedUnivariateSpline(xgrid, ygrid, **kwargs)


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
        A = (nco_id - Z) / 100
    else:
        raise Exception("get_AZN(): invalid nco_id", nco_id)

    return A, Z, A - Z


def get_y(E, eps, particle_id):
    A = particle_id / 100
    return E * eps / (A * m_proton)


def caller_name(skip=2):
    """Get a name of a caller in the format module.class.method

    `skip` specifies how many levels of stack to skip while getting caller
    name. skip=1 means "who calls me", skip=2 "who calls my caller" etc.
    An empty string is returned if skipped levels exceed stack height.abs

    From https://gist.github.com/techtonik/2151727
    """

    stack = inspect.stack()
    start = 0 + skip

    if len(stack) < start + 1:
        return ''

    parentframe = stack[start][0]

    name = []
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

    del parentframe
    return "".join(name)


def info(min_dbg_level, *message):
    if min_dbg_level > config["debug_level"]:
        message = [str(m) for m in message]
        print caller_name() + " ".join(message)


class EnergyGrid(object):
    def __init__(self, lower, upper, bins_dec):
        import numpy as np
        self.bins = np.logspace(lower, upper, (upper - lower) * bins_dec + 1)
        self.grid = 0.5 * (self.bins[1:] + self.bins[:-1])
        self.widths = self.bins[1:] - self.bins[:-1]
        info(0, 'energy grid initialized')
