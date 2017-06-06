"""This module contains utility functions, which fulfill common puposes
in different modules of this project."""

import inspect
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.constants as spc
from prince_config import config


def convert_to_namedtuple(dictionary, name='GenericNamedTuple'):
    """Converts a dictionary to a named tuple."""
    from collections import namedtuple
    return namedtuple(name, dictionary.keys())(**dictionary)


# Default units in Prince are ***cm, s, GeV***
# Define here all constants and unit conversions and use
# throughout the code. Don't write c=2.99.. whatever.
# Write clearly which units a function returns.
# Convert them if not standard unit
# Accept only arguments in the units above

units_and_conversions_def = dict(
    c=1e2 * spc.c,
    cm2Mpc=1. / (spc.parsec * spc.mega * 1e2),
    Mpc2cm=spc.mega * spc.parsec * 1e2,
    m_proton=spc.physical_constants['proton mass energy equivalent in MeV'][0]
    * 1e-3,
    GeV2erg=1. / 624.15,
    erg2GeV=624.15,
    km2cm=1e5,
    Gyr2sec=spc.giga*spc.year,
    cm2sec=1e-2/spc.c,
    yr2sec = 365 * 24 * 60 * 60,
    sec2yr = 1 / (365 * 24 * 60 * 60),
    )

#This is the immutable unit object to be imported throughout the code
pru = convert_to_namedtuple(units_and_conversions_def,
                            "PriNCeUnits")

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


def e_nucleon(e_tot, nco_id):
    """Converts energy in energy per nucleon"""
    A, _, _ = get_AZN(nco_id)
    return e_tot / A


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


def get_y(e, eps, nco_id):
    """Retrns center of mass energy of nucleus-photon system.

    Args:
        e (float): energy (vector) of nucleus(on) in GeV
        eps (float): photon energy in GeV
        nco_id (int): particle index

    Returns:
        (float): center of mass energy :math:`y`
    """

    A = get_AZN(nco_id)[0]

    return e * eps / (A * pru.m_proton)


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
    """Print to console if `min_debug_level <= config["debug_level"]`

    The fuction determines automatically the name of caller and appends
    the message to it. Message can be a tuple of strings or objects
    which can be converted to string using `str()`.

    Args:
        min_dbg_level (int): Minimum debug level in config for printing
        message (tuple): Any argument or list of arguments that casts to str
    """

    if min_dbg_level <= config["debug_level"]:
        message = [str(m) for m in message]
        print caller_name() + " ".join(message)


def load_or_convert_array(fname, **kwargs):
    """ Loads an array from '.npy' file if exists otherwise
    the array is created from CVS file.

    `fname` is expected to be just the file name, without folder.
    The CVS file is expected to be in the `raw_data_dir` directory
    pointed to by the config. The array from the file is stored
    as numpy binary with extension `.npy` in the folder pointed
    by the `data_dir` config variable.

    Args:
        fname (str): File name without path or ending
        kwargs (dict): Is passed to :func:`numpy.loadtxt`
    Returns:
        (numpy.array): Array stored in that file
    """
    from os.path import join, splitext, isfile
    import numpy as np
    fname = splitext(fname)[0]
    info(2, 'Loading file', fname)
    if not isfile(join(config["data_dir"], fname + '.npy')):
        info(2, 'Converting', fname, "to '.npy'")
        arr = np.loadtxt(
            join(config['raw_data_dir'], fname + '.dat'), **kwargs)
        np.save(join(config["data_dir"], fname + '.npy'), arr)
        return arr
    else:
        return np.load(join(config["data_dir"], fname + '.npy'))


class EnergyGrid(object):
    """Class for constructing a grid for discrete distributions.

    Since we discretize everything in energy, the name seems appropriate.
    All grids are log spaced.

    Args:
        lower (float): log10 of low edge of the lowest bin
        upper (float): log10 of upper edge of the highest bin
    """

    def __init__(self, lower, upper, bins_dec):
        import numpy as np
        self.bins = np.logspace(lower, upper, (upper - lower) * bins_dec + 1)
        self.grid = 0.5 * (self.bins[1:] + self.bins[:-1])
        self.widths = self.bins[1:] - self.bins[:-1]
        self.d = self.grid.size
        info(1, 'Energy grid initialized {0:3.1e} - {1:3.1e}, {2} bins'.format(
            self.bins[0], self.bins[-1], self.grid.size))


def dict_add(di, key, value):
    """Adds value to previous value of di[key], otherwise the key
    is created with value set to `value`."""

    if key in di:
        di[key] += value
    else:
        di[key] = value
