"""This module contains utility functions, which fulfill common puposes
in different modules of this project."""

import inspect

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline

from prince_config import config


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

    stack = inspect.stack()
    start = 0 + skip

    if len(stack) < start + 1:
        return ''

    parentframe = stack[start][0]

    name = []

    if config["print_module"]:
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
        print(caller_name() + " ".join(message))


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
    from os.path import join, splitext, isfile, isdir
    from os import listdir
    import numpy as np

    info(10, 'Loading file', fname)
    fname = splitext(fname)[0]

    if not isfile(join(config["data_dir"], fname + '.npy')):
        info(2, 'Converting', fname, "to '.npy'")
        arr = None
        try:
            arr = np.loadtxt(
                join(config['raw_data_dir'], fname + '.dat'), **kwargs)
        except IOError:
            for subdir in listdir(config['raw_data_dir']):
                if (isdir(join(config['raw_data_dir'], subdir)) and isfile(
                        join(config['raw_data_dir'], subdir, fname + '.dat'))):
                    arr = np.loadtxt(
                        join(config['raw_data_dir'], subdir, fname + '.dat'),
                        **kwargs)
        finally:
            if arr is None:
                raise Exception('Required file', fname + '.dat', 'not found')
            np.save(join(config["data_dir"], fname + '.npy'), arr)
        return arr
    else:
        return np.load(join(config["data_dir"], fname + '.npy'), encoding='latin1')


class AdditiveDictionary(dict):
    """This dictionary subclass adds values if keys are
    are already present instead of overwriting. For value tuples
    only the second argument is added and the first kept to its
    original value."""

    def __setitem__(self, key, value):
        if key not in self:
            self[key] = value
        elif isinstance(value, tuple):
            self[key] = (self[key][0], value[1] + self[key][1])
        else:
            self[key] += value
