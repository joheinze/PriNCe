"""This module contains utility functions, which fulfill common puposes
in different modules of this project."""

import inspect
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
import scipy.constants as spc
import numpy as np
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

UNITS_AND_CONVERSIONS_DEF = dict(
    c=1e2 * spc.c,
    cm2Mpc=1. / (spc.parsec * spc.mega * 1e2),
    Mpc2cm=spc.mega * spc.parsec * 1e2,
    m_proton=spc.physical_constants['proton mass energy equivalent in MeV'][0]
    * 1e-3,
    m_electron=spc.physical_constants[
        'electron mass energy equivalent in MeV'][0] * 1e-3,
    r_electron=spc.physical_constants['classical electron radius'][0] * 1e2,
    fine_structure=spc.fine_structure,
    GeV2erg=1. / 624.15,
    erg2GeV=624.15,
    km2cm=1e5,
    yr2sec=spc.year,
    Gyr2sec=spc.giga * spc.year,
    cm2sec=1e-2 / spc.c,
    sec2cm=spc.c * 1e2)

# This is the immutable unit object to be imported throughout the code
PRINCE_UNITS = convert_to_namedtuple(UNITS_AND_CONVERSIONS_DEF, "PriNCeUnits")


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
        Z,A = 0,0

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
        raise Exception('x and y grid do not match z grid shape: {0} != {1}'.
                        format((xgrid.shape, ygrid.shape), zgrid.shape))

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

    return e * eps / (A * PRINCE_UNITS.m_proton)


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
        return np.load(join(config["data_dir"], fname + '.npy'))


class EnergyGrid(object):
    """Class for constructing a grid for discrete distributions.

    Since we discretize everything in energy, the name seems appropriate.
    All grids are log spaced.

    Args:
        lower (float): log10 of low edge of the lowest bin
        upper (float): log10 of upper edge of the highest bin
        bins_dec (int): bins per decade of energy
    """

    def __init__(self, lower, upper, bins_dec):
        self.bins = np.logspace(lower, upper, (upper - lower) * bins_dec + 1)
        self.grid = 0.5 * (self.bins[1:] + self.bins[:-1])
        self.widths = self.bins[1:] - self.bins[:-1]
        self.d = self.grid.size
        info(5, 'Energy grid initialized {0:3.1e} - {1:3.1e}, {2} bins'.format(
            self.bins[0], self.bins[-1], self.grid.size))

class LogEnergyGrid(object):
    """Class for constructing a grid for discrete distributions.

    Since we discretize everything in energy, the name seems appropriate.
    All grids are log spaced.

    Args:
        lower (float): log10 of low edge of the lowest bin
        upper (float): log10 of upper edge of the highest bin
        bins_dec (int): bins per decade of energy
    """

    def __init__(self, lower, upper, bins_dec):
        self.bins = np.linspace(lower, upper, (upper - lower) * bins_dec + 1)
        self.grid = 0.5 * (self.bins[1:] + self.bins[:-1])
        self.bins = 10**self.bins
        self.grid = 10**self.grid
        self.widths = self.bins[1:] - self.bins[:-1]
        self.d = self.grid.size
        info(1, 'LogEnergy grid initialized {0:3.1e} - {1:3.1e}, {2} bins'.format(
            self.bins[0], self.bins[-1], self.grid.size))


def dict_add(di, key, value):
    """Adds value to previous value of di[key], otherwise the key
    is created with value set to `value`."""

    if key in di:
        if isinstance(value, tuple):
            new_value = value[1] + di[key][1]
            di[key] = (di[key][0], new_value)
            # The code below is a template what to do
            # if energy grids are unequal and one needs to
            # sum over common indices
            # try:
            # If energy grids are the same
            # new_value = value[1] + di[key][1]
            # value[1] += di[key][1]
            # di[key] = (di[key][0], new_value)
            # except ValueError:
            #     # old_egr = di[key][0]
            #     # old_val = di[key][1]
            #     # new_egr = value[0]
            #     # new_val = value[1]
            #     # if prevval.shape[1] > value.shape[1]:
            #     #     idcs = np.in1d(di[key][0], value)
            #     #     value += prevval[idcs]
            #     print key, value[1].shape, di[key][1].shape, '\n', value, '\n', di[key]
            #     raise Exception()

        else:
            di[key] += value
    else:
        di[key] = value


def bin_centers(bin_edges):
    """Computes and returns bin centers from given edges."""
    edg = np.array(bin_edges)
    return 0.5 * (edg[1:] + edg[:-1])


def bin_widths(bin_edges):
    """Computes and returns bin widths from given edges."""
    edg = np.array(bin_edges)

    return np.abs(edg[1:, ...] - edg[:-1, ...])


def bin_edges2D(bin_centers):
    lcen = np.log10(bin_centers)
    steps = lcen[1, ...] - lcen[0, ...]
    bins_log = np.zeros_like(lcen)  #(len(lcen) + 1)
    # print bins_log.shape
    bins_log = np.pad(
        bins_log, ((0, 1), (0, 0)), 'constant', constant_values=0.)
    # print bins_log.shape
    bins_log[:lcen.shape[0], ...] = lcen - 0.5 * steps
    bins_log[-1, ...] = lcen[-1, ...] + 0.5 * steps
    return 10**bins_log


def bin_edges1D(bin_centers):
    lcen = np.log10(bin_centers)
    steps = lcen[1] - lcen[0]
    bins_log = np.zeros(len(lcen) + 1)
    bins_log[:lcen.shape[0]] = lcen - 0.5 * steps
    bins_log[-1] = lcen[-1] + 0.5 * steps
    return 10**bins_log



class TheInterpolator(object):
    def __init__(self, bins):
        self.n_ext = 2
        self.n_window = 3
        if self.n_window % 2 == 0:
            raise Exception('Window size must be odd.')
        self.bins = bins
        self._init_grids()
        self._init_matrices()
        
    def _init_grids(self):
        #Bin edges extended by n_ext points on both sides
        nwi2 = (self.n_window - 1)/2
        self.bins_ext = np.zeros(self.bins.size + 2*self.n_ext+self.n_window-1)
        grid_spacing = np.log10(self.bins[1]/self.bins[0])
        # Copy nominal grid
        self.bins_ext[nwi2 + self.n_ext:-nwi2 - self.n_ext] = self.bins
        for i in range(1,self.n_ext + nwi2 + 1):
            self.bins_ext[
                nwi2 + self.n_ext - i] = self.bins[0]*10**(-grid_spacing*i)
            self.bins_ext[
                nwi2 + self.n_ext + self.bins.size  + i - 1
            ] = self.bins[-1]*10**(grid_spacing*i)
        
        self.dim = self.bins.size - 1
        self.dim_ext = self.dim + 2*self.n_ext
        
        self.grid = np.sqrt(self.bins[1:]*self.bins[:-1])
        self.grid_ext = np.sqrt(self.bins_ext[1:]*self.bins_ext[:-1])
        
        self.widths = self.bins[1:] - self.bins[:-1]
        self.widths_ext = self.bins_ext[1:] - self.bins_ext[:-1]
        
        self.b = np.zeros(self.n_window*self.dim_ext)
        
    def _init_matrices(self):
        from scipy.sparse.linalg import factorized
        from scipy.sparse import csc_matrix
        
        intp_mat = np.zeros((self.n_window*self.dim_ext,
                                  self.n_window*self.dim_ext))
        sum_mat = np.zeros((self.dim_ext,self.n_window*self.dim_ext))
        
        nex = self.n_ext
        nwi = self.n_window
        nwi2 = (self.n_window - 1)/2
        # print self.dim_ext
        for i in range(0,self.dim_ext):
            for m in range(nwi):
                intp_mat[nwi*i + m, nwi*i:nwi*(1 +i)] = (
                    self.grid_ext[i:i+nwi]**m*self.widths_ext[i:i+nwi])
        
        idx = lambda i: [(i-k)*nwi + k + nwi2 for k in range(-nwi2, nwi2 +1) 
                         if 0 <= ((i-k)*nwi + k + nwi2) < self.n_window*self.dim_ext]
        
        for i in range(self.dim_ext):
            sum_mat[i,idx(i)] = 1.

        self.intp_mat=csc_matrix(intp_mat)
        self.sum_mat=csc_matrix(sum_mat)
        
        self.solver = factorized(self.intp_mat)
        
    def set_initial_delta(self, norm, energy):
        # Setup initial state
        self.b *= 0.
        cenbin = np.argwhere(energy < self.bins_ext)[0][0] - 1
        if cenbin < 0:
            # print 'energy too low', energy
            raise Exception()
#       print energy, cenbin, self.bins_ext[cenbin:cenbin+2]
        norm*=self.widths_ext[cenbin]
        for m in range(self.n_window):
            self.b[self.n_window*cenbin + m] = norm*energy**m
            
    def set_initial_spectrum(self, fx, fy):
        self.b *= 0.
        for i, x in enumerate(fx):
            cenbin = np.argwhere(x < self.bins_ext)[0][0] - 1
            # print i, x, cenbin, self.bins_ext[cenbin:cenbin+2]
            if cenbin < 0:
                continue
            for m in range(0,self.n_window):
                self.b[self.n_window*cenbin + m] += fy[i]*x**m
    
    def set_initial_spectrum2(self, fx, fy):
        self.b *= 0.
        for m in range(0,self.n_window):
            self.b[
                self.n_ext*self.n_window + m:-self.n_ext*self.n_window + m
            :self.n_window] += fy*fx**m
    
    def get_solution(self):
        return self.sum_mat.dot(self.solver(self.b))[self.n_ext:-self.n_ext]
    
    def get_moment(self, m):
        nwi2 = self.n_window/2
        return np.sum((self.widths_ext[nwi2:-nwi2]*
                self.grid_ext[nwi2:-nwi2]**m*
                self.sum_mat.dot(self.solver(self.b))))