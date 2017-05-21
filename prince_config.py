"""PriNCe configuration module."""

import os
import os.path as path
import platform
import sys
import cPickle as pickle
import numpy as np

base = path.dirname(path.abspath(__file__))
sys.path.append(base)
# sys.path.append(base+"/CRFluxModels")

#detrmine shared library extension and MKL path
lib_ext = None
mkl_default = path.join(sys.prefix, 'lib', 'libmkl_rt')

if platform.platform().find('Linux') != -1:
    lib_ext = '.so'
elif platform.platform().find('Darwin') != -1:
    lib_ext = '.dylib'
else:
    #Windows case
    mkl_default = path.join(sys.prefix, 'pkgs', 'mkl-11.3.3-1', 'Library',
                            'bin', 'mkl_rt')
    lib_ext = '.dll'

config = {

    # Debug flag for verbose printing, 0 = minimum
    "debug_level": 3,

    #=========================================================================
    # Paths and library locations
    #=========================================================================

    # Directory where the data files for the calculation are stored
    "data_dir": path.join(base, 'data'),
    # Directory for raw files if conversion of some sort is needed
    "raw_data_dir": path.join(base, 'utils'),
    # # nuclear cross sections
    # "data_dir": '/data',
    # # File name of particle production yields
    # "yield_fname": "yield_dict.ppd",

    # full path to libmkl_rt.[so/dylib] (only if kernel=='MKL')
    "MKL_path": mkl_default + lib_ext,

    #=========================================================================
    # Physics configuration
    #=========================================================================
    # Cosmological parameters

    # Hubble constant
    "H_0": 70.5,  #km s^-1 Mpc^-1
    "H_0s": 2.28475e-18,  #s^-1

    # Omega_m
    "Omega_m": 0.27,

    # Omega_Lambda
    "Omega_Lambda": 0.73,

    #===========================================================================
    # Grids
    #===========================================================================
    # Format (log10(E_min), log10(E_max), nbins/decade of energy)
    # Main energy grid for solver
    "cosmic_ray_grid": (7, 13, 10),

    # Photon grid of target field, only for calculation of rates
    "photon_grid": (-15, -8, 10),

    #===========================================================================
    # Model options
    #===========================================================================
    # The sophia tables are on a grid with 2000 points. The number will use every
    # N-th entry of the table to reduce memory usage of the interpolator
    "sophia_grid_skip": 4,
    # Threshold lifetime value to consider a particle as woth propagating. It
    # means that if a particle is unstable with lifetime smaller than this threshold
    # will be decayed until all final state particles of this chain are stable.
    # In other words: short intermediate states will be integrated out
    "tau_dec_threshold": np.inf,
    #===========================================================================
    # Parameters of numerical integration
    #===========================================================================

    # Selection of integrator (euler/odepack)
    "integrator": "euler",

    # euler kernel implementation (numpy/MKL/CUDA).
    "kernel_config": "MKL",

    #parameters for the odepack integrator. More details at
    #http://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html
    "ode_params": {
        'name': 'vode',
        'method': 'adams',
        'nsteps': 10000,
        'max_step': 10.0
    },

    # Use sparse linear algebra (recommended!)
    "use_sparse": True,

    #Number of MKL threads (for sparse matrix multiplication the performance
    #advantage from using more than 1 thread is limited by memory bandwidth)
    "MKL_threads": 24,

    # Float precision (32 only yields speed up with CUDA, MKL gets slower?)
    "FP_precision": 64,

    #=========================================================================
    # Advanced settings
    #=========================================================================

    # A more common setting
    "hybrid_crossover": 0.05,

    # Possibilities to control some more advanced stuff
    "adv_settings": {
        # Modify something in special way
        "some_setting": False,
    }
}

#: Dictionary containing particle properties, like mass, charge
#: lifetime or branching ratios
spec_data = pickle.load(
    open(path.join(config["data_dir"], "particle_data.ppo"), "rb"))
