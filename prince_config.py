"""PriNCe configuration module."""

import os
import os.path as path
import platform
import sys
import pickle as pickle
import numpy as np

base = path.dirname(path.abspath(__file__))
sys.path.append(base)


config = {

    # Debug flag for verbose printing, 0 = minimum
    "debug_level": 3,

    # When printing output, prepend module name
    "print_module" : False,

    #=========================================================================
    # Paths and library locations
    #=========================================================================

    # Location of the database
    "data_dir": path.join(base, 'prince', 'data'),
    # PrinceDB file name
    "db_fname": 'prince_db_05.h5',
    # Model file for redistribution functions (from SOPHIA or similar)
    "redist_fname": "sophia_redistribution_logbins.npy",

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

    "E_CMB" : 2.34823e-13,  # = kB*T0 [GeV]

    #===========================================================================
    # Grids
    #===========================================================================
    # Number of bins in multiples of 4 recommended for maximal vectorization
    # efficiency for 256 bit AVX or similar

    # Format (log10(E_min), log10(E_max), nbins/decade of energy)
    # Main energy grid for solver
    "cosmic_ray_grid": (3, 14, 8),
    # Photon grid of target field, only for calculation of rates
    "photon_grid": (-15, -6, 8),
    # Scale of the energy grid
    # 'E': logarithmic in energy E_i = E_min * (Delta)^i
    # 'logE': linear grid in x = log_10(E): x_i = x_min + i * Delta
    "grid_scale":'E',

    "semi_lagr_method":'5th_order',

    #===========================================================================
    # Model options
    #===========================================================================
    # The sophia tables are on a grid with 2000 points. The number will use every
    # N-th entry of the table to reduce memory usage of the interpolator
    # "sophia_grid_skip": 4,

    # Threshold lifetime value to consider a particle as worth propagating. It
    # means that if a particle is unstable with lifetime smaller than this threshold
    # will be decayed until all final state particles of this chain are stable.
    # In other words: short intermediate states will be integrated out
    "tau_dec_threshold": np.inf, # All unstable particles decay
    # "tau_dec_threshold": 0.,   # No unstable particles decay
    # "tau_dec_threshold": 850., # This value is for stable neutrons

    # Particle ID for which redistribution functions are needed to be taken into
    # account. The default value is 101 (proton). All particles with smaller
    # IDs, i.e. neutrinos, pions, muons etc., will have energy redistributions.
    # For larger IDs (nuclei) the boost conservation is employed.
    "redist_threshold_ID": 101,

    # Cut on redistribution functions
    # Resitribution below this x value are set to 0.
    # "x_cut" : 0.,
    # "x_cut_proton" : 0.,
    "x_cut" : 1e-4,
    "x_cut_proton" : 1e-1,

    # cut on photon energy, cross section above y = E_cr e_ph / m_cr does not contribute
    "y_cut": np.inf,

    # Build equation system up to a maximal nuclear mass of
    "max_mass": np.inf,

    # Include secondaries like photons and neutrinos
    "secondaries": True,
    # List of specific particles to ignore
    "ignore_particles": [20,21], #(we ignore photons and electrons, as their physics is not fully implemented)

    #===========================================================================
    # Parameters of numerical integration
    #===========================================================================

    # Update threshold of rates/cross sections
    "update_rates_z_threshold": 0.01,

    # #Number of MKL threads (for sparse matrix multiplication the performance
    # #advantage from using more than 1 thread is limited by memory bandwidth)
    "MKL_threads": 4,

    # Sparse matrix-vector product from "CUPY"|"MKL"|"scipy"
    "linear_algebra_backend": "MKL",

    # Parameters for the lsodes integrator. 
    "ode_params": {
        'name': 'lsodes',
        'method': 'bdf',
        'rtol': 1e-6,
        'atol': 1e68,
        'tcrit': None,
        # 'max_order_s': 2,
        # 'with_jacobian': True
    },

    # # Selection of integrator (euler/odepack)
    # "integrator": "euler",

    #=========================================================================
    # Advanced settings
    #=========================================================================

    # Possibilities to control some more advanced stuff
    "adv_settings": {
        # Modify something in special way
        "some_setting": False,
    }
}

# Check for CUPY library for GPU support
try:
    import cupy
    has_cupy = True
    mempool = cupy.get_default_memory_pool()
    mempool.free_all_blocks()
except ModuleNotFoundError:
    print('CUPY not found for GPU support. Degrading to MKL.')
    if config["linear_algebra_backend"] == 'cupy':
        config["linear_algebra_backend"] = 'MKL'
    has_cupy = False

#: determine shared library extension and MKL path
pf = platform.platform()

if 'Linux' in pf:
    mkl_path = path.join(sys.prefix, 'lib', 'libmkl_rt.so')
elif 'Darwin' in pf:
    mkl_path = path.join(sys.prefix, 'lib', 'libmkl_rt.dylib')
else:
    # Windows case
    mkl_path = path.join(sys.prefix, 'Library', 'bin', 'mkl_rt.dll')

# mkl library handler
mkl = None

# Check if MKL library found
if path.isfile(mkl_path):
    has_mkl = True
else:
    has_mkl = False

def set_mkl_threads(nthreads):
    global mkl
    from ctypes import cdll, c_int, byref
    mkl = cdll.LoadLibrary(mkl_path)
    # Set number of threads
    config["MKL_threads"] = nthreads
    mkl.mkl_set_num_threads(byref(c_int(nthreads)))
    if config['debug_level'] >= 5:
        print('MKL threads limited to {0}'.format(nthreads))

if has_mkl:
    set_mkl_threads(config["MKL_threads"])

if not has_mkl and config["linear_algebra_backend"].lower() == 'mkl':
    print('MKL runtime not found. Degrading to scipy.')
    config["linear_algebra_backend"] = 'scipy'
