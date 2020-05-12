"""PriNCe configuration module."""

import os.path as path
import platform
import sys
import warnings

import numpy as np

base_path = path.dirname(path.abspath(__file__))

#: Debug flag for verbose printing, 0 silences PriNCe entirely
debug_level = 1
#: Printout debug info only for functions in this list (just give the name,
#: "get_solution" for instance) Warning, this option slows down initialization
#: by a lot. Use only when needed.
override_debug_fcn = []
#: Override debug printout for debug levels < value for the functions above
override_max_level = 10
#: Print module name in debug output
print_module = False

# =================================================================
# Paths and library locations
# =================================================================

#: Directory where the data files for the calculation are stored
data_dir = path.join(base_path, 'data')

#: PrinceDB file name
db_fname = 'prince_db_05.h5'

#: Model file for redistribution functions (from SOPHIA or similar)
redist_fname = 'sophia_redistribution_logbins.npy'


#=========================================================================
# Physics configuration
#=========================================================================

#: Cosmological parameters

#: Hubble constant
H_0 = 70.5  #km s^-1 Mpc^-1
H_0s = 2.28475e-18  #s^-1

#: Omega_m
Omega_m = 0.27

#: Omega_Lambda
Omega_Lambda = 0.73

#: CMB energy kB*T0 [GeV]
E_CMB = 2.34823e-13  

#===========================================================================
# Grids
#===========================================================================

#: Cosmic ray energy grid (defines system size for solver)
#: Number of bins in multiples of 4 recommended for maximal vectorization
#: efficiency for 256 bit AVX or similar
#: Format (log10(E_min), log10(E_max), nbins/decade of energy)
cosmic_ray_grid = (3, 14, 8)
#: Photon grid of target field, only for calculation of rates
photon_grid = (-15, -6, 8)

#: Scale of the energy grid
#:'E': logarithmic in energy E_i = E_min * (Delta)^i
#:'logE': linear grid in x = log_10(E): x_i = x_min + i * Delta
grid_scale ='E'

#: Order of semi-lagrangian for energy derivative 
semi_lagr_method ='5th_order'

#===========================================================================
# Model options
#===========================================================================

#: Threshold lifetime value for explicit transport of particles of this type. It
#: means that if a particle is unstable with lifetime smaller than this threshold,
#: it will be decayed until all final state particles of this chain are stable.
#: In other words: short intermediate states will be integrated out
tau_dec_threshold = np.inf # All unstable particles decay
# tau_dec_threshold = 0.  # None unstable particles decay
# tau_dec_threshold = 850. # This value is for stable neutrons

#: Particle ID for which redistribution functions are needed to be taken into
#: account. The default value is 101 (proton). All particles with smaller
#: IDs, i.e. neutrinos, pions, muons etc., will have energy redistributions.
#: For larger IDs (nuclei) the boost conservation is employed.
redist_threshold_ID = 101

#: Cut on energy redistribution functions
#: Resitribution below this x value are set to 0.
#: "x_cut" : 0.,
#: "x_cut_proton" : 0.,
x_cut = 1e-4
x_cut_proton = 1e-1

#: cut on photon energy, cross section above y = E_cr e_ph / m_cr does not contribute
y_cut = np.inf

# Build equation system up to a maximal nuclear mass of
max_mass = np.inf

# Include secondaries like photons and neutrinos
secondaries = True
# List of specific particles to ignore
ignore_particles = [20,21] # (we ignore photons and electrons, as their physics is not fully implemented)

#===========================================================================
# Parameters of numerical integration
#===========================================================================

# Update rates at not more frequently than this value in z
update_rates_z_threshold = 0.01

# #Number of MKL threads (for sparse matrix multiplication the performance
# #advantage from using more than a few threads is limited by memory bandwidth)
MKL_threads = 4

# Sparse matrix-vector product from "CUPY"|"MKL"|"scipy"
linear_algebra_backend = "MKL"


# Check for CUPY library for GPU support
try:
    import cupy
    has_cupy = True
    mempool = cupy.get_default_memory_pool()
    mempool.free_all_blocks()
except ModuleNotFoundError:
    print('CUPY not found for GPU support. Degrading to MKL.')
    if linear_algebra_backend == 'cupy':
        linear_algebra_backend = 'MKL'
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
    global mkl, MKL_threads
    from ctypes import cdll, byref, c_int
    mkl = cdll.LoadLibrary(mkl_path)
    # Set number of threads
    MKL_threads = nthreads
    mkl.mkl_set_num_threads(byref(c_int(nthreads)))
    if debug_level >= 5:
        print('MKL threads limited to {0}'.format(nthreads))

if has_mkl:
    set_mkl_threads(MKL_threads)

if not has_mkl and linear_algebra_backend.lower() == 'mkl':
    print('MKL runtime not found. Degrading to scipy.')
    linear_algebra_backend = 'scipy'

def _download_file(url, outfile):
    """Downloads the PriNCe database from github release binaries."""

    from tqdm import tqdm
    import requests
    import math

    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)

    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024 * 1024
    wrote = 0
    with open(outfile, 'wb') as f:
        for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size // block_size),
                         unit='MB', unit_scale=True):
            wrote = wrote + len(data)
            f.write(data)
    if total_size != 0 and wrote != total_size:
        raise Exception("ERROR, something went wrong")

# Download database file from github
base_url = 'https://github.com/joheinze/PriNCe/releases/download/'
release_tag = 'v0.5_alpha_release/'
url = base_url + release_tag + db_fname
if not path.isfile(path.join(data_dir, db_fname)):
    print('Downloading for PriNCe database file {0}.'.format(db_fname))
    if debug_level >= 2:
        print(url)
    _download_file(url, path.join(data_dir, db_fname))
else:
    import h5py
    try:
        with h5py.File(path.join(data_dir, db_fname), 'r') as prince_db:
            db_version = (prince_db.attrs['version'])
    except:
        print(f'Database file {db_fname} corrupted. Retrying download.')
        _download_file(url, path.join(data_dir, db_fname))
    finally:
        with h5py.File(path.join(data_dir, db_fname), 'r') as prince_db:
            db_version = (prince_db.attrs['version'])
        if debug_level >= 2:
            print(f'Using database file version {db_version}.')

# if path.isfile(path.join(data_dir, '...previous db name...')):
#     import os
#     print('Removing previous database {0}.'.format('...previous db name...'))
#     os.unlink(path.join(data_dir, '...previous db name...'))
