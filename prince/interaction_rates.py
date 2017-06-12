"""The module contains classes for computations of interaction rates"""

import numpy as np
from scipy.sparse import coo_matrix, bmat

from prince.util import get_AZN, get_interp_object, info, load_or_convert_array, pru
from prince_config import config
from abc import ABCMeta, abstractmethod


class InteractionRateBase(object):
    __metaclass__ = ABCMeta

    def __init__(self, prince_run, *args, **kwargs):
        from prince.util import EnergyGrid, get_y

        #: Reference to prince run
        self.prince_run = prince_run

        #: Reference to PhotonField object
        self.photon_field = prince_run.photon_field

        #: Reference to CrossSection object
        self.cross_sections = prince_run.cross_sections

        #: Reference to species manager
        self.spec_man = prince_run.spec_man

        # Initialize grids
        if "photon_bins_per_dec" not in kwargs:
            self.e_photon = prince_run.ph_grid
        else:
            info(2, 'Overriding number of photon bins from config.')
            self.e_photon = EnergyGrid(config["photon_grid"][0],
                                       config["photon_grid"][1],
                                       kwargs["photon_bins_per_dec"])

        if "cr_bins_per_dec" not in kwargs:
            self.e_cosmicray = prince_run.cr_grid
        else:
            info(2, 'Overriding number of cosmic ray bins from config.')
            self.e_cosmicray = EnergyGrid(config["cosmic_ray_grid"][0],
                                          config["cosmic_ray_grid"][1],
                                          kwargs["cr_bins_per_dec"])

        # Create shortcuts for grid dimensions
        self.dim_cr = self.e_cosmicray.d
        self.dim_ph = self.e_photon.d

        # Initialize cache of redshift value
        self._ph_vec_zcache = None
        self._ratemat_zcache = None

        # Variable to cache the photon spectrum at current redshif
        self._photon_vector = None

        # The call to setup initializes all variables
        self._setup()

    @abstractmethod
    def _setup(self):
        """Implement all initialization in the derived class"""
        raise Exception("Base class function called.")

    def photon_vector(self, z):
        """Returns photon vector at redshift `z` on photon grid.

        Return value from cache if redshift value didn't change since last call.
        """

        self._set_photon_vector(z)

        return self._photon_vector

    def _set_photon_vector(self, z):
        """Cache photon vector for the previous value of z.

        Args:
            z (float): redshift
        """

        if self._ph_vec_zcache != z:
            self._photon_vector = self.photon_field.get_photon_density(
                self.e_photon.grid, z)
            self._ph_vec_zcache = z


class PhotoNuclearInteractionRate(InteractionRateBase):
    """Implementation of photo-hadronic/nuclear interaction rates."""

    def __init__(self, prince_run, *args, **kwargs):
        InteractionRateBase.__init__(self, prince_run, *args, **kwargs)

    def _setup(self):
        """Initialization of all members."""

        self._init_matrices()
        self._init_rate_matstruc()
        self._init_reinjection_mat()

    def interaction_rate_single(self, nco_ids, z):
        """Compute a single interaction rate using matrix convolution.

        This method is a high performance integration of equation (10)
        from internal note, using a simple box method.

        Don't use this method if you intend to compute rates for different
        species at the same redshift value. Use :func:`interaction_rate`
        instead.

        The last redshift value is cached to avoid interpolation of the
        photon spectrum at each step.

        Args:
            nco_id (int or tuple): single particle id (neucosma codes) or tuple
                                   with (mother, daughter) for inclusive
                                   reactions
            z (float): redshift

        Returns:
            (numpy.array): interaction length :math:`\\Gamma` in cm^-1
        """

        # Convolve using matrix multiplication
        if isinstance(nco_ids, tuple):
            return self.cross_sections.resp_incl_intp[nco_ids](self.ymat).dot(
                np.diag(self.e_photon.widths)).dot(self.photon_vector(z))
        else:
            return self.cross_sections.resp_nonel_intp[nco_ids](self.ymat).dot(
                np.diag(self.e_photon.widths)).dot(self.photon_vector(z))

    def interaction_rate(self, nco_ids, z):
        """Compute interaction rates using batch matrix convolution.

        This method is a high performance integration of equation (10)
        from internal note, using a simple box method.

        All rates for a certain redshift value are computed at once, thus
        this function is not very efficient if you need a rate for a single
        species at different redshifts values.

        The last redshift value is cached to avoid interpolation of the
        photon spectrum at each step.

        Args:
            nco_id (int or tuple): single particle id (neucosma codes) or tuple
                                   with (mother, daughter) for inclusive
                                   reactions
            z (float): redshift

        Returns:
            (numpy.array): interaction length :math:`\\Gamma` in cm^-1
        """

        self._update_rate_vec(z)

        if isinstance(nco_ids, tuple):
            lidx, ridx = self._incl_batchvec_pointer[nco_ids]
            return self._incl_batch_vec[lidx:ridx]
        else:
            lidx, ridx = self._nonel_batchvec_pointer[nco_ids]
            return self._nonel_batch_vec[lidx:ridx]

    def get_coupling_mat(self, z):
        """Returns the nonel rate vector and coupling matrix.
        """

        self._update_rate_vec(z)
        return self._nonel_batch_vec, self.reinjection_smat.multiply(
            bmat(self.incl_rate_struc))

    def _update_rate_vec(self, z):
        """Batch compute all nonel and inclusive rates if z changes.

        The result is always stored in the same vectors, since '_init_rate_matstruc'
        makes use of views to link ranges of the vector to locations in the matrix.
        """
        if self._ratemat_zcache != z:
            info(3, 'Updating batch rate vectors.')
            np.dot(
                self._nonel_batch_matrix,
                self.photon_vector(z),
                out=self._nonel_batch_vec)
            np.dot(
                self.incl_batch_matrix,
                self.photon_vector(z),
                out=self._incl_batch_vec)
            self._ratemat_zcache = z

    def _init_matrices(self):
        """Initializes batch computation of rates via matrices."""

        from prince.util import get_y
        # Disable pylint warning for since this method is always called
        # via _setup and the constructor
        # pylint: disable=W0201

        # Dense zero matrix
        self.zeros = np.zeros((self.dim_cr, self.dim_ph))

        # Sparse zero matrix
        self.sp_zeros = coo_matrix(
            (np.zeros(self.dim_cr), (np.arange(self.dim_cr),
                                     np.arange(self.dim_cr))),
            shape=(self.dim_cr, self.dim_cr),
            copy=False)

        # Iniialize cross section matrices, evaluated on a grid of y values
        x, y = np.meshgrid(self.e_photon.grid, self.e_cosmicray.grid)

        # Delta eps (photon energy) bin widths
        delta_eps = np.diag(self.e_photon.widths)

        # Compute y matrix only once and then rescale by A
        self.ymat = get_y(x, y, 100)

        # One big matrix for batch-compute all nonel rates at once
        self._nonel_batchvec_pointer = {}
        self._nonel_batch_matrix = np.zeros((self.dim_cr * self.spec_man.nspec,
                                             self.dim_ph))

        self._nonel_batch_vec = np.zeros(self._nonel_batch_matrix.shape[0])

        # Sparse matrix structure (see :func:`_init_rate_matstruc`)
        self.reinjection_struc = []

        info(2, 'Size of nonel batch matrix: {0}x{1}'.format(
            *self._nonel_batch_matrix.shape))

        # One big matrix for batch-compute all incl rates at once

        # [mother_ncoid,daughter_ncoid] -> _incl_batch_vec[lidx:ridx]
        self._incl_batchvec_pointer = {}

        # [mother_princeidx,daughter_princeidx] -> _incl_batch_vec[lidx:ridx]
        self._incl_batchvec_pridx_pointer = {}

        # Matrix: (incl-channels * dim_cr) x dim_ph
        self.incl_batch_matrix = np.zeros(
            (self.dim_cr * len(self.cross_sections.incl_idcs), self.dim_ph))

        # Result vector, which stores computed incl rates
        self._incl_batch_vec = np.zeros(self.incl_batch_matrix.shape[0])

        # Sparse matrix structure (see :func:`_init_rate_matstruc`)
        self.incl_rate_struc = []

        # Sparse matrix structure (see :func:`_init_reinjection_mat`)
        self.reinjection_smat = None

        info(2, 'Size of incl batch matrix: {0}x{1}'.format(
            *self.incl_batch_matrix.shape))

        # Define shortcut for converting ncoid to prince index
        pridx = self.spec_man.ncoid2princeidx

        fill_idx = 0
        for mother in self.spec_man.known_species:
            if mother < 100:
                info(3, "Can not compute interaction rate for", mother)
                fill_idx += 1
                continue

            A = get_AZN(mother)[0]
            lidx, ridx = fill_idx * self.dim_cr, (fill_idx + 1) * self.dim_cr

            self._nonel_batch_matrix[
                lidx:ridx] = self.cross_sections.resp_nonel_intp[mother](
                    self.ymat).dot(delta_eps)
            self._nonel_batchvec_pointer[mother] = (lidx, ridx)

            fill_idx += 1

        # Repeat the same stuff for inclusive channels
        fill_idx = 0
        for mother in self.spec_man.known_species:
            if mother < 100:
                info(3, 'Skip non-hadronic species')
                continue
            for (mo, da) in self.cross_sections.reactions[mother]:

                # Indices in batch matrix
                lidx, ridx = fill_idx * self.dim_cr, (
                    fill_idx + 1) * self.dim_cr

                # Staple ("vstack"") all inclusive (channel) response functions
                self.incl_batch_matrix[
                    lidx:ridx] = self.cross_sections.resp_incl_intp[(
                        mo, da)](self.ymat).dot(delta_eps)

                # Remember how to find the entry for a response function/rate in the
                # matrix or result vector
                self._incl_batchvec_pointer[(mo, da)] = (lidx, ridx)

                # Create association between prince_idx and position in resulting
                # rate vector
                mo_pridx, da_pridx = pridx[mo], pridx[da]
                self._incl_batchvec_pridx_pointer[(mo_pridx,
                                                   da_pridx)] = (lidx, ridx)

                fill_idx += 1

    def _init_rate_matstruc(self):
        """Initialize rate matrix structure.

        The initialization sets up the references to views of the rate
        vectors, and lays out the structure of the coupling rate matrix.
        """

        idcs = np.arange(self.dim_cr)
        shape_submat = (self.dim_cr, self.dim_cr)

        for da_pridx in range(self.prince_run.spec_man.nspec):
            self.incl_rate_struc.append([])
            for mo_pridx in range(self.prince_run.spec_man.nspec):
                if (mo_pridx,
                        da_pridx) not in self._incl_batchvec_pridx_pointer:
                    if mo_pridx == da_pridx:
                        self.incl_rate_struc[da_pridx].append(self.sp_zeros)
                    else:
                        self.incl_rate_struc[da_pridx].append(None)
                    continue
                lidx, ridx = self._incl_batchvec_pridx_pointer[(mo_pridx,
                                                                da_pridx)]
                self.incl_rate_struc[da_pridx].append(
                    coo_matrix(
                        (self._incl_batch_vec[lidx:ridx], (idcs, idcs)),
                        shape=shape_submat,
                        copy=False))

    def _init_reinjection_mat(self):
        # Initialize coupling matrix structure, containing the references
        # to views of the batch vector

        idcs = np.arange(self.dim_cr)
        shape_submat = (self.dim_cr, self.dim_cr)
        prnco = self.prince_run.spec_man.princeidx2ncoid

        for da_pridx in range(self.prince_run.spec_man.nspec):
            self.reinjection_struc.append([])
            for mo_pridx in range(self.prince_run.spec_man.nspec):
                if (mo_pridx,
                        da_pridx) not in self._incl_batchvec_pridx_pointer:
                    if mo_pridx == da_pridx:
                        self.reinjection_struc[da_pridx].append(self.sp_zeros)
                    else:
                        self.reinjection_struc[da_pridx].append(None)
                    continue

                B, A = float(get_AZN(prnco[da_pridx])[0]), float(
                    get_AZN(prnco[mo_pridx])[0])

                lidx, ridx = self._incl_batchvec_pridx_pointer[(mo_pridx,
                                                                da_pridx)]
                if da_pridx == mo_pridx and prnco[da_pridx] == 101:
                    A = 0.8
                    B = 1

                self.reinjection_struc[da_pridx].append(
                    coo_matrix(
                        ((A / B) / np.ones_like(self.e_cosmicray.widths), (
                            idcs, idcs)),
                        shape=shape_submat))

        self.reinjection_smat = bmat(self.reinjection_struc)


class ContinuousLossRates(InteractionRateBase):
    """Implementation of continuous pair production loss rates."""

    def __init__(self, prince_run, *args, **kwargs):
        InteractionRateBase.__init__(self, prince_run, *args, **kwargs)

        # Number of integration steps in phi
        self.xi_steps = 100 if 'xi_steps' not in kwargs else kwargs['xi_steps']

    def _setup(self):
        self._init_constants()

    def _init_constants(self):
        xi = np.logspace(0.30103, 8., self.xi_steps)

        int_prefactor = pru.fine_structure * pru.r_electron**2 * pru.m_electron**2

        self.pref_phi_xi2 = int_prefactor * self._phi(xi) / xi**2

        gamma = self.e_cosmicray.grid / pru.m_proton

    def _phi(self, xi):
        """Phi function as in Blumental 1970"""

        # Simple ultrarelativistic approximation by Blumental 1970
        bltal_ultrarel = np.poly1d([2.667, -14.45, 50.95, -86.07])
        phi_simple = lambda xi: xi * bltal_ultrarel(np.log(xi))

        # random fit parameters, see Chorodowski et al
        c1 = 0.8048
        c2 = 0.1459
        c3 = 1.137e-3
        c4 = -3.879e-6

        f1 = 2.91
        f2 = 78.35
        f3 = 1837
        xi = np.asarray(xi)
        res = np.zeros(xi.shape)

        le = xi < 25.
        he = xi >= 25.

        res[le] = np.pi / 12. * (xi[le] - 2)**4 / (c1 * (xi[le] - 2)**1 + c2 *
                                                   (xi[le] - 2)**2 + c3 *
                                                   (xi[le] - 2)**3 + c4 *
                                                   (xi[le] - 2)**4)

        res[he] = phi_simple(xi[he]) / (
            1 - f1 * xi[he]**-1 - f2 * xi[he]**-2 - f3 * xi[he]**-3)

        return res

    def lossrate(self, energy, z=0, intsteps=100):
        """Loss rate for a proton with lorentz factor gamma 
        propagating through photon_density at redshift z.
        
        Fully vectorized version!
        
        For nuclei multiply the result by Z^2
        result is given in [GeV / cm]
        """
        gamma = energy / 939e-3

        ephoton = np.outer(1 / gamma, xi) * emass / 2
        pdensity = photon_density.get_photon_density

        intmatrix = (pdensity(ephoton.reshape(-1),
                              z).reshape(ephoton.shape)) * phi(xi) / xi**2
        integral = integrate.trapz(intmatrix, xi, axis=1)
        pref = (finestruct * eradius**2 * emass**2)
        return pref * integral

    pair_lossrate = lossrate
