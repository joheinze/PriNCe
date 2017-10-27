"""The module contains classes for computations of interaction rates"""

from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.sparse import bmat, coo_matrix

from prince.util import (get_AZN, get_interp_object, info,
                         load_or_convert_array, PRINCE_UNITS)
from prince_config import config

#import matplotlib.pyplot as plt

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

    @abstractmethod
    def _update_rates(self, z):
        """Updates all rates if photon field/redshift changes"""
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
    """Implementation of photo-hadronic/nuclear interaction rates.
    This Version directly writes the data into a CSC-matrix and only updates the data each time.
    """

    def __init__(self, prince_run=None, *args, **kwargs):
        if prince_run is None:
            # For debugging and independent calculations define
            # a strawman class and supply the required paramters as
            # attributes

            class PrinceRunMock(object):
                pass

            from prince.data import SpeciesManager
            from prince.util import EnergyGrid

            prince_run = PrinceRunMock()
            if "photon_bins_per_dec" not in kwargs:
                kwargs["photon_bins_per_dec"] = config["photon_grid"][2]
            if "cr_bins_per_dec" not in kwargs:
                kwargs["cr_bins_per_dec"] = config["cosmic_ray_grid"][2]

            prince_run.e_photon = EnergyGrid(config["photon_grid"][0],
                                             config["photon_grid"][1],
                                             kwargs["photon_bins_per_dec"])
            prince_run.e_cosmicray = EnergyGrid(config["cosmic_ray_grid"][0],
                                                config["cosmic_ray_grid"][1],
                                                kwargs["cr_bins_per_dec"])
            prince_run.cross_sections = kwargs['cs']
            prince_run.photon_field = kwargs['pf']

            prince_run.spec_man = SpeciesManager(
                prince_run.cross_sections.known_species,
                prince_run.e_cosmicray.d)

        InteractionRateBase.__init__(self, prince_run, *args, **kwargs)

    def _setup(self):
        """Initialization of all members."""
        self._init_matrices()

        #self._fill_matrix_nonel()
        #self._fill_matrix_incl()
        #self._fill_matrix_incl_diff()

        self._fill_batch_matrix()

        self._init_coupling_mat(sp_format='csr')

    def _init_matrices(self):
        """Initializes convenicen matrices for batch computation"""
        from prince.util import get_y

        # Iniialize cross section matrices, evaluated on a grid of y values
        eph_mat, ecr_mat = np.meshgrid(self.e_photon.grid,
                                       self.e_cosmicray.grid)
        # Compute y matrix only once and then rescale by A
        self.ymat = get_y(ecr_mat, eph_mat, 100)

        ecr_mat_in, ecr_mat_out = np.meshgrid(self.e_cosmicray.grid,
                                              self.e_cosmicray.grid)
        decr_mat_in, decr_mat_out = np.meshgrid(self.e_cosmicray.widths,
                                              self.e_cosmicray.widths)
        self.xmat = ecr_mat_out / ecr_mat_in
        # matrix for numeric correction of differential redist:
        # factor D_Ein for integral, factor 1 / E_in for x substitution
        # and factor D_Ein / D_Eout for bin width correction
        
        #print 'integral correction:'
        self.integ_corr_mat = decr_mat_in ** 2 / ecr_mat_in / decr_mat_out
        #print self.integ_corr_mat
        self.integ_corr_mat = decr_mat_in / ecr_mat_in
        #self.integ_corr_mat = np.ones(self.integ_corr_mat.shape)
        #print self.integ_corr_mat

        n_nonel_diff = len([
            species for species in self.spec_man.species_refs
            if species.has_redist
        ])
        n_nonel = self.spec_man.nspec - n_nonel_diff
        n_incl = len([(mo, da)
                      for (mo, da) in self.cross_sections.known_bc_channels
                      if mo != da])
        n_incl_diff = len([(mo, da)
                           for (mo,
                                da) in self.cross_sections.known_diff_channels
                           if mo != da])

        # [mother_ncoid,daughter_ncoid] -> _batch_vec[lidx:uidx]
        self._nonel_batchvec_pointer = {}
        self._incl_batchvec_pointer = {}
        self._incl_diff_batchvec_pointer = {}
        # [mother_princeidx,daughter_princeidx] -> _batch_vec[lidx:uidx]
        self._incl_batchvec_pridx_pointer = {}
        self._incl_diff_batchvec_pridx_pointer = {}

        dim_nonel = self.dim_cr * n_nonel + self.dim_cr**2 * n_nonel_diff
        dim_incl = self.dim_cr * n_incl
        dim_incl_diff = self.dim_cr**2 * n_incl_diff
        dim_ph = self.dim_ph
        info(3, 'Batch matrix nonel dimension: {0}'.format(dim_nonel))
        info(3, 'Batch matrix incl dimension: {0}'.format(dim_incl))
        info(3, 'Batch matrix incl diff dimension: {0}'.format(dim_incl_diff))
        info(3, 'Batch matrix photon dimension: {0}'.format(dim_ph))

        # Convolution matrix for response function
        self._batch_matrix = np.zeros((dim_nonel + dim_incl + dim_incl_diff,
                                       dim_ph))
        info(2, 'Size of complete batch matrix: {0}x{1}'.format(
            *self._batch_matrix.shape))

        # Result vector, which stores computed rates
        self._batch_vec = np.zeros(self._batch_matrix.shape[0])

        # Vector which stores constant prefactors for batch vector
        self._batch_vec_prefac = np.ones(self._batch_matrix.shape[0])

        self._batch_rows = np.zeros(self._batch_matrix.shape[0], dtype=np.int)
        self._batch_cols = np.zeros(self._batch_matrix.shape[0], dtype=np.int)

        # pointer to lower-upper index
        self._batch_vec_pointer = {}

    def _fill_batch_matrix(self):
        """ Fill the batch matrix with physics """
        info(2, 'Starting to fill batch matrix')
        # Delta eps (photon energy) bin widths
        delta_eps = np.diag(self.e_photon.widths)

        species = self.spec_man.ncoid2sref
        resp = self.cross_sections.resp

        # We need x and y on the 3D array E_da x E_mo x eph
        # therefore repeat xmat and ymat accordingly
        x_repeat = np.repeat(
            self.xmat[:, :, np.newaxis], self.ymat.shape[1], axis=2)
        y_repeat = np.repeat(
            self.ymat[:, np.newaxis, :], self.xmat.shape[1], axis=1)
        #print 'integration correction matrix'
        #print 'shape:', self.integ_corr_mat.shape
        #print self.integ_corr_mat
        #plt.spy(self.integ_corr_mat,precision=0.01)
        integ_corr_repeat = np.repeat(
            self.integ_corr_mat[:, :, np.newaxis], self.ymat.shape[1], axis=2)        
        #print 'in repeated form:'
        #print 'shape:', integ_corr_repeat.shape
        #print integ_corr_repeat

        # reshape to 2D grid, to fit the batch matrix
        x_repeat = x_repeat.reshape((-1, x_repeat.shape[2]))
        y_repeat = y_repeat.reshape((-1, y_repeat.shape[2]))
        integ_corr_repeat = integ_corr_repeat.reshape((-1, integ_corr_repeat.shape[2]))

        #print 'repeated correction matrix:'
        #print 'shape:', integ_corr_repeat.shape
        #print integ_corr_repeat
        #plt.spy(integ_corr_repeat,precision=0.01)
        fill_idx = 0

        for mother in self.spec_man.known_species:
            if species[mother].has_redist:
                lidx = fill_idx
                uidx = fill_idx + self.dim_cr**2

                prindices_mo = species[mother].indices()
                prindices_in = np.tile(prindices_mo, self.xmat.shape[0])
                prindices_out = np.repeat(prindices_mo, self.xmat.shape[1])
                #prindices_in = np.repeat(prindices_mo, self.xmat.shape[1])
                #prindices_out = np.tile(prindices_mo, self.xmat.shape[0])
                self._batch_rows[lidx:uidx] = prindices_out
                self._batch_cols[lidx:uidx] = prindices_in

                if mother < 100:
                    # Note: in this case the batch vector will be zero anyway
                    info(3, "Can not compute interaction rate for", mother)
                    fill_idx += self.dim_cr**2
                    continue

                self._batch_matrix[lidx:uidx] = resp.get_full(
                    mother, mother, y_repeat, xgrid=x_repeat).dot(delta_eps)
                self._batch_vec_pointer[mother] = (lidx, uidx)

                fill_idx += self.dim_cr**2
            else:
                lidx = fill_idx
                uidx = fill_idx + self.dim_cr

                prindices = species[mother].indices()
                self._batch_rows[lidx:uidx] = prindices
                self._batch_cols[lidx:uidx] = prindices

                if mother < 100:
                    # Note: in this case the batch vector will be zero anyway
                    info(3, "Can not compute interaction rate for", mother)
                    fill_idx += self.dim_cr
                    continue

                self._batch_matrix[lidx:uidx] = resp.get_full(
                    mother, mother, self.ymat).dot(delta_eps)
                self._batch_vec_pointer[mother] = (lidx, uidx)

                fill_idx += self.dim_cr

            for (mo, da) in self.cross_sections.reactions[mother]:

                if mo == da:
                    # these were already covered before the loop
                    continue

                if (mo, da) in self.cross_sections.known_bc_channels:
                    # Indices in batch matrix
                    lidx = fill_idx
                    uidx = fill_idx + self.dim_cr

                    # Staple ("vstack"") all inclusive (channel) response functions
                    self._batch_matrix[lidx:uidx] = resp.get_full(
                        mo, da, self.ymat).dot(delta_eps)

                    B = float(get_AZN(da)[0])
                    A = float(get_AZN(mo)[0])

                    self._batch_vec_prefac[lidx:uidx].fill(A / B)

                    prindices_mo = species[mo].indices()
                    prindices_da = species[da].indices()

                    self._batch_rows[lidx:uidx] = prindices_da
                    self._batch_cols[lidx:uidx] = prindices_mo
                    self._batch_vec_pointer[(mo, da)] = (lidx, uidx)

                    fill_idx += self.dim_cr

                elif (mo, da) in self.cross_sections.known_diff_channels:
                    # Indices in batch matrix
                    lidx = fill_idx
                    uidx = fill_idx + self.dim_cr**2

                    self._batch_matrix[lidx:uidx] = resp.get_full(
                        mo, da, y_repeat, xgrid=x_repeat).dot(delta_eps)
                    if mo == 101:
                        print 'for daugter {:}:'.format(da)
                        print self._batch_matrix[lidx:uidx]
                    self._batch_matrix[lidx:uidx] = integ_corr_repeat * self._batch_matrix[lidx:uidx]
                    if mo == 101:
                        print 'after multiplying correction'
                        print self._batch_matrix[lidx:uidx]
                        print 'correction matrix:'
                        print integ_corr_repeat
                        print '-'*25
                    B = float(get_AZN(da)[0])
                    A = float(get_AZN(mo)[0])

                    self._batch_vec_prefac[lidx:uidx].fill(A / B)

                    prindices_mo = species[mo].indices()
                    prindices_da = species[da].indices()

                    prindices_in = np.tile(prindices_mo, self.xmat.shape[0])
                    prindices_out = np.repeat(prindices_da, self.xmat.shape[1])
                    #prindices_in = np.repeat(prindices_mo, self.xmat.shape[1])
                    #prindices_out = np.tile(prindices_da, self.xmat.shape[0])
                    self._batch_rows[lidx:uidx] = prindices_out
                    self._batch_cols[lidx:uidx] = prindices_in
                    self._batch_vec_pointer[(mo, da)] = (lidx, uidx)

                    fill_idx += self.dim_cr**2
        info(2, 'Finished filling of batch matrix!')

    def _init_coupling_mat(self, sp_format):
        """Initialises the coupling matrix directly in sparse (csc) format.
        """
        info(2, 'Initiating coupling matrix in ({:}) format'.format(sp_format))

        if sp_format == 'csc':
            from scipy.sparse import csc_matrix
            self.coupling_mat = csc_matrix(
                (self._batch_vec, (self._batch_rows, self._batch_cols)),
                copy=True)
            # create an index to sort by columns and then rows,
            # which is the same ordering CSC has internally
            # lexsort sorts by last argument first!!!
            self.sortidx = np.lexsort((self._batch_rows, self._batch_cols))
        elif sp_format == 'csr':
            from scipy.sparse import csr_matrix
            self.coupling_mat = csr_matrix(
                (self._batch_vec, (self._batch_rows, self._batch_cols)),
                copy=True)
            # create an index to sort by rows and then columns,
            # which is the same ordering CSR has internally
            # lexsort sorts by last argument first!!!
            self.sortidx = np.lexsort((self._batch_cols, self._batch_rows))
        else:
            raise Exception(
                'Unsupported sparse format ({:}) for coupling matrix, choose (csc) or (csr)'.
                format(sp_format))

        # TODO: For now the reordering is done in each step in _update_coupling_mat()
        #   Doing the reordering and multiplying the prefactor vector here
        #   can speed up the each step by ~0.7 ms (vs ~4.5 ms, so about 20%)
        # the reordering as commented out below does however not seem to work properly
        # maybe reordering on the 2D array does not work as expected

        #self._batch_matrix = self._batch_matrix[sortidx] # JH: this might not work
        #self._batch_vec = self._batch_vec[sortidx]
        #self._batch_vec = self._batch_vec_prefac[sortidx]
        #self._batch_rows = self._batch_rows[sortidx]
        #self._batch_cols = self._batch_cols[sortidx]

    def _update_coupling_mat(self, z):
        """Updates the sparse (csr) coupling matrix
        Only the data vector is updated to minimize computation
        """
        from scipy.sparse import csc_matrix
        self._update_rates(z)

        # TODO: The reording here does currently take 0.3 ms (vs 4.5 ms for the complete call)
        # If this will get time critical with further optimization,
        # one can do the reordring in _init_coupling_mat(self) once for the batch matrix
        self.coupling_mat.data = (
            self._batch_vec * self._batch_vec_prefac)[self.sortidx]

    def get_hadr_jacobian(self, z):
        """Returns the nonel rate vector and coupling matrix.
        """
        self._update_coupling_mat(z)
        return self.coupling_mat

    def _update_rates(self, z):
        """Batch compute all nonel and inclusive rates if z changes.

        The result is always stored in the same vectors, since '_init_rate_matstruc'
        makes use of views to link ranges of the vector to locations in the matrix.
        """
        if self._ratemat_zcache != z:
            info(5, 'Updating batch rate vectors.')
            np.dot(
                self._batch_matrix, self.photon_vector(z), out=self._batch_vec)
            self._ratemat_zcache = z

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
        response = self.cross_sections.resp
        # Convolve using matrix multiplication
        if isinstance(nco_ids, tuple):
            # assume nco_id is tuple of (mo, da)
            mo, da = nco_ids
            if (mo, da) in self.cross_sections.known_bc_channels:
                res = response.get_full(mo, da, self.ymat)
            elif (mo, da) in self.cross_sections.known_diff_channels:
                res = response.get_full(mo, da, self.ymat, xgrid=self.xmat)
            else:
                raise Exception(
                    'Unknown inclusive channel {:} -> {:}'.format(mo, da))
        else:
            # assume nco_ids is just one integer, mo = da
            mo = da = nco_ids
            res = response.nonel_intp[mo](self.ymat)
        return res.dot(np.diag(self.e_photon.widths)).dot(
            self.photon_vector(z))

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

        self._update_rates(z)
        lidx, uidx = self._batch_vec_pointer[nco_ids]
        return self._batch_vec[lidx:uidx]

class ContinuousLossRates(InteractionRateBase):
    """Implementation of continuous pair production loss rates."""

    def __init__(self, prince_run, *args, **kwargs):
        # Number of integration steps in phi
        self.xi_steps = 100 if 'xi_steps' not in kwargs else kwargs['xi_steps']

        InteractionRateBase.__init__(self, prince_run, *args, **kwargs)

    def pprod_loss_rate(self, nco_id, z):
        """Returns energy loss rate in GeV/cm."""

        self._update_rates(z)
        s = self.spec_man.ncoid2sref[nco_id]

        return self.pprod_loss_vector[s.lidx():s.uidx()]

    def adiabatic_losses(self, z):
        """Returns adiabatic loss vector at redshift z"""
        from prince.cosmology import H
        return H(z) * PRINCE_UNITS.cm2sec * self.adiabatic_loss_vector

    def pprod_losses(self, z):
        """Returns pair production losses at redshift z"""
        self._update_rates(z)
        return self.pprod_loss_vector

    def loss_vector(self, z):
        """Returns all continuous losses on dim_states grid"""

        return self.pprod_losses(z) + self.adiabatic_losses(z)

    def _setup(self):

        # Photon grid for parallel integration of the "Blumental integral"
        # for each point of the cosmic ray grid

        # xi is dimensionless (natural units) variable
        self.xi = np.logspace(np.log10(2 + 1e-8), 8., self.xi_steps)

        # weights for integration
        self.phi_xi2 = self._phi() / (self.xi**2)

        # Gamma factor of the cosmic ray
        gamma = self.e_cosmicray.grid / PRINCE_UNITS.m_proton

        # Scale vector containing the units and factors of Z**2 for nuclei
        self.scale_vec = self._init_scale_vec()

        # Init adiabatic loss vector
        self.adiabatic_loss_vector = self._init_adiabatic_vec()

        # Grid of photon energies for interpolation
        self.photon_grid = np.outer(1 / gamma,
                                    self.xi) * PRINCE_UNITS.m_electron / 2.
        self.pg_desort = self.photon_grid.reshape(-1).argsort()
        self.pg_sorted = self.photon_grid.reshape(-1)[self.pg_desort]

        # Rate vector (dim_states x 1) containing all rates
        self.pprod_loss_vector = np.zeros(self.prince_run.dim_states)

    def _update_rates(self, z):
        """Updates photon fields and computes rates for all species."""
        from scipy.integrate import trapz

        np.multiply(
            self.scale_vec,
            np.tile(
                trapz(self.photon_vector(z) * self.phi_xi2, self.xi, axis=1),
                self.spec_man.nspec),
            out=self.pprod_loss_vector)

    def _set_photon_vector(self, z):
        """Get and cache photon vector for the previous value of z.

        This vector is in fact a matrix of vectors of the interpolated
        photon field with dimensions (dim_cr, xi_steps).

        Args:
            z (float): redshift
        """
        if self._photon_vector is None:
            self._photon_vector = np.zeros_like(self.photon_grid)

        if self._ph_vec_zcache != z:
            self._photon_vector.reshape(-1)[
                self.pg_desort] = self.photon_field.get_photon_density(
                    self.pg_sorted, z)
            self._ph_vec_zcache = z

    def _init_scale_vec(self):
        """Prepare vector for scaling with units, charge and mass."""

        scale_vec = np.zeros(self.prince_run.dim_states)
        units = PRINCE_UNITS.fine_structure * PRINCE_UNITS.r_electron**2 * PRINCE_UNITS.m_electron**2

        for spec in self.spec_man.species_refs:
            if not spec.is_nucleus:
                continue
            scale_vec[spec.lidx():spec.uidx()] = units * abs(
                spec.charge) * spec.Z**2 / float(spec.A) * np.ones(
                    self.dim_cr, dtype='double')

        return scale_vec

    def _init_adiabatic_vec(self):
        """Prepare vector for scaling with units, charge and mass."""

        adiabatic_loss_vector = np.zeros(self.prince_run.dim_states)

        for spec in self.spec_man.species_refs:
            adiabatic_loss_vector[spec.lidx():
                                  spec.uidx()] = self.e_cosmicray.grid

        return adiabatic_loss_vector

    def _phi(self):
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

        xi = self.xi
        res = np.zeros(xi.shape)

        le = np.where(xi < 25.)
        he = np.where(xi >= 25.)

        res[le] = np.pi / 12. * (xi[le] - 2)**4 / (c1 * (xi[le] - 2)**1 + c2 *
                                                   (xi[le] - 2)**2 + c3 *
                                                   (xi[le] - 2)**3 + c4 *
                                                   (xi[le] - 2)**4)

        res[he] = phi_simple(xi[he]) / (
            1 - f1 * xi[he]**-1 - f2 * xi[he]**-2 - f3 * xi[he]**-3)

        return res
