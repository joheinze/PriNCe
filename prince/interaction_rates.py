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

    def __init__(self, prince_run=None, with_dense_jac=True, *args, **kwargs):

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

        self.with_dense_jac = with_dense_jac

        InteractionRateBase.__init__(self, prince_run, *args, **kwargs)

    def _setup(self):
        """Initialization of all members."""

        self._batch_rows = []
        self._batch_cols = []
        self._batch_matrix = None
        self._batch_vec = None
        self.coupling_mat = None
        self.dense_coupling_mat = None

        self._init_matrices()
        self._init_coupling_mat(sp_format='csr')

    def _init_matrices(self):
        """ A new take on filling the matrices"""

        # Define some short-cuts
        known_species = self.spec_man.known_species[::-1]
        sp_id_ref = self.spec_man.ncoid2sref
        resp = self.cross_sections.resp
        m_pr = PRINCE_UNITS.m_proton

        # Energy variables
        dcr = self.dim_cr
        dph = self.dim_ph
        ecr = self.e_cosmicray.grid
        bcr = self.e_cosmicray.bins
        eph = self.e_photon.grid
        bph = self.e_photon.bins

        # Edges of each CR energy bin
        elims = np.vstack([bcr[:-1], bcr[1:]])

        # Edges of each photon energy bin
        plims = np.vstack([bph[:-1], bph[1:]])

        # Widths
        delta_ec = np.subtract(*-elims[:, :])
        delta_ph = np.subtract(*-plims[:, :])

        # CR and photon grid indices
        emo_idcs = eda_idcs = np.arange(dcr)
        p_idcs = np.arange(dph)

        # calculate dimension of the batch matrix
        batch_dim = 0
        for specid in known_species:
            if specid < 100:
                continue
            # Add the main diagonal self-couplings (absoption)
            batch_dim += dcr
            for rtup in self.cross_sections.reactions[specid]:
                # Off main diagonal couplings (reinjection)
                if rtup in self.cross_sections.known_bc_channels:
                    batch_dim += dcr
                elif rtup in self.cross_sections.known_diff_channels:
                    # Only half of the elements can be non-zero (energy conservation)
                    batch_dim += int(dcr**2 / 2) + 1
                    
        info(2, 'Batch matrix dimensions are {0}x{1}'.format(batch_dim, dph))
        info(3, 'Memory usage: {0} MB'.format(batch_dim * dph * 8 / 1024**2))

        self._batch_matrix = np.zeros((batch_dim, dph))
        ibatch = 0
        self._batch_rows = []
        self._batch_cols = []

        # We are gonna fill the batch matrix with the cross section using iterators
        # The op_axes arguments define how to compute the outer product.

        # Iterate over all combinations of species
        spec_iter = np.nditer(
            [known_species, known_species], op_axes=[[0, -1], [-1, 0]])

        # Iterate over mother and daughter indices synchronously
        it_bc = np.nditer(
            [emo_idcs, eda_idcs, p_idcs],
            op_axes=[[0, -1], [0, -1], [-1, 0]],
            flags=['external_loop'])

        # Iterate over outer product of m_energy x d_energy x ph_energy
        it_diff = np.nditer(
            [emo_idcs, eda_idcs, p_idcs],
            op_axes=[[0, -1, -1], [-1, 0, -1], [-1, -1, 0]])

        x_cut = config['x_cut']

        for moid, daid in spec_iter:
            # Cast from np.array to Python int
            moid, daid = int(moid), int(daid)

            # Mass number of mother and daughter
            # (float) cast needed for exact ratio
            mass_mo = float(get_AZN(moid)[0])
            mass_da = float(get_AZN(daid)[0])

            if mass_mo < mass_da or moid < 100:
                continue
            # TODO: workaround for missing injection into redist
            if mass_mo > 1 and daid < 100:
                print 'skip', moid, daid
                continue

            if moid == daid:
                intp_nonel = resp.nonel_intp[moid].antiderivative()
                has_nonel = True
            else:
                has_nonel = False

            if (((moid, daid) in self.cross_sections.known_bc_channels) or
                (has_nonel and
                 (moid, daid) not in self.cross_sections.known_diff_channels)):

                it_bc.reset()
                if (moid, daid) in resp.incl_intp:
                    intp_bc = resp.incl_intp[(moid, daid)].antiderivative()
                    has_incl = True
                else:
                    has_incl = False
                    info(1, 'Inclusive interpolator not found for', (moid,
                                                                     daid))
                if not (has_nonel or has_incl):
                    raise Exception('Channel without interactions:', (moid,
                                                                      daid))

                for m_eidx, d_eidx, ph_idx in it_bc:
                    m_eidx = m_eidx[0]
                    d_eidx = d_eidx[0]
                    emo = ecr[m_eidx]

                    # Differential factors
                    int_fac = (delta_ec[m_eidx] * delta_ph[ph_idx] / emo *
                               mass_mo / mass_da)
                    diff_fac = 1. / delta_ec[d_eidx] / delta_ph[ph_idx] * m_pr

                    yl = plims[0, ph_idx] * emo / m_pr
                    yu = plims[1, ph_idx] * emo / m_pr

                    if has_incl:
                        self._batch_matrix[ibatch, :] = (
                            intp_bc(yu) - intp_bc(yl)) * int_fac * diff_fac
                    if has_nonel:
                        self._batch_matrix[ibatch, :] -= (
                            intp_nonel(yu) - intp_nonel(yl)
                        ) * int_fac * diff_fac

                    # Try later to check for zero result to save more zeros.
                    ibatch += 1
                    self._batch_rows.append(sp_id_ref[daid].lidx() + d_eidx)
                    self._batch_cols.append(sp_id_ref[moid].lidx() + m_eidx)

            elif (moid, daid) in self.cross_sections.known_diff_channels:
                it_diff.reset()
                if (moid, daid) in resp.incl_diff_intp:
                    intp_diff = resp.incl_diff_intp[(moid, daid)]
                    ymin = np.min(intp_diff.get_knots()[1])
                    has_redist = True
                else:
                    has_redist = False
                    raise Exception('This should not occur.')

                for m_eidx, d_eidx, ph_idx in it_diff:

                    if d_eidx > m_eidx:
                        continue

                    emo = ecr[m_eidx]

                    xl = elims[0, d_eidx] / emo
                    xu = elims[1, d_eidx] / emo
                    yl = plims[0, ph_idx] * emo / m_pr
                    yu = plims[1, ph_idx] * emo / m_pr

                    #TODO: Thresholds set for testing
                    if daid == 101 and xl < 0.5:
                        continue
                    if xl < 1e-6 or yu < ymin:
                        continue

                    # Differential factors
                    int_fac = (delta_ec[m_eidx] * delta_ph[ph_idx] / emo *
                               mass_mo / mass_da)
                    diff_fac = 1. / delta_ec[d_eidx] / delta_ph[ph_idx] * m_pr

                    self._batch_matrix[ibatch, ph_idx] = intp_diff.integral(
                        xl, xu, yl, yu) * diff_fac * int_fac

                    if has_nonel and m_eidx == d_eidx:
                        self._batch_matrix[ibatch, ph_idx] -= (
                            intp_nonel(yu) - intp_nonel(yl)
                        ) * diff_fac * int_fac

                    if ph_idx == p_idcs[-1]:
                        ibatch += 1
                        self._batch_rows.append(sp_id_ref[daid].lidx() +
                                                d_eidx)
                        self._batch_cols.append(sp_id_ref[moid].lidx() +
                                                m_eidx)
            else:
                info(2, 'Species combination not supported', moid, daid)

        self._batch_matrix.resize((ibatch, dph))
        self._batch_rows = np.array(self._batch_rows)
        self._batch_cols = np.array(self._batch_cols)
        self._batch_vec = np.zeros(ibatch)

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

        # Reorder batch matrix according to order in coupling_mat
        self._batch_matrix = self._batch_matrix[self.sortidx, :]
        self._batch_rows = self._batch_rows[self.sortidx]
        self._batch_cols = self._batch_cols[self.sortidx]

        if self.with_dense_jac:
            self.dense_coupling_mat = np.zeros((self.coupling_mat.shape))

    def _update_coupling_mat(self, z, scale_fac, force_update=False):
        """Updates the sparse (csr) coupling matrix
        Only the data vector is updated to minimize computation
        """
        from scipy.sparse import csc_matrix

        # Do not execute dot product if photon field didn't change
        if self._update_rates(z, force_update):
            self.coupling_mat.data = scale_fac * self._batch_vec

    def get_hadr_jacobian(self, z, scale_fac=1., force_update=False):
        """Returns the nonel rate vector and coupling matrix.
        """
        self._update_coupling_mat(z, scale_fac, force_update)
        return self.coupling_mat

    def get_dense_hadr_jacobian(self, z, scale_fac=1., force_update=False):
        """Returns the nonel rate vector and coupling matrix.
        """
        if not self.with_dense_jac:
            raise Exception('Dense jacobian not activated.')

        # return self.get_hadr_jacobian(z, scale_fac, force_update).todense()
        self._update_coupling_mat(z, scale_fac, force_update)

        self.dense_coupling_mat[self._batch_rows,
                                self._batch_cols] = scale_fac * self._batch_vec
        return self.dense_coupling_mat

    def _update_rates(self, z, force_update=False):
        """Batch compute all nonel and inclusive rates if z changes.

        The result is always stored in the same vectors, since '_init_rate_matstruc'
        makes use of views to link ranges of the vector to locations in the matrix.

        Args:
            z (float): Redshift value at which the photon field is taken.

        Returns:
            (bool): True if fields we indeed updated, False if nothing happened.
        """

        if self._ratemat_zcache != z or force_update:
            info(5, 'Updating batch rate vectors.')
            np.dot(
                self._batch_matrix, self.photon_vector(z), out=self._batch_vec)
            self._ratemat_zcache = z
            return True
        else:
            return False

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
        units = (PRINCE_UNITS.fine_structure * PRINCE_UNITS.r_electron**2 *
                 PRINCE_UNITS.m_electron**2)

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
