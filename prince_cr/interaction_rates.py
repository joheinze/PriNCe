"""The module contains classes for computations of interaction rates"""

import numpy as np
from scipy.integrate import trapz

from prince_cr.data import PRINCE_UNITS
from prince_cr.util import info
import prince_cr.config as config

using_cupy = False
# Use GPU support
if config.has_cupy and config.linear_algebra_backend.lower() == 'cupy':
    import cupy
    from prince_cr.config import mempool
    using_cupy = True

class PhotoNuclearInteractionRate(object):
    """Implementation of photo-hadronic/nuclear interaction rates.
    This Version directly writes the data into a CSC-matrix and only updates the data each time.
    """

    def __init__(self, prince_run=None, with_dense_jac=True, *args, **kwargs):
        info(3, 'creating instance')
        self.with_dense_jac = with_dense_jac

        #: Reference to PhotonField object
        self.photon_field = prince_run.photon_field

        #: Reference to CrossSection object
        self.cross_sections = prince_run.cross_sections

        #: Reference to species manager
        self.spec_man = prince_run.spec_man

        # Initialize grids
        self.e_photon = prince_run.ph_grid
        self.e_cosmicray = prince_run.cr_grid

        # Initialize cache of redshift value
        self._ph_vec_zcache = None
        self._ratemat_zcache = None

        # Initialize the matrices for batch computation
        self._batch_rows = None
        self._batch_cols = None
        self._batch_matrix = None
        self._batch_vec = None
        self.coupling_mat = None
        self.dense_coupling_mat = None

        self._estimate_batch_matrix()
        self._init_matrices()
        self._init_coupling_mat()

    def photon_vector(self, z):
        """Returns photon vector at redshift `z` on photon grid.

        This vector is in fact a matrix of vectors of the interpolated
        photon field with dimensions (dim_cr, xi_steps).

        Args:
            z (float): redshift

        Return value from cache if redshift value didn't change since last call.
        """
        # photon_vector = np.zeros_like(self.e_photon.grid)
        photon_vector = self.photon_field.get_photon_density(
            self.e_photon.grid, z)

        return photon_vector

    def _estimate_batch_matrix(self):
        ''' estimate dimension of the batch matrix'''
        dcr = self.e_cosmicray.d
        dph = self.e_photon.d

        batch_dim = 0
        for specid in self.spec_man.known_species:
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
        self._batch_matrix = np.zeros((batch_dim, dph))
        self._batch_rows = []
        self._batch_cols = []
        info(
            3,
            'Memory usage: {0} MB'.format(self._batch_matrix.nbytes / 1024**2))

    def _init_matrices(self):
        """ A new take on filling the matrices"""

        # Define some short-cuts
        known_species = self.spec_man.known_species[::-1]
        sp_id_ref = self.spec_man.ncoid2sref
        resp = self.cross_sections.resp
        m_pr = PRINCE_UNITS.m_proton

        # Energy variables
        dcr = self.e_cosmicray.d
        dph = self.e_photon.d
        ecr = self.e_cosmicray.grid
        bcr = self.e_cosmicray.bins
        eph = self.e_photon.grid
        bph = self.e_photon.bins
        delta_ec = self.e_cosmicray.widths
        delta_ph = self.e_photon.widths

        # Edges of each CR energy bin and photon energy bin
        elims = np.vstack([bcr[:-1], bcr[1:]])
        plims = np.vstack([bph[:-1], bph[1:]])

        # CR and photon grid indices
        emo_idcs = np.arange(dcr)
        eda_idcs = np.arange(dcr)
        p_idcs = np.arange(dph)

        # values for x and y to cut on:
        x_cut = config.x_cut
        y_cut = config.y_cut
        x_cut_proton = config.x_cut_proton

        ibatch = 0
        import itertools
        spec_iter = itertools.product(known_species, known_species)
        for moid, daid in spec_iter:

            if moid < 100:
                continue
            else:
                info(10, f'Filling channel {moid} -> {daid}')

            has_nonel = moid == daid
            if has_nonel:
                intp_nonel = resp.nonel_intp[moid].antiderivative()

            if (((moid, daid) in self.cross_sections.known_bc_channels) or
                (has_nonel and
                 (moid, daid) not in self.cross_sections.known_diff_channels)):

                has_incl = (moid, daid) in resp.incl_intp
                if has_incl:
                    intp_bc = resp.incl_intp[(moid, daid)].antiderivative()
                else:
                    info(1, 'Inclusive interpolator not found for',
                         (moid, daid))

                if not (has_nonel or has_incl):
                    raise Exception('Channel without interactions:',
                                    (moid, daid))

                # The cross sections need to be evaluated
                # on x = E_{CR,da} / E_{CR,mo} and y = E_ph * E_{CR,mo} / m_proton
                # To vectorize the evaluation, we create outer products using numpy
                # broadcasting:

                emo = ecr
                xl = elims[0] / emo
                xu = elims[1] / emo
                delta_x = delta_ec / emo

                yl = plims[0, None, :] * emo[:, None] / m_pr
                yu = plims[1, None, :] * emo[:, None] / m_pr
                delta_y = delta_ph[None, :] * emo[:, None] / m_pr

                int_fac = (delta_ec[:, None] * delta_ph[None, :] / emo[:, None])
                diff_fac = 1. / delta_x[:, None] / delta_y

                # This takes the average by evaluating the integral and dividing by bin
                # width
                if has_incl:
                    self._batch_matrix[ibatch:ibatch + len(emo), :] = (
                        intp_bc(yu) - intp_bc(yl)) * int_fac * diff_fac
                if has_nonel:
                    self._batch_matrix[ibatch:ibatch + len(emo), :] -= (intp_nonel(
                        yu) - intp_nonel(yl)) * int_fac * diff_fac

                # finally map this to the coupling matrix
                ibatch += len(emo)
                self._batch_rows.append(sp_id_ref[daid].lidx() + eda_idcs)
                self._batch_cols.append(sp_id_ref[moid].lidx() + emo_idcs)

            elif (moid, daid) in self.cross_sections.known_diff_channels:

                has_redist = (moid, daid) in resp.incl_diff_intp
                if has_redist:
                    intp_diff = resp.incl_diff_intp[(moid, daid)]
                    intp_diff_integral = resp.incl_diff_intp_integral[(moid, daid)]
                    intp_nonel = resp.nonel_intp[moid]
                    intp_nonel_antid = resp.nonel_intp[moid].antiderivative()

                    ymin = np.min(intp_diff.get_knots()[1])
                else:
                    raise Exception('This should not occur.')

                ibatch_bf = ibatch
                # generate outer products using broadcasting
                emo = ecr[:, None, None]
                eda = ecr[None, :, None]
                epho = eph[None, None, :]
                target_shape = np.ones_like(emo * eda * epho)

                xl = elims[0, None, :, None] / emo * target_shape
                xu = elims[1, None, :, None] / emo * target_shape
                delta_x = delta_ec[None, :, None] / emo

                yl = plims[0, None, None, :] * emo / m_pr * target_shape
                yu = plims[1, None, None, :] * emo / m_pr * target_shape
                delta_y = delta_ph[None, None, :] * emo / m_pr

                int_fac = (delta_ec[:, None, None] *
                           delta_ph[None, None, :] / emo) * target_shape
                diff_fac = 1. / delta_x / delta_y

                # Generate boolean arrays to cut on xvalues
                if daid == 101:
                    cuts = np.logical_and(xl >= x_cut_proton, xl <= 1)
                else:
                    # or (yu < ymin) or (yl > y_cut)
                    cuts = np.logical_and(xl >= x_cut, xl <= 1)
                cuts = cuts[:, :, 0]

                # # NOTE JH: This is an old version, which brute force vectorizes the integral with numpy
                # # I am leaving this in the comments, in case we want to go back for testing-
                # integrator = np.vectorize(intp_diff.integral)
                # res = integrator(xl[cuts], xu[cuts], yl[cuts], yu[cuts]) * diff_fac[cuts] * int_fac[cuts]

                # This takes the average by evaluating the integral and dividing by bin width
                # intp_diff_integral contains the antiderivate, to to get the integral (xl,yl,xu,yu)
                # we need to substract INT = (0,0,xu,yu) - (0,0,xl,yu) - (0,0,xu,yl) +
                # (0,0,xl,yl)
                res = intp_diff_integral.ev(xu[cuts], yu[cuts])
                res -= intp_diff_integral.ev(xl[cuts], yu[cuts])
                res -= intp_diff_integral.ev(xu[cuts], yl[cuts])
                res += intp_diff_integral.ev(xl[cuts], yl[cuts])
                res *= diff_fac[cuts] * int_fac[cuts]
                res[res < 0] = 0.

                # Since we made cuts on x, we need to make the same cut on the index
                # mapping
                emoidx, edaidx, _ = np.meshgrid(sp_id_ref[moid].lidx() + emo_idcs,
                                                sp_id_ref[daid].lidx() + eda_idcs,
                                                p_idcs, indexing='ij')
                emoidx, edaidx = emoidx[cuts], edaidx[cuts]

                # Now add the nonel interactions on the main diagonal
                if has_nonel:
                    res[emoidx == edaidx] -= (
                        intp_nonel_antid(yu[cuts][emoidx == edaidx]) -
                        intp_nonel_antid(yl[cuts][emoidx == edaidx])) * (
                            diff_fac[cuts][emoidx == edaidx] *
                            int_fac[cuts][emoidx == edaidx])

                # Finally write this to the batch matrix
                self._batch_matrix[ibatch:ibatch + len(emoidx), :] = res
                self._batch_rows.append(edaidx[:, 0])
                self._batch_cols.append(emoidx[:, 0])
                ibatch += len(emoidx)

            else:
                info(20, 'Species combination not included in model', moid,
                     daid)

        self._batch_matrix = self._batch_matrix[:ibatch, :]
        self._batch_rows = np.concatenate(self._batch_rows, axis=None)
        self._batch_cols = np.concatenate(self._batch_cols, axis=None)
        self._batch_vec = np.zeros(ibatch)

        info(2, f'Batch matrix shape: {self._batch_matrix.shape}')
        info(2, f'Batch rows shape: {self._batch_rows.shape}')
        info(2, f'Batch cols shape: {self._batch_cols.shape}')
        info(2, f'Batch vector shape: {self._batch_vec.shape}')

        memory = (self._batch_matrix.nbytes + self._batch_rows.nbytes +
                  self._batch_cols.nbytes + self._batch_vec.nbytes) / 1024**2
        info(3, "Memory usage after initialization: {:} MB".format(memory))

    def _init_coupling_mat(self):
        """Initialises the coupling matrix directly in sparse (csr) format.
        """
        info(0, 'Initiating coupling matrix in ({:}) format'.format('CSR'))

        from scipy.sparse import csr_matrix
        if using_cupy:
            # For GPU we initialize the csr matrix on the host and then cast to GPU
            from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
            self.coupling_mat_np = csr_matrix(
                (self._batch_vec.astype(np.float32),
                    (self._batch_rows, self._batch_cols)),
                copy=True)
            self.coupling_mat = cp_csr_matrix(self.coupling_mat_np, copy=True)
            self._batch_vec = self.coupling_mat.data
            del self.coupling_mat_np
        else:
            self.coupling_mat = csr_matrix(
                (self._batch_vec, (self._batch_rows, self._batch_cols)),
                copy=True)

        # create an index to sort by rows and then columns,
        # which is the same ordering CSR has internally
        # lexsort sorts by last argument first!!!
        self.sortidx = np.lexsort((self._batch_cols, self._batch_rows))

        self._batch_rows = self._batch_rows[self.sortidx]
        self._batch_cols = self._batch_cols[self.sortidx]

        # Reorder batch matrix according to order in coupling_mat
        if using_cupy:
            self._batch_matrix = cupy.array(
                self._batch_matrix[self.sortidx, :], dtype=np.float32)
        else:
            self._batch_matrix = self._batch_matrix[self.sortidx, :]

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

            if using_cupy:
                if isinstance(self._batch_matrix, np.ndarray):
                    self._init_coupling_mat()
                cupy.dot(
                    self._batch_matrix, cupy.array(
                        self.photon_vector(z), dtype=np.float32),
                    out=self._batch_vec)
            else:
                np.dot(
                    self._batch_matrix, self.photon_vector(z), out=self._batch_vec)
            self._ratemat_zcache = z
            return True
        else:
            return False

    def _update_coupling_mat(self, z, scale_fac, force_update=False):
        """Updates the sparse (csr) coupling matrix
        Only the data vector is updated to minimize computation
        """
        # Do not execute dot product if photon field didn't change
        if self._update_rates(z, force_update):
            self.coupling_mat.data = scale_fac * self._batch_vec

    def get_hadr_jacobian(self, z, scale_fac=1., force_update=False):
        """Returns the nonel rate vector and coupling matrix.
        """
        self._update_coupling_mat(z, scale_fac, force_update)
        return self.coupling_mat

    def single_interaction_length(self, pid, z, pfield=None):
        """Returns energy loss length in cm
        (convenience function for plotting)
        """
        if pfield is not None:
            mem_pfield = self.photon_field
            self.photon_field = pfield

        species = self.spec_man.ncoid2sref[pid]
        egrid = self.e_cosmicray.grid * species.A
        rate = -1 * self.get_hadr_jacobian(
            force_update=True, z=z).toarray()[
                species.sl, species.sl].diagonal()

        length = 1 / rate

        self.photon_field = mem_pfield
        return egrid, length


class ContinuousAdiabaticLossRate(object):
    """Implementation of continuous pair production loss rates."""

    def __init__(self, prince_run, energy='grid', *args, **kwargs):
        info(3, 'creating instance')
        #: Reference to species manager
        self.spec_man = prince_run.spec_man

        # Initialize grids
        self.e_cosmicray = prince_run.cr_grid
        self.dim_states = prince_run.dim_states
        self.dim_bins = prince_run.dim_bins
        # Init adiabatic loss vector
        self.energy_vector = self._init_energy_vec(energy)

    def loss_vector(self, z, energy=None):
        """Returns all continuous losses on dim_states grid"""
        # return self.adiabatic_losses(z)
        from prince_cr.cosmology import H
        if energy is None:
            return H(z) * PRINCE_UNITS.cm2sec * self.energy_vector
        else:
            return H(z) * PRINCE_UNITS.cm2sec * energy

    def _init_energy_vec(self, energy):
        """Prepare vector for scaling with units, charge and mass."""
        if energy == 'grid':
            energy_vector = np.zeros(self.dim_states)
            for spec in self.spec_man.species_refs:
                energy_vector[spec.lidx():spec.uidx()] = self.e_cosmicray.grid
        elif energy == 'bins':
            energy_vector = np.zeros(self.dim_bins)
            for spec in self.spec_man.species_refs:
                energy_vector[spec.lbin():spec.ubin()] = self.e_cosmicray.bins
        else:
            raise Exception(
                'Unexpected energy keyword ({:}), use either (grid) or (bins)',
                format(energy))

        return energy_vector

    def single_loss_length(self, pid, z):
        """Returns energy loss length in cm
        (convenience function for plotting)
        """
        species = self.spec_man.ncoid2sref[pid]

        egrid = self.energy_vector[species.sl] * species.A
        rate = self.loss_vector(z)[species.sl] * species.A
        length = egrid / rate
        return egrid, length


class ContinuousPairProductionLossRate(object):
    """Implementation of continuous pair production loss rates."""

    def __init__(self, prince_run, energy='grid', *args, **kwargs):
        info(3, 'creating instance')
        #: Reference to species manager
        self.spec_man = prince_run.spec_man

        #: Reference to PhotonField object
        self.photon_field = prince_run.photon_field

        # Initialize grids
        self.e_cosmicray = prince_run.cr_grid
        self.dim_states = prince_run.dim_states
        self.dim_bins = prince_run.dim_bins
        self.e_photon = prince_run.ph_grid

        # xi is dimensionless (natural units) variable
        xi_steps = 400 if 'xi_steps' not in kwargs else kwargs['xi_steps']
        info(2, 'using', xi_steps, 'steps in xi')
        self.xi = np.logspace(np.log10(2 + 1e-8), 16., xi_steps)

        # weights for integration
        self.phi_xi2 = self._phi(self.xi) / (self.xi**2)

        # Scale vector containing the units and factors of Z**2 for nuclei
        self.scale_vec = self._init_scale_vec(energy)

        # Gamma factor of the cosmic ray
        if energy == 'grid':
            gamma = self.e_cosmicray.grid / PRINCE_UNITS.m_proton
        elif energy == 'bins':
            gamma = self.e_cosmicray.bins / PRINCE_UNITS.m_proton
        else:
            raise Exception(
                'Unexpected energy keyword ({:}), use either (grid) or (bins)',
                format(energy))
        # Grid of photon energies for interpolation
        self.photon_grid = np.outer(1 / gamma,
                                    self.xi) * PRINCE_UNITS.m_electron / 2.
        self.pg_desort = self.photon_grid.reshape(-1).argsort()
        self.pg_sorted = self.photon_grid.reshape(-1)[self.pg_desort]

    def loss_vector(self, z):
        """Returns all continuous losses on dim_states grid"""

        rate_single = trapz(
            self.photon_vector(z) * self.phi_xi2, self.xi, axis=1)
        pprod_loss_vector = self.scale_vec * np.tile(rate_single,
                                                     self.spec_man.nspec)

        return pprod_loss_vector

    def photon_vector(self, z):
        """Returns photon vector at redshift `z` on photon grid.

        This vector is in fact a matrix of vectors of the interpolated
        photon field with dimensions (dim_cr, xi_steps).

        Args:
            z (float): redshift

        Return value from cache if redshift value didn't change since last call.
        """
        photon_vector = np.zeros_like(self.photon_grid)
        photon_vector.reshape(-1)[
            self.pg_desort] = self.photon_field.get_photon_density(
                self.pg_sorted, z)

        return photon_vector

    def _init_scale_vec(self, energy):
        """Prepare vector for scaling with units, charge and mass."""
        if energy == 'grid':
            scale_vec = np.zeros(self.dim_states)
            units = (PRINCE_UNITS.fine_structure * PRINCE_UNITS.r_electron**2 *
                     PRINCE_UNITS.m_electron**2)
            for spec in self.spec_man.species_refs:
                if not spec.is_nucleus:
                    continue
                scale_vec[spec.lidx():spec.uidx()] = units * abs(
                    spec.charge)**2 / float(spec.A) * np.ones_like(
                        self.e_cosmicray.grid, dtype='double')
        elif energy == 'bins':
            scale_vec = np.zeros(self.dim_bins)
            units = (PRINCE_UNITS.fine_structure * PRINCE_UNITS.r_electron**2 *
                     PRINCE_UNITS.m_electron**2)
            for spec in self.spec_man.species_refs:
                if not spec.is_nucleus:
                    continue
                scale_vec[spec.lbin():spec.ubin()] = units * abs(
                    spec.charge)**2 / float(spec.A) * np.ones_like(
                        self.e_cosmicray.bins, dtype='double')
        else:
            raise Exception(
                'Unexpected energy keyword ({:}), use either (grid) or (bins)',
                format(energy))
        return scale_vec

    def single_loss_length(self, pid, z, pfield=None):
        """Returns energy loss length in cm
        (convenience function for plotting)
        """
        if pfield is not None:
            mem_pfield = self.photon_field
            self.photon_field = pfield

        species = self.spec_man.ncoid2sref[pid]

        egrid = self.e_cosmicray.grid * species.A
        rate = self.loss_vector(z)[species.sl] * species.A
        length = egrid / rate

        self.photon_field = mem_pfield
        return egrid, length

    def _phi(self, xi):
        """Phi function as in Blumental 1970"""

        # Simple ultrarelativistic approximation by Blumental 1970
        bltal_ultrarel = np.poly1d([2.667, -14.45, 50.95, -86.07])
        def phi_simple(xi): return xi * bltal_ultrarel(np.log(xi))

        # random fit parameters, see Chorodowski et al
        c1 = 0.8048
        c2 = 0.1459
        c3 = 1.137e-3
        c4 = -3.879e-6

        f1 = 2.91
        f2 = 78.35
        f3 = 1837

        res = np.zeros(xi.shape)

        le = np.where(xi < 25.)
        he = np.where(xi >= 25.)

        res[le] = np.pi / 12. * (xi[le] - 2)**4 / (
            c1 * (xi[le] - 2)**1 + c2 * (xi[le] - 2)**2 + c3 *
            (xi[le] - 2)**3 + c4 * (xi[le] - 2)**4)

        res[he] = phi_simple(
            xi[he]) / (1 - f1 * xi[he]**-1 - f2 * xi[he]**-2 - f3 * xi[he]**-3)

        return res
