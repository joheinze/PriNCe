"""The module contains classes for computations of interaction rates"""

import numpy as np

from prince.util import get_AZN, get_interp_object, info, load_or_convert_array
from prince_config import config


class PhotoNuclearInteractionRate(object):
    """This class handles the computation of interaction rates."""

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
            self.e_photon = EnergyGrid(*config["photon_grid"])
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
        self.z_cache = None
        self._photon_vector = None

        # Zero matrix
        self.zeros = np.zeros((self.dim_cr, self.dim_ph))

        # Initialize the response matrices on dim_cr x dim_ph grid
        self._init_matrices()

    def _init_matrices(self):
        from util import get_y
        # Iniialize cross section matrices, evaluated on a grid of y values
        x, y = np.meshgrid(self.e_photon.grid, self.e_cosmicray.grid)

        # Delta eps (photon energy) bin widths
        delta_eps = np.diag(self.e_photon.widths)

        # TODO: Obsolete, remove it
        # if species_manager is None:
        #     known_species = cross_section.nonel_idcs
        # else:
        #     known_species = species_manager.known_species

        # Compute y matrix only once and then rescale by A
        ymat = get_y(x, y, 100)

        #: Matrices containing response function sampled on ymat
        #: TODO: (shall become obsolete)
        self.resp_matrices = {}

        # One big matrix for batch re-compute nonel rates at once
        self.nonel_batch_pointer = {}
        self.nonel_batch_matrix = np.zeros(
            (self.dim_cr * len(self.cross_sections.nonel_idcs), self.dim_ph))
        info(2,
             'Size of nonel batch matrix: {0}x{1}'.format(*self.nonel_batch_matrix.shape))

        # One big matrix for batch re-compute incl rates at once
        self.incl_batch_pointer = {}
        self.coupling_mat_pointers = {}
        self.incl_batch_matrix = np.zeros(
            (self.dim_cr * len(self.cross_sections.incl_idcs), self.dim_ph))
        info(2,
             'Size of incl batch matrix: {0}x{1}'.format(*self.incl_batch_matrix.shape))

        # Define shortcut for converting ncoid to prince index
        pridx = self.spec_man.ncoid2princeidx

        fill_idx = 0
        for mother in self.spec_man.known_species:
            if mother <= 100:
                info(3, "Can not compute interaction rate for", mother)
                # TODO: Save work by ignoring non-hadronic species
                continue
            A = get_AZN(mother)[0]
            self.resp_matrices[mother] = self.cross_sections.resp_nonel_intp[
                mother](ymat).dot(delta_eps)

            # TODO: Duplicate entry for batch compute, the normal dictionary can be
            # removed when replacement using the index vector implemented
            self.nonel_batch_matrix[fill_idx * self.dim_cr:(
                fill_idx + 1) * self.dim_cr] = self.resp_matrices[mother]
            self.nonel_batch_pointer[mother] = fill_idx
            fill_idx += 1

        fill_idx = 0
        for (mo, da) in self.cross_sections.incl_idcs:
            # Compute rates of inclusive reactions
            self.resp_matrices[(mo, da)] = self.cross_sections.resp_incl_intp[(
                mo, da)](ymat).dot(delta_eps)
            # TODO: Here same thing with duplicate

            # Indices in batch matrix
            lidx, ridx = fill_idx * self.dim_cr, (fill_idx + 1) * self.dim_cr

            # Staple ("vstack"") all inclusive (channel) response functions
            self.incl_batch_matrix[lidx:ridx] = self.resp_matrices[(mo, da)]

            # Remember how to find the entry for a response function/rate in the
            # matrix or result vector
            self.incl_batch_pointer[(mo, da)] = (lidx, ridx)

            # Create association between prince_idx and position in resulting
            # rate vector
            mo_pridx, da_pridx = pridx[mo], pridx[da]
            self.coupling_mat_pointers[(mo_pridx, da_pridx)] = (lidx, ridx)

            fill_idx += 1

        # Initialize coupling matrix structure, containing the references
        # to views of the batch vector
        self.coupling_mat_refs = self.spec_man.nspec * [
            self.spec_man.nspec * [None]
        ]
    
    # TODO: Obsolete stuff below
    # def f_submat(self, mother):
    #     """Returns redistribution function f matrix.
    #     """
    #     if mother < 100:
    #         return self.zeros

    #     return self.resp_matrices[mother]

    # def g_submat(self, mother, daughter):
    #     """Returns redistribution function g matrix for inclusive channel.

    #     """
    #     A, _, _ = get_AZN(mother)
    #     B, _, _ = get_AZN(daughter)

    #     if mother < 100:
    #         return self.zeros
    #     elif (mother, daughter) == (101, 101):
    #         # TODO: Workaround for missing redistribution functions
    #         return 0.8 * self.resp_matrices[(mother, daughter)]

    #     return self.resp_matrices[(mother, daughter)]

    def photon_vector(self, z):
        """Returns photon vector."""

        self._set_photon_vector(z)

        return self._photon_vector

    def _set_photon_vector(self, z):
        """Cache photon vector for the previous value of z.

        Args:
            z (float): redshift
        """

        if self.z_cache != z:
            self._photon_vector = self.photon_field.get_photon_density(
                self.e_photon.grid, z)
            self.z_cache = z
    
    # TODO: Obsolete stuff
    # def fg_submat(self, nco_ids, z):
    #     if isinstance(nco_ids, tuple) and nco_ids[0] == nco_ids[1]:
    #         print 'fg_submat', nco_ids
    #         return (-self.f_submat(nco_ids[0]) +
    #                 self.g_submat(*nco_ids)).dot(self.photon_vector(z))

    #     # Convolve using matrix multiplication
    #     return self.g_submat(*nco_ids).dot(self.photon_vector(z))

    def interation_rate(self, nco_ids, z):
        """Compute interaction rates using matrix convolution.

        This method is a high performance integration of equation (10)
        from internal note, using a simple box method.

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
        return self.resp_matrices[nco_ids].dot(self.photon_vector(z))

    def _batch_compute_int_rate(self, z):
        # Batch compute nonel and inclusive rates
        nonel_rates = self.nonel_batch_matrix.dot(self.photon_vector(z))
        incl_rates = self.incl_batch_matrix.dot(self.photon_vector(z))

        for (mo_pridx, da_pridx), (lidx, ridx) in self.coupling_mat_pointers:
            self.coupling_mat_refs[da_pridx, mo_pridx] = incl_rates[lidx, ridx]

        return nonel_rates, incl_rates

    def sparse_rate_matstruc(self, z):
        batch_rate_vec = self._batch_compute_int_rate(z)
        self.abs_rates = np.zeros(self.prince_run.dim_states)
        mat_struc = []
        for i in self.spec_man.nspec:
            pass

    def sparse_interaction_rate(self, nco_ids, z):
        """Identical to :func:`interaction_rate` but returns a sparse
        `dia` matrix."""
        from scipy.sparse import dia_matrix

        return dia_matrix(
            (self.interation_rate(nco_ids, z), [0]),
            shape=(self.dim_cr, self.dim_cr))
