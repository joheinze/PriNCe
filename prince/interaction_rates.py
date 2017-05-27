"""The module contains classes for computations of interaction rates"""

import numpy as np

from prince.util import get_AZN, get_interp_object, info, load_or_convert_array
from prince_config import config


class PhotoNuclearInteractionRate(object):
    """This class handles the computation of interaction rates."""

    def __init__(self,
                 photon_field,
                 cross_section,
                 cr_grid=None,
                 species_manager=None,
                 *args,
                 **kwargs):
        from prince.util import EnergyGrid, get_y

        # Reference to PhotonField object
        self.photon_field = photon_field

        # Reference to CrossSection object
        self.cross_section = cross_section

        # Initialize grids
        if "photon_bins_per_dec" not in kwargs:
            self.e_photon = EnergyGrid(*config["photon_grid"])
        else:
            info(2, 'Overriding number of photon bins from config.')

            self.e_photon = EnergyGrid(config["photon_grid"][0],
                                       config["photon_grid"][1],
                                       kwargs["photon_bins_per_dec"])
        if cr_grid is None:
            self.e_cosmicray = EnergyGrid(*config["cosmic_ray_grid"])
        else:
            self.e_cosmicray = cr_grid

        # Create shortcuts for grid dimensions
        self.dim_cr = self.e_cosmicray.d
        self.dim_ph = self.e_photon.d
        # Initialize cache of redshit value
        self.z_cache = None
        self._photon_vector = None

        # Zero matrix
        self.zeros = np.zeros((self.dim_cr, self.dim_ph))

        # Iniialize cross section matrices, evaluated on a grid of
        # y values
        x, y = np.meshgrid(self.e_photon.grid, self.e_cosmicray.grid)
        self.matrix = {}
        delta_eps = np.diag(self.e_photon.widths)
        # Note: No advantage from sparse matrices, since matrix too small
        # to benefit from sparse algebra.
        known_species = []
        if species_manager is None:
            known_species = cross_section.nonel_idcs
        else:
            known_species = species_manager.known_species

        self.nspec = len(known_species)

        # Compute y matrix only once and then rescale by A
        ymat = get_y(x, y, 100)

        # TDOD: Warning. Removed division of ymat by A to output rates in energy
        # per nucleon
        for mother in known_species:
            if mother <= 100:
                info(3, "Can not compute interaction rate for", mother)
                continue
            A = get_AZN(mother)[0]
            self.matrix[mother] = self.cross_section.resp_nonel_intp[mother](
                ymat).dot(delta_eps)
            # Compute rates of inclusive reactions
            for (mo, da) in self.cross_section.reactions[mother]:
                self.matrix[(mo, da)] = self.cross_section.resp_incl_intp[(
                    mo, da)](ymat).dot(delta_eps)

    def f_submat(self, mother):
        """Returns redistribution function f matrix.
        """
        if mother < 100:
            return self.zeros

        return self.matrix[mother]

    def g_submat(self, mother, daughter):
        """Returns redistribution function g matrix for inclusive channel.

        """
        if mother < 100:
            return self.zeros

        return self.matrix[(mother, daughter)]

    def photon_vector(self, z):
        """Return photon vector, repeated ntiles times."""

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

    def fg_submat(self, nco_ids, z):
        if isinstance(nco_ids, tuple) and nco_ids[0] == nco_ids[1]:
            return (-self.f_submat(nco_ids[0]) +
                    self.g_submat(*nco_ids)).dot(self.photon_vector(z))

        # Convolve using matrix multiplication
        return self.matrix[nco_ids].dot(self.photon_vector(z))

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
        return self.matrix[nco_ids].dot(self.photon_vector(z))
