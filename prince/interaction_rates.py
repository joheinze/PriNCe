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
                 species_list=None,
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

        # Initialize cache of redshit value
        self.z_cache = None
        self.photon_vector = None

        # Iniialize cross section matrices, evaluated on a grid of
        # y values
        x, y = np.meshgrid(self.e_photon.grid, self.e_cosmicray.grid)
        self.matrix = {}
        delta_eps = np.diag(self.e_photon.widths)
        # Note: No advantage from sparse matrices, since matrix too small
        # to benefit from sparse algebra.
        if species_list is None:
            species_list = cross_section.nonel_idcs

        # Compute y matrix only once and then rescale by A
        ymat = get_y(x, y, 100)
        from scipy.sparse import csr_matrix
        # Warning!! Don't divide by A for energy per nucleon grid, currently
        # per nucleus.
        for mother in species_list:
            A = get_AZN(mother)[0]
            self.matrix[mother] = csr_matrix(self.cross_section.resp_nonel_intp[mother](
                ymat/A).dot(delta_eps))
            if "ignore_incl" in kwargs and kwargs["ignore_incl"]:

                continue
            # Compute rates of inclusive reactions
            for (mo, da) in self.cross_section.reactions[mother]:
                self.matrix[(mo, da)] = csr_matrix(self.cross_section.resp_incl_intp[(
                    mo, da)](ymat/A).dot(delta_eps))

    def _set_photon_vector(self, z):
        """Cache photon vector for the previous value of z.

        Args:
            z (float): redshift
        """

        if self.z_cache != z:
            self.photon_vector = self.photon_field.get_photon_density(
                self.e_photon.grid, z)
            self.z_cache = z

    def get_interation_rate(self, nco_id, z):
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
        # Check if z has changed and recompute photon vector if needed
        self._set_photon_vector(z)

        # Convolve using matrix multiplication
        return self.matrix[nco_id].dot(self.photon_vector)
