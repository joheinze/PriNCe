'''Provides user interface and runtime management.'''

import cPickle as pickle
from os import path
from prince import photonfields, intcs, interaction_rates, data, util, solvers
from prince.util import info
from prince_config import config, spec_data
import numpy as np


class PriNCeRun(object):
    """This is a draft of the main class.

    This class is supposed to interprete the config options
    and initialize all necessary stuff in order. This class is
    meant to keep all separate ingredients together in one place,
    and it is inteded to be passed to further classes via `self`.
    """

    def __init__(self, *args, **kwargs):

        # Initialize energy grid
        self.cr_grid = util.EnergyGrid(*config["cosmic_ray_grid"])
        # energy grid
        self._egrid = self.cr_grid.grid
        #: Dimension of energy grid
        self.ed = self.cr_grid.d

        # Cross section handler
        self.cross_sections = intcs.CrossSectionInterpolator(
            [(0., intcs.NeucosmaFileInterface, ()),
             (0.8, intcs.SophiaSuperposition, ())])

        # Photon field handler
        self.photon_field = photonfields.CombinedPhotonField(
            [photonfields.CMBPhotonSpectrum, photonfields.CIBInoue2D])

        # Store adv_set
        self.adv_set = config["adv_settings"]

        # Limit max nuclear mass of eqn system
        system_species = self.cross_sections.known_species
        if "max_mass" in kwargs:
            system_species = [
                s for s in system_species if s < 100 * (kwargs["max_mass"] + 1)
            ]
        # Initialize species manager for all species for which cross sections are known
        self.spec_man = data.SpeciesManager(system_species, self.ed)

        # Total dimension of system
        self.dim_states = self.ed * self.spec_man.nspec

        # Initialize the interaction rates
        self.int_rates = interaction_rates.PhotoNuclearInteractionRate(
            prince_run=self)

        # Let species manager know about the photon grid dimensions (for idx calculations)
        # it is accesible under index "ph" for lidx(), uidx() calls
        self.spec_man.add_grid('ph', self.int_rates.dim_ph)


        # TODO: Move all matrix generation to the solvers module
        # # Assemble the interaction matrix (-F + G)
        # self._fill_interaction_matrices()

    # def _fill_interaction_matrices(self):
    #     """Creates the F and the G matrix.

    #     """
    #     from scipy.sparse import bsr_matrix
    #     # Dimension of matrices in target photon axis
    #     dim_states_photon = self.spec_man.nspec * self.int_rates.dim_ph

    #     info(1, 'Creating F and G matrix.')
    #     self.FGmat = np.zeros((self.dim_states, dim_states_photon))
    #     self.struct_mat = np.zeros((self.dim_states, self.dim_states))
    #     struct_eye = np.eye(self.int_rates.dim_cr)
    #     return
    #     for s in self.spec_man.species_refs:
    #         info(5, 'Fill matrices for species', s.ncoid)
    #         if s.ncoid < 100:
    #             info(3, 'No interactions for', s.ncoid)
    #             continue
    #         # Store G matrix elements
    #         for mo, da in self.cross_sections.reactions[s.ncoid]:
    #             da_ref = self.spec_man.ncoid2sref[da]
    #             # self.FGmat[da_ref.lidx():da_ref.uidx(
    #             # ), s.lidx("ph"):s.uidx("ph")] += self.int_rates.g_submat(
    #             #     mo, da)
    #             # self.struct_mat[da_ref.lidx():da_ref.uidx(),
    #             #                 s.lidx():s.uidx()] = struct_eye

    #         # Add F matrix elements
    #         # self.FGmat[s.lidx():s.uidx(
    #         # ), s.lidx("ph"):s.uidx("ph")] -= self.int_rates.f_submat(s.ncoid)

    #     # self.FGmat = bsr_matrix(
    #     #     self.FGmat,
    #     #     blocksize=(self.int_rates.dim_cr, self.int_rates.dim_ph))
    #     # self.Gmat = csr_matrix(self.Gmat)

    #     info(1, 'Done.')

    @property
    def egrid(self):
        """Energy grid used for species."""
        return self._egrid
