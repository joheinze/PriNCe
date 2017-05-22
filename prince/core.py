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
    and initialize all necessary stuff. Then it provides methods
    to handle the computation. Maybe it will be split in different
    parts.
    """

    def __init__(self, *args, **kwargs):

        # Initialize energy grid
        self._grid = util.EnergyGrid(*config["cosmic_ray_grid"])
        # energy grid
        self._egrid = self._grid.grid
        #: Dimension of energy grid
        self.ed = self._grid.d

        # Cross section handler
        self.cross_sections = intcs.CrossSectionInterpolator(
            [(0., intcs.NeucosmaFileInterface, ()),
             (0.8, intcs.SophiaSuperposition, ())])

        # Photon field handler
        self.photon_field = photonfields.CombinedPhotonField(
            [photonfields.CMBPhotonSpectrum, photonfields.CIBFranceschini2D])

        # Store adv_set
        self.adv_set = config["adv_settings"]

        # Replace it temporarily by a system with A <= 4
        self.spec_man = data.SpeciesManager(
            [mo for mo in self.cross_sections.nonel_idcs if mo < 500], self.ed)

        # Total dimension of system
        self.dim_states = self.ed * self.spec_man.nspec

        # Initialize the interaction rates
        self.int_rates = interaction_rates.PhotoNuclearInteractionRate(
            photon_field=self.photon_field,
            cross_section=self.cross_sections,
            cr_grid=self._grid,
            species_mananager=self.spec_man)

        self.sim_environment = solvers.ExtragalacticSpace(int_rates)

    @property
    def egrid(self):
        """Energy grid used for species."""
        return self._egrid
