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

        # Photon grid for rate computations
        self.ph_grid = util.EnergyGrid(*config["photon_grid"])

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

        # Initialize continuous energy losses
        self.continuous_losses = interaction_rates.ContinuousLossRates(
            prince_run=self)

        # Let species manager know about the photon grid dimensions (for idx calculations)
        # it is accesible under index "ph" for lidx(), uidx() calls
        self.spec_man.add_grid('ph', self.int_rates.dim_ph)

    @property
    def egrid(self):
        """Energy grid used for species."""
        return self.cr_grid.grid
