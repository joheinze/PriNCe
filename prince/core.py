'''Provides user interface and runtime management.'''

import cPickle as pickle
from os import path
from prince import photonfields, cross_sections, interaction_rates, data, util, solvers
from prince.util import info, get_AZN
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

        # TODO: dirty workarround, pass max mass to config
        # to delete heavier particle from crosssection
        if "max_mass" in kwargs:
            config["max_mass"] = kwargs["max_mass"]

        # Initialize energy grid
        self.cr_grid = util.EnergyGrid(*config["cosmic_ray_grid"])

        # Photon grid for rate computations
        self.ph_grid = util.EnergyGrid(*config["photon_grid"])

        #: Dimension of energy grid
        self.ed = self.cr_grid.d

        # Cross section handler
        self.cross_sections = cross_sections.CompositeCrossSection(
            [(0., cross_sections.TabulatedCrossSection, ('CRP2_TALYS',)),
             (0.14, cross_sections.SophiaSuperposition, ())])
        
        # self.cross_sections = cross_sections.SophiaSuperposition()

        # Photon field handler
        if 'photon_field' in kwargs:
            self.photon_field = kwargs['photon_field']
        else:
            self.photon_field = photonfields.CombinedPhotonField(
                [photonfields.CMBPhotonSpectrum, 
                 photonfields.CIBFranceschini2D])

        # Store adv_set
        self.adv_set = config["adv_settings"]

        # Limit max nuclear mass of eqn system
        system_species = [
            s for s in self.cross_sections.known_species
            if get_AZN(s)[0] <= config["max_mass"]
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

        self.cross_sections
        # Let species manager know about the photon grid dimensions (for idx calculations)
        # it is accesible under index "ph" for lidx(), uidx() calls
        self.spec_man.add_grid('ph', self.int_rates.dim_ph)

    @property
    def egrid(self):
        """Energy grid used for species."""
        return self.cr_grid.grid
