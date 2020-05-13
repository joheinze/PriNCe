'''Provides user interface and runtime management.'''

import pickle as pickle
from os import path

import numpy as np

from prince_cr import cross_sections, data, interaction_rates
from prince_cr.data import EnergyGrid
from prince_cr.util import info, get_AZN
import prince_cr.config as config


class PriNCeRun(object):
    """This is a draft of the main class.

    This class is supposed to interprete the config options
    and initialize all necessary stuff in order. This class is
    meant to keep all separate ingredients together in one place,
    and it is inteded to be passed to further classes via `self`.
    """

    def __init__(self, *args, **kwargs):

        if "max_mass" in kwargs:
            max_mass = kwargs.pop("max_mass", config.max_mass)

        # Initialize energy grid
        if config.grid_scale == 'E':
            info(1, 'initialising Energy grid')
            self.cr_grid = EnergyGrid(*config.cosmic_ray_grid)
            self.ph_grid = EnergyGrid(*config.photon_grid)
        else:
            raise Exception(
                "Unknown energy grid scale {:}, adjust config.grid_scale".
                format(config.grid_scale))

        # Cross section handler
        if 'cross_sections' in kwargs:
            self.cross_sections = kwargs['cross_sections']
        else:
            self.cross_sections = cross_sections.CompositeCrossSection(
                [(0., cross_sections.TabulatedCrossSection, ('CRP2_TALYS', )),
                 (0.14, cross_sections.SophiaSuperposition, ())])

        # Photon field handler
        if 'photon_field' in kwargs:
            self.photon_field = kwargs['photon_field']
        else:
            import prince_cr.photonfields as pf
            self.photon_field = pf.CombinedPhotonField(
                [pf.CMBPhotonSpectrum, pf.CIBGilmore2D])

        # Limit max nuclear mass of eqn system
        if "species_list" in kwargs:
            system_species = list(
                set(kwargs["species_list"]) & set(
                    self.cross_sections.known_species))
        else:
            system_species = [
                s for s in self.cross_sections.known_species
                if get_AZN(s)[0] <= max_mass
            ]
        # Disable photo-meson production
        if not config.secondaries:
            system_species = [s for s in system_species if s >= 100]
        # Remove particles that are explicitly excluded
        for pid in config.ignore_particles:
            if pid in system_species:
                system_species.remove(pid)

        # Initialize species manager for all species for which cross sections are known
        self.spec_man = data.SpeciesManager(system_species, self.cr_grid.d)

        # Total dimension of system
        self.dim_states = self.cr_grid.d * self.spec_man.nspec
        self.dim_bins = (self.cr_grid.d + 1) * self.spec_man.nspec

        # Initialize continuous energy losses
        self.adia_loss_rates_grid = interaction_rates.ContinuousAdiabaticLossRate(
            prince_run=self, energy='grid')
        self.pair_loss_rates_grid = interaction_rates.ContinuousPairProductionLossRate(
            prince_run=self, energy='grid')
        self.adia_loss_rates_bins = interaction_rates.ContinuousAdiabaticLossRate(
            prince_run=self, energy='bins')
        self.pair_loss_rates_bins = interaction_rates.ContinuousPairProductionLossRate(
            prince_run=self, energy='bins')

        # Initialize the interaction rates
        self.int_rates = interaction_rates.PhotoNuclearInteractionRate(
            prince_run=self)

        # Let species manager know about the photon grid dimensions (for idx calculations)
        # it is accesible under index "ph" for lidx(), uidx() calls
        self.spec_man.add_grid('ph', self.ph_grid.d)


    def set_photon_field(self, pfield):
        self.photon_field = pfield
        self.adia_loss_rates_grid.photon_field = pfield
        self.pair_loss_rates_grid.photon_field = pfield
        self.adia_loss_rates_bins.photon_field = pfield
        self.pair_loss_rates_bins.photon_field = pfield
        self.int_rates.photon_field = pfield
