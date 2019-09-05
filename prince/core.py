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
        if config["grid_scale"] == 'E':
            info(1, 'initialising Energy grid')
            self.cr_grid = util.EnergyGrid(*config["cosmic_ray_grid"])
            self.ph_grid = util.EnergyGrid(*config["photon_grid"])
        elif config["grid_scale"] == 'logE':
            info(1, 'initialising logEnergy grid')
            self.cr_grid = util.LogEnergyGrid(*config["cosmic_ray_grid"])
            self.ph_grid = util.LogEnergyGrid(*config["photon_grid"])
        else:
            raise Exception(
                "Unknown energy grid scale {:}, adjust config['grid_scale']".
                format(config['grid_scale']))
        # Dimension of energy grid
        self.ed = self.cr_grid.d

        # Cross section handler
        if 'cross_sections' in kwargs:
            self.cross_sections = kwargs['cross_sections']
        else:
            self.cross_sections = cross_sections.CompositeCrossSection(
                [(0., cross_sections.TabulatedCrossSection, ('CRP2_TALYS', )),
                 (0.14, cross_sections.SophiaSuperposition, ())])
            # self.cross_sections = cross_sections.CompositeCrossSection(
            #     [(0., cross_sections.TabulatedCrossSection, ('peanut_IAS',)),
            #      (0.14, cross_sections.SophiaSuperposition, ())])

        # Photon field handler
        if 'photon_field' in kwargs:
            self.photon_field = kwargs['photon_field']
        else:
            self.photon_field = photonfields.CombinedPhotonField(
                [photonfields.CMBPhotonSpectrum, photonfields.CIBGilmore2D])

        # Store adv_set
        self.adv_set = config["adv_settings"]

        # Limit max nuclear mass of eqn system
        if "species_list" in kwargs:
            system_species = list(
                set(kwargs["species_list"]) & set(
                    self.cross_sections.known_species))
        else:
            system_species = [
                s for s in self.cross_sections.known_species
                if get_AZN(s)[0] <= config["max_mass"]
            ]
        # If secondaries are disabled in config, delete them from system species
        if not config["secondaries"]:
            system_species = [s for s in system_species if s >= 100]
        # Remove particles that are explicitly excluded
        for pid in config["ignore_particles"]:
            system_species.remove(pid)

        # Initialize species manager for all species for which cross sections are known
        self.spec_man = data.SpeciesManager(system_species, self.ed)

        # Total dimension of system
        self.dim_states = self.ed * self.spec_man.nspec
        self.dim_bins = (self.ed + 1) * self.spec_man.nspec

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
        self.spec_man.add_grid('ph', self.int_rates.dim_ph)

    @property
    def egrid(self):
        """Energy grid used for single species state"""
        return self.cr_grid.grid

    @property
    def ebins(self):
        """Energy bins used for single species state"""
        return self.cr_grid.bins

    def compute_propagation(self, initial_z=1., final_z=0., ncoid=101):
        """Computes a propagation with standard model input (see kwargs)
            WARNING: Not yet fully implemented!
        """
        from solvers import UHECRPropagationSolver
        solver = UHECRPropagationSolver(
            initial_z=initial_z, final_z=final_z, prince_run=self)

    def set_photon_field(self, pfield):
        self.photon_field = pfield
        self.adia_loss_rates_grid.photon_field = pfield
        self.pair_loss_rates_grid.photon_field = pfield
        self.adia_loss_rates_bins.photon_field = pfield
        self.pair_loss_rates_bins.photon_field = pfield
        self.int_rates.photon_field = pfield