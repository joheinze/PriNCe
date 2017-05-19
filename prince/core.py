'''
Created on Feb 22, 2017

@author: Anatoli Fedynitch
'''

import cPickle as pickle
from os import path
from prince import photonfields, intcs, interaction_rates, data, util
from util import info
from prince_config import config
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

        #: Dictionary containing particle properties, like mass, charge
        #: lifetime or branching ratios
        self.pd = pickle.load(
            open(path.join(config["data_dir"], "particle_data.ppo"), "rb"))

        # Replace it temporarily by a system with A <= 4
        self.species_refs = self._gen_species_list(
            [mo for mo in self.cross_sections.nonel_idcs if mo < 500])

        # Initialize the interaction rates
        self.int_rates = interaction_rates.PhotoNuclearInteractionRate(
            photon_field=self.photon_field,
            cross_section=self.cross_sections,
            cr_grid=self._grid,
            species_list=[
                mo for mo in self.cross_sections.nonel_idcs if mo < 500
            ])

        # Store adv_set
        self.adv_set = config["adv_settings"]

    def _gen_species_list(self, ncoid_list=None):
        info(2, "Generating list of species.")
        # TODO: Probably bad idea
        if ncoid_list is None:
            ncoid_list = self.cross_sections.nonel_idcs
        ncoid_list += self.pd["non_nuclear_ids"]
        ncoid_list.sort()
        
        species_refs = []
        for ncoid in ncoid_list:
            species_refs.append(data.PrinceSpecies(ncoid, self.pd, self.ed))
        return species_refs

    def _init_particle_tables(self):

        # Particle index shortcuts
        #: (dict) Converts PDG ID to index in state vector
        self.ncoid2princeidx = {}
        #: (dict) Converts particle name to index in state vector
        self.pname2princeidx = {}
        #: (dict) Converts PDG ID to reference of :class:`data.MCEqParticle`
        self.ncoid2pref = {}
        #: (dict) Converts particle name to reference of :class:`data.MCEqParticle`
        self.pname2pref = {}
        #: (dict) Converts index in state vector to PDG ID
        self.princeidx2ncoid = {}
        #: (dict) Converts index in state vector to reference of :class:`data.MCEqParticle`
        self.princeidx2pname = {}

        for p in self.particle_species:
            try:
                princeidx = p.princeidx
            except:
                princeidx = -1
            self.ncoid2princeidx[p.ncoid] = princeidx
            self.pname2princeidx[p.name] = princeidx
            self.princeidx2ncoid[princeidx] = p.ncoid
            self.princeidx2pname[princeidx] = p.name
            self.ncoid2pref[p.ncoid] = p
            self.pname2pref[p.name] = p

    @property
    def egrid(self):
        """Energy grid used for species."""
        return self._egrid
