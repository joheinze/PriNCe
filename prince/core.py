'''
Created on Feb 22, 2017

@author: Anatoli Fedynitch
'''

import cPickle as pickle
from os import path
from prince import photonfields, intcs, interaction_rates, data, util
from util import info
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
        self.species_refs = self._gen_species_list(
            [mo for mo in self.cross_sections.nonel_idcs if mo < 500])

        # Create tables for shortcuts and conversions
        self._init_species_tables()

        # Further short-cuts depending on previous initializations
        self.n_tot_species = len(self.species_refs)

        # Total dimension of system
        self.dim_states = self.ed * self.n_tot_species

        # Initialize the interaction rates
        self.int_rates = interaction_rates.PhotoNuclearInteractionRate(
            photon_field=self.photon_field,
            cross_section=self.cross_sections,
            cr_grid=self._grid,
            species_list=[sorted(self.ncoid2sref.keys())])

    def _gen_species_list(self, ncoid_list=None):
        info(2, "Generating list of species.")
        # TODO: Probably bad idea multi-use the argument
        if ncoid_list is None:
            ncoid_list = self.cross_sections.nonel_idcs
        ncoid_list += spec_data["non_nuclear_species"]
        # Make sure list is unique and sorted
        ncoid_list = sorted(list(set(ncoid_list)))

        species_refs = []
        # Define position in state vector (princeidx) by simply
        # incrementing it with the (sorted) list of Neucosma IDs
        for princeidx, ncoid in enumerate(ncoid_list):
            info(3, "Appending species {0} at position {1}".format(
                ncoid, princeidx))
            species_refs.append(
                data.PrinceSpecies(ncoid, princeidx, spec_data, self.ed))
        return species_refs

    def _init_species_tables(self):

        # Particle index shortcuts
        #: (dict) Converts Neucosma ID to index in state vector
        self.ncoid2princeidx = {}
        #: (dict) Converts particle name to index in state vector
        self.sname2princeidx = {}
        #: (dict) Converts Neucosma ID to reference of :class:`data.MCEqParticle`
        self.ncoid2sref = {}
        #: (dict) Converts particle name to reference of :class:`data.MCEqParticle`
        self.sname2sref = {}
        #: (dict) Converts index in state vector to Neucosma ID
        self.princeidx2ncoid = {}
        #: (dict) Converts index in state vector to reference of :class:`data.MCEqParticle`
        self.princeidx2pname = {}

        for p in self.species_refs:
            self.ncoid2princeidx[p.ncoid] = p.princeidx
            self.sname2princeidx[p.sname] = p.princeidx
            self.princeidx2ncoid[p.princeidx] = p.ncoid
            self.princeidx2pname[p.princeidx] = p.sname
            self.ncoid2sref[p.ncoid] = p
            self.sname2sref[p.sname] = p
    

    @property
    def egrid(self):
        """Energy grid used for species."""
        return self._egrid
