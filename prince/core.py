'''
Created on Feb 22, 2017

@author: Anatoli Fedynitch
'''

from prince import photonfields, intcs, interaction_rates, util
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
        self.grid_instance = util.EnergyGrid(*config["cosmic_ray_grid"])
        self._egrid = self.grid_instance.grid

        # Cross section handler
        self.cross_sections = intcs.CrossSectionInterpolator(
            [(0., intcs.NeucosmaFileInterface, ()),
             (0.8, intcs.SophiaSuperposition, ())])

        # Photon field handler
        self.photon_field = photonfields.CombinedPhotonField(
            [photonfields.CMBPhotonSpectrum, photonfields.CIBFranceschini2D])

        # Gather the list of particles which is supported by model
        self.mothers = self.cross_sections.nonel_idcs
        # Replace it temporarily by a system with A <= 4
        self.mothers = [mo for mo in self.mothers if mo < 500]

        # Initialize the interaction rates
        self.int_rates = interaction_rates.PhotoNuclearInteractionRate(
            photon_field=self.photon_field,
            cross_section=self.cross_sections,
            cr_grid=self.grid_instance,
            species_list=self.mothers)





    @property
    def egrid(self):
        """Energy grid used for species."""
        return self._egrid
