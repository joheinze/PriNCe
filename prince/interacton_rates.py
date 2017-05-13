from abc import ABCMeta, abstractmethod
from os.path import isfile, join

import numpy as np

from prince.util import get_AZN, get_interp_object, info, load_or_convert_array
from prince_config import config


class PhotoNuclearInteractionRate(object):
    def __init__(self,
                 photon_field,
                 cross_section,
                 ebins_dec=10,
                 cbins_dec=10,
                 *args,
                 **kwargs):
        from prince.util import EnergyGrid, get_y
        self.photon_field = photon_field  # object of CombinedPhotonFiled
        self.cross_section = cross_section

        self.e_photon = EnergyGrid(-15, -8, ebins_dec)
        self.e_cosmicray = EnergyGrid(7, 13, cbins_dec)

        x, y = np.meshgrid(self.e_photon.grid, self.e_cosmicray.grid)
        self.matrix = {}
        for proj_id in cross_section.resp_nonel:
            self.matrix[proj_id] = self.cross_section.resp_nonel[proj_id](
                get_y(x, y, proj_id))

    def get_interation_rate(self, proj_id, z):
        # proj_id = PDG & neucosma_codes
        # http://pdg.lbl.gov/2010/reviews/rpp2010-rev-monte-carlo-numbering.pdf

        # Nuclei CORSIKA_ID A*100 + Z, 54Fe  = 5426

        # Requirements: vectorized in E (optional in z) or result output directly on egrid
        from prince.util import get_y

        #x,y = np.meshgrid(self.e_photon.grid, E)

        #M = self.cross_section.resp_nonel[proj_id](get_y(x, y, proj_id))

        photon_vector = self.photon_field.get_photon_density(
            self.e_photon.grid, z)
        return self.matrix[proj_id].dot(self.e_photon.widths * photon_vector)