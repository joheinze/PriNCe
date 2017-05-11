'''
Created on Feb 22, 2017

@author: Anatoli Fedynitch
'''

import prince.photonfields
import prince.intcs
from prince_config import config
import numpy as np

class PriNCeRun(object):
    def __init__(self, *args, **kwargs):


        

        self.photon_field = prince.photonfields.CombinedPhotonField(
            [prince.photonfields.CMBPhotonSpectrum,
             prince.photonfields.CIBFranceschini2D]
        )

        self.int_rates = prince.intcs.CrossSectionCombined(
            [prince.intcs.CrossSectionPeanut, 
            prince.intcs.CrossSectionSophia],
            1.
        )

        