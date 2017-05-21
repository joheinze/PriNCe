"""Defines interfaces to cosmic ray source models."""

from prince.cosmology import star_formation_rate

class CosmicRaySource(object):

    def evolution(self, z):
        return star_formation_rate(z)

    def injection_rate(self, z):
        raise Exception('Not implmented!')

class SimpleSource(CosmicRaySource):
    def __init__(self, spectral_index = 2., emax=1e12):
        self.spectral_index = spectral_index
        self.emax = emax
