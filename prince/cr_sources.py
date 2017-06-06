"""Defines interfaces to cosmic ray source models."""

from prince.cosmology import star_formation_rate
from prince_config import config
from prince.util import info, pru

class CosmicRaySource(object):
    def _normalize_spectrum(self):
        """
        Normalize the spectrum to the local cosmic ray injection rate of 1e44 erg MpC^-3 yr^-1
        """
        from scipy import integrate
        intenergy, _ = integrate.quad(lambda energy: energy * self.injection_spectrum(energy), 1e10, 1e12)
        newnorm = 1e44 * pru.erg2GeV / pru.Mpc2cm**3 / pru.yr2sec

        info(2,"Integrated energy is in total: " + str(intenergy))
        info(4,"Renormalizing the integrated energy to: " + str(newnorm))
        self.norm = newnorm / intenergy # output is supposed to be in GeV * cm**-3 * s**-1

    def evolution(self, z):
        return star_formation_rate(z)

    def injection_rate(self, z):
        """
        return the injection rate on the given energy grid
        """
        return self.evolution(z) * self.injection_grid

class SimpleSource(CosmicRaySource):
    def __init__(self, spectral_index=2., emax=1e12, pid=101):
        from prince.util import EnergyGrid
        self.norm = 1.
        self.e_cosmicray = EnergyGrid(*config["cosmic_ray_grid"])
        
        self.spectral_index = spectral_index
        self.emax = emax
        self.particle_id = pid
        self._normalize_spectrum()

        # compute the injection rate on the fixed energy grid
        egrid = self.e_cosmicray.grid
        self.injection_grid = self.injection_spectrum(egrid)

    def injection_spectrum(self, energy):
        """
        power-law injection spectrum with spectral index and maximal energy cutoff
        """
        from numpy import exp
        result = self.norm * energy**(- self.spectral_index) * exp(- energy / self.emax)
        
        return result



class SpectrumSource(CosmicRaySource):
    def __init__(self, edata, specdata, pid = 101):
        from scipy.interpolate import InterpolatedUnivariateSpline
        from prince.util import EnergyGrid
        self.injection_spline = InterpolatedUnivariateSpline(edata, specdata, ext = 'zeros')

        self.norm = 1.
        self.e_cosmicray = EnergyGrid(*config["cosmic_ray_grid"])
        
        self.particle_id = pid
        self._normalize_spectrum()

        # compute the injection rate on the fixed energy grid
        egrid = self.e_cosmicray.grid
        self.injection_grid = self.injection_spectrum(egrid)

    def injection_spectrum(self, energy):
        """
        power-law injection spectrum with spectral index and maximal energy cutoff
        """
        result = self.norm * self.injection_spline(energy)
        return result