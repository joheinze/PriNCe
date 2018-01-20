"""Defines interfaces to cosmic ray source models."""

from abc import ABCMeta, abstractmethod

import numpy as np

from prince.cosmology import star_formation_rate
from prince.util import info, PRINCE_UNITS, get_AZN
from prince_config import config


class CosmicRaySource(object):
    __metaclass__ = ABCMeta

    def __init__(self, prince_run, ncoid, norm=1., *args, **kwargs):

        self.prince_run = prince_run
        self.cr_grid = prince_run.cr_grid.grid

        self.norm = norm

        self.injection_grid = np.zeros(self.prince_run.dim_states)
        self.inj_spec = prince_run.spec_man.ncoid2sref[ncoid]

        # compute the injection rate on the fixed energy grid
        self.injection_grid[self.inj_spec.lidx():self.inj_spec.uidx(
        )] = self.injection_spectrum(self.cr_grid)

    def _normalize_spectrum(self):
        """
        Normalize the spectrum to the local cosmic ray injection rate of 1e44 erg MpC^-3 yr^-1
        """
        from scipy import integrate
        intenergy, _ = integrate.quad(
            lambda energy: energy * self.injection_spectrum(energy), 1e10,
            1e12)
        newnorm = 1e44 * PRINCE_UNITS.erg2GeV / PRINCE_UNITS.Mpc2cm**3 / PRINCE_UNITS.yr2sec

        info(2, "Integrated energy is in total: " + str(intenergy))
        info(4, "Renormalizing the integrated energy to: " + str(newnorm))
        self.norm *= newnorm / intenergy  # output is supposed to be in GeV * cm**-3 * s**-1

    @abstractmethod
    def injection_spectrum(self, energy):
        """Prototype for derived source class"""

    @abstractmethod
    def evolution(self, z):
        """Prototype for derived source class"""

    def injection_rate(self, z):
        """
        return the injection rate on the given energy grid
        """
        return self.evolution(z) * self.injection_grid

    def injection_rate_single(self,energy,z):
        """
        return the injection rate for a single energy and redshift
        """
        return self.evolution(z) * self.injection_spectrum(energy)

class SimpleSource(CosmicRaySource):
    def __init__(self,
                 prince_run,
                 spectral_index=2.,
                 emax=1e13,
                 m=0.,
                 ncoid=101,
                 *args,
                 **kwargs):

        self.spectral_index = spectral_index
        # Convert maximal energy given per particle in energy per nucleon
        self.inj_spec = prince_run.spec_man.ncoid2sref[ncoid]
        self.emax = emax / self.inj_spec.A

        self.m = m

        CosmicRaySource.__init__(self, prince_run, ncoid=ncoid, *args, **kwargs)

    def injection_spectrum(self, energy):
        """
        power-law injection spectrum with spectral index and maximal energy cutoff
        """
        from numpy import exp
        result = energy**(-self.spectral_index) * exp(-energy / self.emax)

        return result

    def evolution(self, z):
        """Source evolution function

        By providing an m parameter, evolution can be scaled.
        """

        return (1 + z)**self.m * star_formation_rate(z)


class AugerFitSource(CosmicRaySource):
    def __init__(self, prince_run, ncoid, rmax, spectral_index, norm, *args,
                 **kwargs):

        self.inj_spec = prince_run.spec_man.ncoid2sref[ncoid]

        self.emax = rmax * self.inj_spec.Z
        self.spectral_index = spectral_index
        norm = norm

        CosmicRaySource.__init__(
            self, prince_run, ncoid=ncoid, norm=norm, *args, **kwargs)

        # self._normalize_spectrum()

    def evolution(self, z):
        """Source evolution function

        Uniform source distribution.
        """

        return 1.

    def injection_spectrum(self, energy):
        """
        power-law injection spectrum with spectral index and maximal energy cutoff
        """
        from numpy import exp

        e_k = energy
        A = float(self.inj_spec.A)
        result = self.norm * (e_k/1e9)**(-self.spectral_index) * np.where(
            e_k*A < self.emax, 1., exp(1 - (e_k*A) / self.emax))
        
        return result 


class SpectrumSource(CosmicRaySource):
    def __init__(self, edata, specdata, pid=101):
        from scipy.interpolate import InterpolatedUnivariateSpline
        from prince.util import EnergyGrid
        self.injection_spline = InterpolatedUnivariateSpline(
            edata, specdata, ext='zeros')

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
