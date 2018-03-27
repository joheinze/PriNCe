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

    @abstractmethod
    def injection_spectrum(self, energy):
        """Prototype for derived source class"""

    @abstractmethod
    def evolution(self, z):
        """Prototype for derived source class"""

    def get_local_emissivity(self, Emin, Emax):
        """
        Normalize the spectrum to the local cosmic ray injection rate of 1e44 erg MpC^-3 yr^-1
        """
        from scipy import integrate
        A = self.inj_spec.A
        intenergy, _ = integrate.quad(
            lambda energy: energy / A * self.injection_spectrum(energy / A), 
            Emin, Emax)
        return intenergy

    def injection_rate(self, z):
        """
        return the injection rate on the given energy grid
        """
        return self.norm * self.evolution(z) * self.injection_grid

    def injection_rate_single(self,energy,z):
        """
        return the injection rate for a single energy and redshift
        """
        return self.norm * self.evolution(z) * self.injection_spectrum(energy)

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

        CosmicRaySource.__init__(
            self, prince_run, ncoid=ncoid, norm=norm, *args, **kwargs)

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

        A = float(self.inj_spec.A)
        e_k = A * energy
        result = A * (e_k/1e9)**(-self.spectral_index) * np.where(
            e_k < self.emax, 1., exp(1 - (e_k) / self.emax))

        return result

class RigdityCutoffSource(CosmicRaySource):
    def __init__(self, prince_run, ncoid, rmax, spectral_index, norm, *args,
                 **kwargs):

        self.inj_spec = prince_run.spec_man.ncoid2sref[ncoid]

        self.emax = rmax * self.inj_spec.Z
        self.spectral_index = spectral_index
        norm = norm

        CosmicRaySource.__init__(
            self, prince_run, ncoid=ncoid, norm=norm, *args, **kwargs)

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

        A = float(self.inj_spec.A)
        e_k = A * energy
        result = A * self.norm * (e_k/1e9)**(-self.spectral_index) * exp(1 - (e_k) / self.emax)

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
