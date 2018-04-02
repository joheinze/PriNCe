"""Defines interfaces to cosmic ray source models."""

from abc import ABCMeta, abstractmethod

import numpy as np

from prince_config import config


class CosmicRaySource(object):
    __metaclass__ = ABCMeta

    def __init__(self, prince_run, ncoids = None, params = None, norm=1., m=0, *args, **kwargs):
        # read out standard information from core class
        self.prince_run = prince_run
        self.cr_grid = prince_run.cr_grid.grid
        self.norm = norm
        self.spec_man = prince_run.spec_man(self.cr_grid)

        self.ncoids = ncoids
        self.params = params
        self.m = m
        self._compute_injection_grid()

    def _compute_injection_grid(self):
        self.injection_grid = np.zeros(self.prince_run.dim_states)
        for pid in self.ncoids:
            if pid in self.params:
                params = self.params[pid]
            else:
                params = params
            inj_spec = self.spec_man.ncoid2sref[pid]
            self.injection_grid[inj_spec.lidx():inj_spec.uidx(
            )] = self.injection_spectrum(pid, self.cr_grid, params)

    def injection_rate(self, z):
        """
        return the injection rate on the given energy grid
        """
        return self.norm * self.evolution(z) * self.injection_grid

    def injection_rate_single(self,pid,energy,z):
        """
        return the injection rate for a single energy and redshift
        """
        if pid in self.params:
            params = self.params[pid]
        else:
            params = params
        return self.norm * self.evolution(z) * self.injection_spectrum(pid, energy,params)

    @abstractmethod
    def injection_spectrum(self, pid, energy, params):
        """Prototype for derived source class"""


    def evolution(self, z):
        """Source evolution function

        By providing an m parameter, evolution can be scaled.
        """
        from cosmology import star_formation_rate
        return (1 + z)**self.m * star_formation_rate(z)

class SimpleSource(CosmicRaySource):

    def injection_spectrum(self, pid, energy, params):
        """
        power-law injection spectrum with spectral index and maximal energy cutoff
        """
        spectral_index, emax = params
        inj_spec = self.spec_man.ncoid2sref[pid]
        emax = emax / inj_spec.A
        result = energy**(-spectral_index) * np.exp(-energy / emax)

        return result

class AugerFitSource(CosmicRaySource):

    def injection_spectrum(self, pid, energy, params):
        """
        power-law injection spectrum with spectral index and maximal energy cutoff
        """
        spectral_index, rcut = params
        inj_spec = self.spec_man.ncoid2sref[pid]
        A = float(inj_spec.A)
        emax = rcut * inj_spec.Z
        e_k = A * energy
        result = A * (e_k/1e9)**(-spectral_index) * np.where(
            e_k < emax, 1., np.exp(1 - e_k / emax))

        return result

class RigdityCutoffSource(CosmicRaySource):

    def injection_spectrum(self, pid, energy, params):
        """
        power-law injection spectrum with spectral index and maximal energy cutoff
        """
        spectral_index, rcut = params
        inj_spec = self.spec_man.ncoid2sref[pid]
        A = float(inj_spec.A)
        emax = rcut * inj_spec.Z
        e_k = A * energy
        result = A * (e_k/1e9)**(-spectral_index) * np.exp(1 - e_k / emax)

        return result

class SpectrumSource(CosmicRaySource):

    def injection_spectrum(self, pid, energy, params):
        egrid, specgrid = params
        inj_spec = self.spec_man.ncoid2sref[pid]
        egrid = egrid / inj_spec.A
        specgrid = specgrid * inj_spec.A
        result = np.interp(energy,egrid,specgrid,left=0.,right=0.)

        return result

