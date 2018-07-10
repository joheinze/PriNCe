"""Defines interfaces to cosmic ray source models."""

from abc import ABCMeta, abstractmethod

import numpy as np

from prince_config import config
from prince.util import info

class CosmicRaySource(object):
    __metaclass__ = ABCMeta

    def __init__(self, prince_run, ncoids = None, params = None, norm=1., m='flat', *args, **kwargs):
        # read out standard information from core class
        self.prince_run = prince_run
        self.cr_grid = prince_run.cr_grid.grid
        self.norm = norm
        self.spec_man = prince_run.spec_man

        self.params = params
        self.ncoids = np.array(ncoids if ncoids is not None else params.keys())
        self.ncoids.sort()
        self.source_evo_m = m
        self._compute_injection_grid()

    def _compute_injection_grid(self):
        self.injection_grid = np.zeros(self.prince_run.dim_states)
        for pid in self.ncoids:
            if pid in self.params:
                params = self.params[pid]
            else:
                params = params
            # info(0, 'Injecting particle {:} with parameters {:}'.format(pid, params))
            inj_spec = self.spec_man.ncoid2sref[pid]
            self.injection_grid[inj_spec.lidx():inj_spec.uidx(
            )] = self.injection_spectrum(pid, self.cr_grid, params)

    def integrated_lum(self, Emin=6e9):
        integrals = np.zeros_like(self.ncoids,dtype=np.float)
        integrals2 = np.zeros_like(self.ncoids,dtype=np.float)

        from scipy.integrate import trapz
        for idx, pid in enumerate(self.ncoids):
            s = self.spec_man.ncoid2sref[pid]
            A = s.A

            egrid = self.cr_grid * A
            injec = self.injection_grid[s.slice] / A

            sl = np.argwhere(egrid > Emin)
            sl = sl.flatten()
            integrals[idx] = trapz(injec[sl], egrid[sl])
            integrals2[idx] = trapz(injec[sl] * egrid[sl], egrid[sl])
        return integrals, integrals2

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
        return self.norm * self.evolution(z) * self.injection_spectrum(pid, energy, params)

    @abstractmethod
    def injection_spectrum(self, pid, energy, params):
        """Prototype for derived source class"""

    def evolution(self, z):
        """Source evolution function

        By providing an m parameter, evolution can be scaled.
        """
        from cosmology import star_formation_rate, grb_rate, agn_rate

        # TODO: Current workarround, as the integrator might try 
        # negative z to estimate the source evolution
        if z < -1.:
            raise Exception('Source evolution not defined for negative z = {:}'.format(z))

        if self.source_evo_m == 'flat':
            return 1.
        elif type(self.source_evo_m) is float:
            return (1 + z)**self.source_evo_m * star_formation_rate(z)
        elif type(self.source_evo_m) is tuple:
            if self.source_evo_m[0] == 'SFR':
                return (1 + z)**self.source_evo_m[1] * star_formation_rate(z)
            elif self.source_evo_m[0] == 'GRB':
                return (1 + z)**self.source_evo_m[1] * grb_rate(z)
            elif self.source_evo_m[0] == 'AGN':
                return (1 + z)**self.source_evo_m[1] * agn_rate(z)
            elif self.source_evo_m[0] == 'simple':
                return (1 + z)**self.source_evo_m[1]
            elif self.source_evo_m[0] == 'simple_flat':
                if z <= 1:
                    return (1 + z)**self.source_evo_m[1]
                else:
                    return (1 + 1)**self.source_evo_m[1]
            elif self.source_evo_m[0] == 'simple_SFR':
                if z <= 1:
                    return (1 + z)**self.source_evo_m[1]
                else:
                    return (1 + 1)**3.6 * (1 + z)**(self.source_evo_m[1] - 3.6)
        else:
            raise Exception('Unknown source evo type: {:}'.format(self.source_evo_m))

class SimpleSource(CosmicRaySource):

    def injection_spectrum(self, pid, energy, params):
        """
        power-law injection spectrum with spectral index and maximal energy cutoff
        """
        spectral_index, emax, relnorm = params
        inj_spec = self.spec_man.ncoid2sref[pid]
        emax = emax / inj_spec.A
        result = relnorm * energy**(-spectral_index) * np.exp(-energy / emax)

        return result

class AugerFitSource(CosmicRaySource):

    def injection_spectrum(self, pid, energy, params):
        """
        power-law injection spectrum with spectral index and maximal energy cutoff
        """
        spectral_index, rcut, relnorm = params
        inj_spec = self.spec_man.ncoid2sref[pid]
        A = float(inj_spec.A)
        emax = rcut * inj_spec.Z
        e_k = A * energy
        result = relnorm * A * (e_k/1e9)**(-spectral_index) * np.where(
            e_k < emax, 1., np.exp(1 - e_k / emax))

        return result

class RigidityFlexSource(CosmicRaySource):

    def injection_spectrum(self, pid, energy, params):
        """
        power-law injection spectrum with spectral index and maximal energy cutoff
        """
        spectral_index, rcut, alpha, relnorm = params
        inj_spec = self.spec_man.ncoid2sref[pid]
        A = float(inj_spec.A)
        emax = rcut * inj_spec.Z**alpha
        e_k = A * energy
        result = relnorm * A * (e_k/1e9)**(-spectral_index) * np.where(
            e_k < emax, 1., np.exp(1 - e_k / emax))

        return result


class RigdityCutoffSource(CosmicRaySource):

    def injection_spectrum(self, pid, energy, params):
        """
        power-law injection spectrum with spectral index and maximal energy cutoff
        """
        spectral_index, rcut, relnorm = params
        inj_spec = self.spec_man.ncoid2sref[pid]
        A = float(inj_spec.A)
        emax = rcut * inj_spec.Z
        e_k = A * energy
        result = relnorm * A * (e_k/1e9)**(-spectral_index) * np.exp(1 - e_k / emax)

        return result

class SpectrumSource(CosmicRaySource):

    def injection_spectrum(self, pid, energy, params):
        egrid, specgrid = params
        inj_spec = self.spec_man.ncoid2sref[pid]
        egrid = egrid / inj_spec.A
        specgrid = specgrid * inj_spec.A
        result = np.interp(energy,egrid,specgrid,left=0.,right=0.)

        return result

