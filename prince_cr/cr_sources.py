"""Defines source models for cosmic ray propagation

    The standard interface requires to UHECRSolvers requires
    that each source defines a methods injection_rate(self, z)
"""

from abc import ABCMeta, abstractmethod

import numpy as np

from prince_cr.util import info


class CosmicRaySource(object, metaclass=ABCMeta):
    def __init__(self,
                 prince_run,
                 ncoids=None,
                 params=None,
                 norm=1.,
                 m='flat',
                 *args,
                 **kwargs):
        # read out standard information from core class
        self.cr_grid = prince_run.cr_grid.grid
        self.dim_states = prince_run.dim_states
        self.norm = norm
        self.spec_man = prince_run.spec_man

        # read out the input parameters, this is a dictionary with particle ids as key
        # the parameters are interpreted by each child class.
        self.params = params
        self.ncoids = np.array(ncoids if ncoids is not None else list(params.keys()))
        self.ncoids.sort()
        self.source_evo_m = m
        self._compute_injection_grid()

    def _compute_injection_grid(self):
        """Precompute the injection for all species on a single grid.
        
        Assumes that the injection is factorized in E and z"""
        self.injection_grid = np.zeros(self.dim_states)

        for pid in self.ncoids:
            if pid in self.params:
                params = self.params[pid]
            else:
                params = params

            info(
                4, 'Injecting particle {:} with parameters {:}'.format(
                    pid, params))
            inj_spec = self.spec_man.ncoid2sref[pid]
            self.injection_grid[inj_spec.sl] = self.injection_spectrum(
                pid, self.cr_grid, params)

    def integrated_lum(self, Emin=1e9):
        """Integrates the injected spectrum for all particles
        
        Args:
            Emin (float): lower limit of the integration
    
        Returns:
            float array: tuple with integrated number and luminosity for each species 
        """
        num_int = np.zeros_like(self.ncoids, dtype=np.float)
        lum_int = np.zeros_like(self.ncoids, dtype=np.float)

        from scipy.integrate import trapz
        for idx, pid in enumerate(self.ncoids):
            # get the inection for the species and subsitute back from E_A =  E / A to E
            s = self.spec_man.ncoid2sref[pid]
            A = s.A
            egrid = self.cr_grid * A
            injec = self.injection_grid[s.sl] / A

            # Get the ranges where the energy is greater than Emin and integrate
            mask = np.argwhere(egrid > Emin)
            mask = mask.flatten()
            num_int[idx] = trapz(injec[mask], egrid[mask])
            lum_int[idx] = trapz(injec[mask] * egrid[mask], egrid[mask])
        return num_int, lum_int

    def injection_rate(self, z):
        """Returns the injection rate on the given self.cr_grid

        Args:
            z (float): redshift

        Returns:
            float array: array of the same length as self.cr_grid
        """
        return self.norm * self.evolution(z) * self.injection_grid

    def injection_rate_single(self, pid, energy, z):
        """Return the injection rate for a single energy and redshift

        Args:
            pid (int): particle id for which to return the injection
            energy (float): single float or array for which to return the injection
            z (float): redshift

        Return:
            float: injection on the grid defined by energy
        """
        if pid in self.params:
            params = self.params[pid]
        return self.norm * self.evolution(z) * self.injection_spectrum(
            pid, energy, params)

    @abstractmethod
    def injection_spectrum(self, pid, energy, params):
        """Prototype to be defined in each child class"""

    def evolution(self, z):
        """Returns the source evolution function at given redshift

        Note: The source evolution will depend on self.source_evo_m
              This can be a float, a keyword or a tuple of both.
              See code for supported combinations

        Args:
            z (float): redshift

        Return:
            float: Relative source evolution, typically normalized to 1 at z=0
        """
        from .cosmology import star_formation_rate, grb_rate_wp, agn_rate

        # Check if negative z is called
        if z < 0:
            raise Exception(
                'Source evolution not defined for negative z = {:}'.format(z))

        # flat source evolution
        if self.source_evo_m == 'flat':
            return 1.
        # simple source evolution (1+z)**m
        elif type(self.source_evo_m) is float:
            return (1 + z)**self.source_evo_m * star_formation_rate(z)
        # for a tuple decide based on keyword
        elif type(self.source_evo_m) is tuple:
            mkwd, mval = self.source_evo_m
            # in these cases the evolution is taken from keyword and scaled by m
            if mkwd == 'SFR':
                return (1 + z)**mval * star_formation_rate(z)
            elif mkwd == 'GRB':
                return (1 + z)**mval * grb_rate_wp(z)
            elif mkwd == 'AGN':
                return (1 + z)**mval * agn_rate(z)
            elif mkwd == 'TDE':
                return (1 + z)**(mval-3.)
            elif mkwd == 'simple':
                return (1 + z)**mval
            # local evolution as (1+z)**m and flat beyond z = 1
            elif mkwd == 'simple_flat':
                if z <= 1:
                    return (1 + z)**mval
                else:
                    return (1 + 1)**mval
            # local evolution as (1+z)**m and a break of n = m - 3.6 at z = 1
            elif mkwd == 'simple_SFR':
                if z <= 1:
                    return (1 + z)**mval
                else:
                    return (1 + 1)**3.6 * (1 + z)**(mval - 3.6)
        else:
            raise Exception('Unknown source evo type: {:}'.format(
                self.source_evo_m))


class SimpleSource(CosmicRaySource):
    """Simple source class with spectral index and cutoff

        inj(E) = norm * E**-gamma * exp(- E / Emax)

        params defined as {pid: gamma, Emax, norm}
    """

    def injection_spectrum(self, pid, energy, params):
        gamma, emax, norm = params
        inj_spec = self.spec_man.ncoid2sref[pid]
        emax = emax / inj_spec.A
        result = norm * energy**(-gamma) * np.exp(-energy / emax)

        return result


class RigdityCutoffSource(CosmicRaySource):
    """Simple source class with spectral index and rigidity dependent cutoff

        inj(E) = norm * E**-gamma * exp(- E / Z * Rcut)

        params defined as {pid: gamma, Rcut, norm}
    """

    def injection_spectrum(self, pid, energy, params):
        spectral_index, rcut, relnorm = params
        inj_spec = self.spec_man.ncoid2sref[pid]
        A = float(inj_spec.A)
        emax = rcut * inj_spec.Z
        e_k = A * energy
        result = relnorm * A * (e_k / 1e9)**(
            -spectral_index) * np.exp(1 - e_k / emax)

        return result


class AugerFitSource(CosmicRaySource):
    """Simple source class with spectral index and rigidity dependent cutoff
        Defined to be parrallel for all species below the cutoff as in Auger Combined Fit paper

        if E <  Z * Rcut:
            inj(E) = norm * E**-gamma
        else:
            inj(E) = norm * E**-gamma * exp(- E / Z * Rcut)

        params defined as {pid: gamma, Rcut, norm}
    """

    def injection_spectrum(self, pid, energy, params):
        spectral_index, rcut, relnorm = params
        inj_spec = self.spec_man.ncoid2sref[pid]
        A = float(inj_spec.A)
        emax = rcut * inj_spec.Z
        e_k = A * energy
        result = relnorm * A * (e_k / 1e9)**(-spectral_index) * np.where(
            e_k < emax, 1., np.exp(1 - e_k / emax))

        return result


class RigidityFlexSource(CosmicRaySource):
    """Simple source class with spectral index and rigidity dependent cutoff
        Parameter alpha to scaled the rigidity dependence

        inj(E) = norm * E**-gamma * exp(- E / Z**alpha * Rcut)

        params defined as {pid: gamma, Rcut, alpha, norm}
    """

    def injection_spectrum(self, pid, energy, params):
        spectral_index, rcut, alpha, relnorm = params
        inj_spec = self.spec_man.ncoid2sref[pid]
        A = float(inj_spec.A)
        emax = rcut * inj_spec.Z**alpha
        e_k = A * energy
        result = relnorm * A * (e_k / 1e9)**(-spectral_index) * np.where(
            e_k < emax, 1., np.exp(1 - e_k / emax))

        return result


class SpectrumSource(CosmicRaySource):
    """Source class with the spectrum defined externally by an array
        The spectrum might be interpolated as needed
       
        params defined as {pid: egrid, specgrid}
    """

    def injection_spectrum(self, pid, energy, params):
        egrid, specgrid = params
        inj_spec = self.spec_man.ncoid2sref[pid]
        egrid = egrid / inj_spec.A
        specgrid = specgrid * inj_spec.A
        result = np.interp(energy, egrid, specgrid, left=0., right=0.)

        return result
