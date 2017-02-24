'''
Created on Feb 22, 2017

@author: Anatoli Fedynitch
'''
import numpy as np
import cosmology as cosm
from abc import abstractmethod

class PhotonField(object):
    """Base class for constructing target photon densitites.
    
    Derived classes have to implment the method  :func:`PhotonField.get_photon_densities`.
    """
    @abstractmethod
    def get_photon_density(self, E, z):
        raise Exception(self.__class__.__name__ + '::get_photon_density():' +
                        'Base class method called accidentally.')
    
    
class CMBPhotonSpectrum(PhotonField):
    """Redshift-scaled number density of CMB photons
    
    In the CMB frame (equivalent to the observer's frame). Normalisation from Planck's spectrum. 
    The scaling goes as :math:`n(E,z) = (1+z)^3 n(E/(1+z), z = 0)`. 
    The CMB spectrum is a blackbody spectrum with the present-day temperature T0 = 2.725 K.                                    

    Ref.:                                                                                    
        M. Ahlers, L.A. Anchordoqui, and S. Sarkar, Phys. Rev. D 79, 083009 (2009) [0902.3993]    
    """
    
    def get_photon_density(self, E, z):
        """Returns the redshift-scaled number density of CMB photons
    
        Args:
          z (float): redshift
          E (float): photon energy (GeV)
        
        Returns:
          float: CMB photon spectrum in :math:`{\\rm GeV}}^{-1} {\\rm cm}}^{-3}`
        """
                                                
        pref = 1.31868e40  # 1/pi^2/(hbar*c)^3 [GeV^-3 cm^-3]
        Ered = E / (1. + z)
        # density at z = 0, for energy E / (1 + z); ECMB = kB * T0
        nlocal = pref * Ered ** 2 / (np.exp(Ered / cosm.E_CMB) - 1.0) 
        return (1. + z) ** 3 * nlocal

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    erange = np.logspace(-20, -6, 100)
    cmb = CMBPhotonSpectrum()
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.loglog(erange, erange*cmb.get_photon_density(erange, z=0.), ls='-', lw=2, color='k')
#     ax.loglog(erange, erange*CMB_photon_spectrum(erange, z=6.), ls='--', color='k')
    ax.set_ylim(1e-9, 1e3)
#     ax.fill_between(erange, CMB_photon_spectrum(erange, z=0.),
#                     CMB_photon_spectrum(erange, z=6.), color='b', alpha=0.3)
    ax.set_xlabel(r'$\epsilon$ d$n/$d$\epsilon$ cm$^{-3}$')
    ax.set_ylabel(r'Photon energy $\epsilon$ (GeV)')
    
    plt.show()
