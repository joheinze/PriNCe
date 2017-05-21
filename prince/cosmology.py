'''
Created on Feb 22, 2017

@author: Anatoli Fedynitch
'''

from prince_config import config
import numpy as np

H0 = config['H_0s']
Omega_m = config['Omega_m']
Omega_Lambda = config['Omega_Lambda']

E_CMB = 2.34823e-13  # = kB*T0 [GeV]


def H(z, H0=H0):
    """Expansion rate of the universe.

    :math:`H(z) = H_0 \\sqrt{\\Omega_m (1 + z)^3 + \\Omega_\\Lambda}`
    Args:
      z (float): redshift

    Returns:
      float: expansion rate in :math:`s^{-1}`
    """

    return H0 * np.sqrt(Omega_m * (1 + z)**3 + Omega_Lambda)


def star_formation_rate(z, z_inhomogeneous=0.):
    """Returns the star formation rate, per comoving volume, evaluated at the specified redshift.

    Ref:
        A.M. Hopkins and J.F. Beacom, Astrophys. J. 651, 142 (2006) [astro-ph/060146]
        P. Baerwald, S. Huemmer, and W. Winter, Astropart. Phys. 35, 508 (2012) [1107.5583]

    Args:
      z (float): redshift

    Returns:
      float: star formation rate in :math:`{\\rm Mpc}}^{-3}`
    """

    if z < z_inhomogeneous:
        return 0.
    elif z <= 0.97:
        return (1. + z)**3.44
    elif z > 0.97 and z <= 4.48:
        return 10.**1.09 * (1. + z)**-0.26
    else:
        return 10.**6.66 * (1. + z)**-7.8


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    zrange = np.linspace(0, 6, 30)
    sfr = np.vectorize(star_formation_rate)
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot(zrange, sfr(zrange))
    ax.set_xlabel(r'Redshift $z$')
    ax.set_ylabel(r'Star formation rate Mpc$^{-3}$')

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot(zrange, H(zrange, H0=config['H_0']))
    ax.set_xlabel(r'Redshift $z$')
    ax.set_ylabel(r'Expansion rate of the universe (km s$^{-1}$ Mpc$^{-1}$')

    plt.show()
