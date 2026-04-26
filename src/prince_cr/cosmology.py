'''
Contains basic functions to handle standard cosmology
'''

import numpy as np

import prince_cr.config as config

def H(z, H0=config.H_0s):
    """Expansion rate of the universe.

    :math:`H(z) = H_0 \\sqrt{\\Omega_m (1 + z)^3 + \\Omega_\\Lambda}`
    Args:
      z (float): redshift

    Returns:
      float: expansion rate in :math:`s^{-1}`
    """

    return H0 * np.sqrt(config.Omega_m * (1 + z)**3 + config.Omega_Lambda)


def star_formation_rate(z, z_inhom=0.):
    """Returns the star formation rate, per comoving volume, evaluated at the specified redshift.

    Ref:
        A.M. Hopkins and J.F. Beacom, Astrophys. J. 651, 142 (2006) [astro-ph/060146]
        P. Baerwald, S. Huemmer, and W. Winter, Astropart. Phys. 35, 508 (2012) [1107.5583]

    Args:
      z (float): redshift
      z_inhom (float): redshift where the universe becomes inhomogenous, return 0. below this values

    Returns:
      float: star formation rate normalized to 1. at y = 0.
    """

    if z < z_inhom:
        return 0.
    elif z <= 0.97:
        return (1. + z)**3.44
    elif 0.97 < z <= 4.48:
        return 10.**1.09 * (1. + z)**-0.26
    else:
        return 10.**6.66 * (1. + z)**-7.8


def grb_rate(z, z_inhom=0.):
    """Returns the rate of Gamma-Ray Burst, per comoving volume, evaluated at the specified redshift.

    Ref:
        TODO: Add reference

    Args:
      z (float): redshift
      z_inhom (float): redshift where the universe becomes inhomogenous, return 0. below this values

    Returns:
      float: GRB rate normalized to 1. at y = 0.
    """
    return (1 + z)**1.4 * star_formation_rate(z, z_inhom=z_inhom)

def grb_rate_wp(z, z_inhom=0.):
    """Returns the rate of Gamma-Ray Burst, per comoving volume, evaluated at the specified redshift.

    Ref:
        Wanderman, Piran: Mon.Not.Roy.Astron.Soc.406:1944-1958,2010

    Args:
      z (float): redshift
      z_inhom (float): redshift where the universe becomes inhomogenous, return 0. below this values

    Returns:
      float: GRB rate normalized to 1. at y = 0.
    """
    if z < z_inhom:
        return 0.
    elif z <= 3:
        return (1 + z)**2.1
    else:
        return (1 + 3)**(2.1 + 1.4) * (1 + z)**-1.4


def agn_rate(z, z_inhom=0.):
    """Returns the rate of Active Galactic Nuclei, per comoving volume, evaluated at the specified redshift.

    Ref:
        TODO: Add reference

    Args:
      z (float): redshift
      z_inhom (float): redshift where the universe becomes inhomogenous, return 0. below this values

    Returns:
      float: AGN rate normalized to 1. at y = 0.
    """
    if z < z_inhom:
        return 0.
    elif z <= 1.7:
        return (1 + z)**5
    elif 1.7 < z <= 2.7:
        return (1 + 1.7)**5
    else:
        return (1 + 1.7)**5 * 10**(2.7 - z)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    zrange = np.linspace(0, 6, 30)
    sfr = np.vectorize(star_formation_rate)
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot(zrange, sfr(zrange))
    ax.set_xlabel(r'Redshift $z$')
    ax.set_ylabel(r'Star formation rate Mpc$^{-3}$')

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot(zrange, H(zrange, H0=config.H_0s))
    ax.set_xlabel(r'Redshift $z$')
    ax.set_ylabel(r'Expansion rate of the universe (km s$^{-1}$ Mpc$^{-1}$')

    plt.show()
