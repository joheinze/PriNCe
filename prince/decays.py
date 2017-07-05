"""The module contains everything to handle cross section interfaces."""

import numpy as np

from prince_config import config, spec_data

# JH: I am still not sure, how this class should look like.
# maybe we do not even need a class and can just use the decay distributions
# as functions on an x-grid, where the management is done bei another class


def get_particle_channels(mo, mo_energy, da_energy):
    """
    Will loop over all channels of a mother and generate a list of redistributions for all daughters
    """

    dbentry = spec_data[mo]
    x_grid = da_energy.outer(1 / mo_energy)

    for branching, daughters in dbentry['branchings']:
        redist = []
        for da in daughters:
            if da > 99:  # daughter is a nucleus, we have lorentz factor conservation
                res = np.ones(x_grid.shape)
            else:
                res = get_decay_matrix(mo, da, x_grid)
            redist.append((da, branching * res))

    return redist


def get_decay_matrix(mo, da, x_grid):
    """
    Generator function. Will select the correct redistribution for the given channel.

    Args:
      mo (int): index of the mother
      da (int): index of the daughter

      x_grid (float):

    Returns:
      float: redistribution on the grid mo_energy / da_energy
    """

    # pi+ to numu or pi- to nummubar
    if mo in [2, 3] and da in [13, 14]:
        return pion_to_numu(x_grid)

    # pi+ to mu+ or pi- to mu-
    elif mo in [2, 3] and da in [5, 6, 7, 8, 9, 10]:
        if da in [7, 10]:  # (any helicity)
            return pion_to_muon(x_grid)
        elif da in [5, 8]:  # left handed, hel = -1
            return pion_to_muon(x_grid) * prob_muon_hel(-1.)
        elif da in [6, 9]:  # right handed, hel = 1
            return pion_to_muon(x_grid) * prob_muon_hel(1.)
        else:
            raise Exception(
                'This should newer have happened, check if-statements above!')

    # muon to neutrino
    elif mo in [5, 6, 7, 8, 9, 10] and da in [11, 12, 13, 14]:
        muon_hel = {
            5: 1.,
            6: -1.,
            7: 0.5,
            8: 1.,
            9: -1.,
            10: 0.5,
        }
        hel = muon_hel[mo]
        if mo in [5, 6, 7] and da in [11]:  # muon+ to electron neutrino
            return muonplus_to_nue(x_grid, hel)
        elif mo in [5, 6, 7] and da in [14]:  # muon+ to muon anti-neutrino
            return muonplus_to_numubar(x_grid, hel)
        elif mo in [8, 9, 10] and da in [12
                                         ]:  # muon- to electron anti-neutrino
            return muonplus_to_nue(x_grid, -1 * hel)
        elif mo in [8, 9, 10] and da in [13]:  # muon- to muon neutrino
            return muonplus_to_numubar(x_grid, -1 * hel)

    # neutrinos from beta decays
    elif mo > 99 and da == 11:  # beta-
        return beta_decay(x_grid, mo, mo + 1)
    elif mo > 99 and da == 12:  # beta+
        return beta_decay(x_grid, mo, mo - 1)
    else:
        info(
            1,
            'Called with unknown channel {:} to {:}, returning an empty redistribution'.
            format(mo, da))
        # no known channel, return zeros
        return np.zeros(x_grid.shape)


def pion_to_numu(x):
    """
    Energy distribution of a numu from the decay of pi

    Args:
      x (float): energy fraction transferred to the secondary
    Returns:
      float: probability density at x
    """
    res = np.zeros(x.shape)

    r = m_muon**2 / m_pion**2
    cond = np.where(np.logical_and(0 <= x, x < 1 - r))

    res[cond] = 1 / (1 - r)
    return res


def pion_to_muon(x):
    """
    Energy distribution of a muon from the decay of pi

    Args:
      x (float): energy fraction transferred to the secondary
    Returns:
      float: probability density at x
    """
    res = np.zeros(x.shape)

    r = m_muon**2 / m_pion**2
    cond = np.where(np.logical_and(r < x, x <= 1))

    res[cond] = 1 / (1 - r)
    return res


def prob_muon_hel(x, h):
    """
    probability for muon+ from pion+ decay to have helicity h
    the result is only valid for x > r

    Args:
      h (int): helicity +/- 1
    Returns:
      float: probability for this helicity
    """
    r = m_muon**2 / m_pion**2

    #helicity expectation value
    hel = 2 * r / (1 - r) / x - (1 + r) / (1 - r)

    return (1 + hel * h) / 2  #this result is only correct for x > r


def muonplus_to_numubar(x, h):
    """
    Energy distribution of a numu_bar from the decay of muon+ with helicity h
    (For muon- decay calc with h = -h due to CP invariance)

    Args:
      x (float): energy fraction transferred to the secondary
      h (float): helicity of the muon

    Returns:
      float: probability density at x
    """
    p1 = np.poly1d([4. / 3., -3., 0., 5. / 3.])
    p2 = np.poly1d([-8. / 3., 3., 0., -1. / 3.])

    return p1(x) + h * p2(x)


def muonplus_to_nue(x, h):
    """
    Energy distribution of a n from the decay of muon+ with helicity h
    (For muon- decay calc with h = -h due to CP invariance)

    Args:
      x (float): energy fraction transferred to the secondary
      h (float): helicity of the muon

    Returns:
      float: probability density at x
    """
    p1 = np.poly1d([4., -6., 0., 2.])
    p2 = np.poly1d([-8., 18., -12., 2.])

    return p1(x) + h * p2(x)


mass_el = spec_data[11]


def beta_decay(x_grid, mother, daughter):
    """
    Energy distribution of a neutrinos from beta-decays of mother to daughter

    Args:
      x (float): energy fraction transferred to the secondary
      mother (int): id of mother
      daughter (int): id of daughter
    Returns:
      float: probability density at x
    """
    mass_mo = spec_data[mother]['mass']
    mass_da = spec_data[daughter]['mass']

    # this is different for beta+ emission, atleast in NeuCosmA.nco_decay.c, l.126  (JH: really? why?)
    E0 = mass_el + mass_mo - mass_da
    ye = mass_el / E0
    y_grid = mass_mo / 2 / E0 * x_grid

    norm = 1. / 60. * (np.sqrt(1. - ye**2) * (2 - 9 * ye**2 - 8 * ye**4) +
                       15 * ye**4 * np.log(ye / (1 - np.sqrt(1 - ye**2))))

    cond = y_grid < 1 - ye
    yshort = y_grid[cond]

    result = np.zeros(y_grid.shape)
    result[cond] = 1 / norm * yshort**2 * (
        1 - yshort) * np.sqrt((1 - yshort)**2 - ye**2)

    return result