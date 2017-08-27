"""The module contains everything to handle cross section interfaces."""

import numpy as np

from prince.util import info
from prince_config import config, spec_data

# JH: I am still not sure, how this class should look like.
# maybe we do not even need a class and can just use the decay distributions
# as functions on an x-grid, where the management is done bei another class


def get_particle_channels(mo, mo_energy, da_energy):
    """
    Will loop over all channels of a mother and generate a list of redistributions for all daughters
    """

    dbentry = spec_data[mo]
    x_grid = np.outer(da_energy, (1 / mo_energy))

    redist = {}
    for branching, daughters in dbentry['branchings']:
        for da in daughters:
            # daughter is a nucleus, we have lorentz factor conservation
            if da > 99:
                res = np.zeros(x_grid.shape)
                res[x_grid == 1.] = 1.
            else:
                res = get_decay_matrix(mo, da, x_grid)
            redist[da] = branching * res

    return x_grid, redist


def get_decay_matrix(mo, da, x_grid, x_widths=None):
    """
    Generator function. Will select the correct redistribution for the given channel.

    Args:
      mo (int): index of the mother
      da (int): index of the daughter

      x_grid (float):

    Returns:
      float: redistribution on the grid mo_energy / da_energy
    """

    if x_widths is None:
        x_widths = np.ones_like(x_grid)

    # pi+ to numu or pi- to nummubar
    if mo in [2, 3] and da in [13, 14]:
        return x_widths * pion_to_numu(x_grid)

    # pi+ to mu+ or pi- to mu-
    elif mo in [2, 3] and da in [5, 6, 7, 8, 9, 10]:
        # (any helicity)
        if da in [7, 10]:
            return x_widths * pion_to_muon(x_grid)
        # left handed, hel = -1
        elif da in [5, 8]:
            return x_widths * pion_to_muon(x_grid) * prob_muon_hel(-1.)
        # right handed, hel = 1
        elif da in [6, 9]:
            return x_widths * pion_to_muon(x_grid) * prob_muon_hel(1.)
        else:
            raise Exception(
                'This should newer have happened, check if-statements above!')

    # muon to neutrino
    elif mo in [5, 6, 7, 8, 9, 10] and da in [11, 12, 13, 14]:
        # translating muon ids to helicity
        muon_hel = {
            5: 1.,
            6: -1.,
            7: 0.,
            8: 1.,
            9: -1.,
            10: 0.,
        }
        hel = muon_hel[mo]
        # muon+ to electron neutrino
        if mo in [5, 6, 7] and da in [11]:
            return x_widths * muonplus_to_nue(x_grid, hel)
        # muon+ to muon anti-neutrino
        elif mo in [5, 6, 7] and da in [14]:
            return x_widths * muonplus_to_numubar(x_grid, hel)
        # muon- to elec anti-neutrino
        elif mo in [8, 9, 10] and da in [12]:
            return x_widths * muonplus_to_nue(x_grid, -1 * hel)
        # muon- to muon neutrino
        elif mo in [8, 9, 10] and da in [13]:
            return x_widths * muonplus_to_numubar(x_grid, -1 * hel)

    # neutrinos from beta decays
    # beta-
    elif mo > 99 and da == 11:
        print 'beta- decay', mo, mo - 1
        return x_widths * beta_decay(x_grid, mo, mo - 1)
    # beta+
    elif mo > 99 and da == 12:
        print 'beta+ decay', mo, mo + 1
        return x_widths * beta_decay(x_grid, mo, mo + 1)
    else:
        info(
            1,
            'Called with unknown channel {:} to {:}, returning an empty redistribution'.
            format(mo, da))
        # no known channel, return zeros
        return x_widths * np.zeros(x_grid.shape)


def pion_to_numu(x):
    """
    Energy distribution of a numu from the decay of pi

    Args:
      x (float): energy fraction transferred to the secondary
    Returns:
      float: probability density at x
    """
    res = np.zeros(x.shape)

    m_muon = spec_data[7]['mass']
    m_pion = spec_data[2]['mass']

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

    m_muon = spec_data[7]['mass']
    m_pion = spec_data[2]['mass']

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

    m_muon = spec_data[7]['mass']
    m_pion = spec_data[2]['mass']

    r = m_muon**2 / m_pion**2

    #helicity expectation value
    hel = 2 * r / (1 - r) / x - (1 + r) / (1 - r)

    res = np.zeros(x.shape)
    res[x > r] = (1 + hel * h) / 2  #this result is only correct for x > r
    return res


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

    res = np.zeros(x.shape)
    cond = x <= 1.
    res[cond] = p1(x[cond]) + h * p2(x[cond])
    return res


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

    res = np.zeros(x.shape)
    cond = x <= 1.
    res[cond] = p1(x[cond]) + h * p2(x[cond])
    return res


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
    mass_el = spec_data[20]['mass']
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