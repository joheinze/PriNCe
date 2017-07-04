"""The module contains everything to handle cross section interfaces."""

import numpy as np

from prince_config import config, spec_data

# JH: I am still not sure, how this class should look like.
# maybe we do not even need a class and can just use the decay distributions
# as functions on an x-grid, where the management is done bei another class

# def AllDecays(object):
#     def __init__(self):
#         self.data = spec_data

#     def get_decay_matrix(mo, da, mo_energy, da_energy):
#         dbentry = self.data[mo]

#         for branching, daughters in dbentry['branchings']:
#             if da in daughters:
#                 x_matrix = da_energy.outer(1 / mo_energy) # the grid in x values

#                 # pion to neutrino
#                 if mo in [2,3] and da in [13,14]:
#                     return branching * pion_to_numu(x_matrix)
#                 # pion to muon
#                 if mo in [2,3] and da in [7,8,9,10]:
#                     return branching * pion_to_muon(x_matrix)
#                 # muon to electron neutrino
#                 if mo in [7,8,9,10]:
#                     return branching * prob_muon_hel(x_matrix, h) * 
#                 # muon to muon neutrino
#                 if mo in [7,8,9,10]:
#             else:
#                 info(6, 'Daughter {:} not found in decay channel of Mother {:} with branching {:} and products {:}'.format(da, mo, branching, daughters))
#                 continue


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
    cond = np.where( np.logical_and(0 <= x, x < 1 - r) )
    
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
    cond = np.where( np.logical_and(r < x, x <= 1) )

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
    
    return (1 + hel * h) / 2 #this result is only correct for x > r

def muon_to_numu(x, h):
    """
    Energy distribution of a numu_bar from the decay of muon+ with helicity h
    (For muon- decay calc with h = -h due to CP invariance)

    Args:
      x (float): energy fraction transferred to the secondary
      h (float): helicity of the muon

    Returns:
      float: probability density at x
    """
    p1 = np.poly1d([4./3., -3., 0., 5./3.])
    p2 = np.poly1d([-8./3., 3., 0., -1./3.])

    return p1(x) + h * p2(x)

def muon_to_emu(x, h):
    """
    Energy distribution of a numu_bar from the decay of muon+ with helicity h
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
    E0 = m_elec + mass_mo - mass_da
    ye = m_elec / E0
    y_grid = mass_mo / 2 / E0 * x_grid
    
    norm = 1. / 60. * (np.sqrt(1. - ye**2) * (2 - 9 * ye**2 - 8 * ye**4) + 15 * ye**4 * np.log(ye/(1 - np.sqrt(1 - ye**2))))
    
    cond = y_grid < 1 - ye
    yshort = y_grid[cond]

    result = np.zeros(y_grid.shape)
    result[cond] = 1 / norm * yshort**2 * (1 - yshort) * np.sqrt((1 - yshort)**2 - ye**2)

    return result