"""The module contains everything to handle cross section interfaces."""

import numpy as np
from scipy.integrate import trapz

from prince_cr.util import info, get_AZN
from prince_cr.data import spec_data

def get_particle_channels(mo, mo_energy, da_energy):
    """
    Loops over a all daughers for a given mother and generates
    a list of redistribution matrices on the grid:
     np.outer( da_energy , 1 / mo_energy )
    
    Args:
      mo (int): id of the mother particle
      mo_energy (float): energy grid of the mother particle
      da_energy (float): energy grid of the daughter particle (same for all daughters)
    Returns:
      list of np.array: list of redistribution functions on on xgrid 
    """
    info(10, 'Generating decay redistribution for', mo)
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


def get_decay_matrix(mo, da, x_grid):
    """
    Selects the correct redistribution for the given decay channel.
    If the channel is unknown a zero grid is returned instead of raising an error

    Args:
      mo (int): index of the mother
      da (int): index of the daughter
      x_grid (float): grid in x = E_da / E_mo on which to return the result
                      (If x is a 2D matrix only the last column is computed
                      and then repeated over the matrix assuming that the 
                      main diagonal is always x = 1)
    Returns:
      float: redistribution on the grid mo_energy / da_energy
    """

    info(10, 'Generating decay redistribution for', mo, da)

    # --------------------------------
    # pi+ to numu or pi- to nummubar
    # --------------------------------
    if mo in [2, 3] and da in [13, 14]:
        return pion_to_numu(x_grid)

    # --------------------------------
    # pi+ to mu+ or pi- to mu-
    # --------------------------------
    elif mo in [2, 3] and da in [5, 6, 7, 8, 9, 10]:
        # (any helicity)
        if da in [7, 10]:
            return pion_to_muon(x_grid)
        # left handed, hel = -1
        elif da in [5, 8]:
            return pion_to_muon(x_grid) * prob_muon_hel(x_grid, -1.)
        # right handed, hel = 1
        elif da in [6, 9]:
            return pion_to_muon(x_grid) * prob_muon_hel(x_grid, 1.)
        else:
            raise Exception(
                'This should newer have happened, check if-statements above!')

    # --------------------------------
    # muon to neutrino
    # --------------------------------
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
            return muonplus_to_nue(x_grid, hel)
        # muon+ to muon anti-neutrino
        elif mo in [5, 6, 7] and da in [14]:
            return muonplus_to_numubar(x_grid, hel)
        # muon- to elec anti-neutrino
        elif mo in [8, 9, 10] and da in [12]:
            return muonplus_to_nue(x_grid, -1 * hel)
        # muon- to muon neutrino
        elif mo in [8, 9, 10] and da in [13]:
            return muonplus_to_numubar(x_grid, -1 * hel)

    # --------------------------------
    # neutrinos from beta decays
    # --------------------------------

    # beta-
    elif mo > 99 and da == 11:
        info(10, 'nu_e from beta- decay', mo, mo - 1, da)
        return nu_from_beta_decay(x_grid, mo, mo - 1)
    # beta+
    elif mo > 99 and da == 12:
        info(10, 'nubar_e from beta+ decay', mo, mo + 1, da)
        return nu_from_beta_decay(x_grid, mo, mo + 1)
    # neutron
    elif mo > 99 and 99 < da < 200:
        info(10, 'beta decay boost conservation', mo, da)
        return boost_conservation(x_grid)
    else:
        info(
            5,
            'Called with unknown channel {:} to {:}, returning an empty redistribution'.
            format(mo, da))
        # no known channel, return zeros
        return np.zeros(x_grid.shape)


def get_decay_matrix_bin_average(mo, da, x_lower, x_upper):
    """
    Selects the correct redistribution for the given decay channel.
    If the channel is unknown a zero grid is returned instead of raising an error

    Args:
      mo (int): index of the mother
      da (int): index of the daughter
      x_grid (float): grid in x = E_da / E_mo on which to return the result

    Returns:
      float: redistribution on the grid mo_energy / da_energy
    """

    # TODO: Some of the distribution are not averaged yet.
    # The error is small for smooth distributions though
    info(10, 'Generating decay redistribution for', mo, da)

    x_grid = (x_upper + x_lower) / 2

    # remember shape, but only calculate for last column, as x repeats in each column
    from scipy.integrate import trapz
    shape = x_grid.shape

    if len(shape) == 2:
        x_grid = x_grid[:, -1]
        x_upper = x_upper[:, -1]
        x_lower = x_lower[:, -1]

    # --------------------------------
    # pi+ to numu or pi- to nummubar
    # --------------------------------
    if mo in [2, 3] and da in [13, 14]:
        result = pion_to_numu_avg(x_lower, x_upper)

    # --------------------------------
    # pi+ to mu+ or pi- to mu-
    # --------------------------------
    # TODO: The helicity distr need to be averaged analyticaly
    elif mo in [2, 3] and da in [5, 6, 7, 8, 9, 10]:
        # (any helicity)
        if da in [7, 10]:
            result = pion_to_muon_avg(x_lower, x_upper)
        # left handed, hel = -1
        elif da in [5, 8]:
            result = pion_to_muon_avg(x_lower, x_upper) * prob_muon_hel(
                x_grid, -1.)
        # right handed, hel = 1
        elif da in [6, 9]:
            result = pion_to_muon_avg(x_lower, x_upper) * prob_muon_hel(
                x_grid, 1.)
        else:
            raise Exception(
                'This should newer have happened, check if-statements above!')

    # --------------------------------
    # muon to neutrino
    # --------------------------------
    # TODO: The following distr need to be averaged analyticaly
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
            result = muonplus_to_nue(x_grid, hel)
        # muon+ to muon anti-neutrino
        elif mo in [5, 6, 7] and da in [14]:
            result = muonplus_to_numubar(x_grid, hel)
        # muon- to elec anti-neutrino
        elif mo in [8, 9, 10] and da in [12]:
            result = muonplus_to_nue(x_grid, -1 * hel)
        # muon- to muon neutrino
        elif mo in [8, 9, 10] and da in [13]:
            result = muonplus_to_numubar(x_grid, -1 * hel)

    # --------------------------------
    # neutrinos from beta decays
    # --------------------------------
    # TODO: The following beta decay to neutrino distr need to be averaged analyticaly
    # TODO: Also the angular averaging is done numerically still
    # beta-
    elif mo > 99 and da == 11:
        info(10, 'nu_e from beta+ decay', mo, mo - 1, da)
        result = nu_from_beta_decay(x_grid, mo, mo - 1)
    # beta+
    elif mo > 99 and da == 12:
        info(10, 'nubar_e from beta- decay', mo, mo + 1, da)
        result = nu_from_beta_decay(x_grid, mo, mo + 1)
    # neutron
    elif mo > 99 and 99 < da < 200:
        info(10, 'beta decay boost conservation', mo, da)
        result = boost_conservation_avg(x_lower, x_upper)
    else:
        info(
            5,
            'Called with unknown channel {:} to {:}, returning an empty redistribution'.
            format(mo, da))
        # no known channel, return zeros
        result = np.zeros(x_grid.shape)

    # now fill this into diagonals of matrix
    if len(shape) == 2:
        #'filling matrix'
        res_mat = np.zeros(shape)
        for idx, val in enumerate(result[::-1]):
            np.fill_diagonal(res_mat[:, idx:], val)
        result = res_mat

    return result


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
    xmin = 0.
    xmax = 1 - r

    cond = np.where(np.logical_and(xmin < x, x <= xmax))
    res[cond] = 1 / (1 - r)
    return res


def pion_to_numu_avg(x_lower, x_upper):
    """
    Energy distribution of a numu from the decay of pi

    Args:
      x_lower,x_lower (float): energy fraction transferred to the secondary, lower/upper bin edge
    Returns:
      float: average probability density in bins (xmin,xmax)
    """
    if x_lower.shape != x_upper.shape:
        raise Exception('different grids for xmin, xmax provided')

    bins_width = x_upper - x_lower
    res = np.zeros(x_lower.shape)

    m_muon = spec_data[7]['mass']
    m_pion = spec_data[2]['mass']
    r = m_muon**2 / m_pion**2
    xmin = 0.
    xmax = 1 - r

    # lower bin edged not contained
    cond = np.where(np.logical_and(xmin > x_lower, xmin < x_upper))
    res[cond] = 1 / (1 - r) * (x_upper[cond] - xmin) / bins_width[cond]

    # upper bin edge not contained
    cond = np.where(np.logical_and(x_lower < xmax, x_upper > xmax))
    res[cond] = 1 / (1 - r) * (xmax - x_lower[cond]) / bins_width[cond]

    # bins fully contained
    cond = np.where(np.logical_and(xmin <= x_lower, x_upper <= xmax))
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
    xmin = r
    xmax = 1.

    cond = np.where(np.logical_and(xmin < x, x <= xmax))
    res[cond] = 1 / (1 - r)
    return res


def pion_to_muon_avg(x_lower, x_upper):
    """
    Energy distribution of a numu from the decay of pi

    Args:
      x_lower,x_lower (float): energy fraction transferred to the secondary, lower/upper bin edge
    Returns:
      float: average probability density in bins (xmin,xmax)
    """
    if x_lower.shape != x_upper.shape:
        raise Exception('different grids for xmin, xmax provided')

    bins_width = x_upper - x_lower
    res = np.zeros(x_lower.shape)

    m_muon = spec_data[7]['mass']
    m_pion = spec_data[2]['mass']
    r = m_muon**2 / m_pion**2
    xmin = r
    xmax = 1.

    # lower bin edged not contained
    cond = np.where(np.logical_and(xmin > x_lower, xmin < x_upper))
    res[cond] = 1 / (1 - r) * (x_upper[cond] - xmin) / bins_width[cond]

    # upper bin edge not contained
    cond = np.where(np.logical_and(x_lower < xmax, x_upper > xmax))
    res[cond] = 1 / (1 - r) * (xmax - x_lower[cond]) / bins_width[cond]

    # bins fully contained
    cond = np.where(np.logical_and(xmin <= x_lower, x_upper <= xmax))
    res[cond] = 1 / (1 - r)

    return res


def prob_muon_hel(x, h):
    """
    Probability for muon+ from pion+ decay to have helicity h
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
    cond = np.where(np.logical_and(x > r, x <= 1))
    res[cond] = (1 + hel * h) / 2  #this result is only correct for x > r
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


def boost_conservation(x):
    """Returns an x=1 distribution for ejected nucleons"""
    dist = np.zeros_like(x)
    # dist[(x == np.max(x)) & (x > 0.9)] = 1.*20.
    dist[x == 1.] = 1. / 0.115
    return dist


def boost_conservation_avg(x_lower, x_upper):
    """Returns an x=1 distribution for ejected nucleons"""
    dist = np.zeros_like(x_lower)

    # boost conservation is a delta peak at x = 1
    # if is is contained in the bin, the the value
    # to 1 / width, else set it to zero
    cond = np.where(np.logical_and(x_lower < 1., x_upper > 1.))
    bins_width = x_upper[cond] - x_lower[cond]
    dist[cond] = 1. / bins_width
    return dist


def nu_from_beta_decay(x_grid, mother, daughter, Gamma=200, angle=None):
    """
    Energy distribution of a neutrinos from beta-decays of mother to daughter
    The res frame distrution is boosted to the observers frame and then angular averaging is done numerically

    Args:
      x_grid (float): energy fraction transferred to the secondary
      mother (int): id of mother
      daughter (int): id of daughter
      Gamma (float): Lorentz factor of the parent particle, default: 200
                     For large Gamma this should not play a role, as the decay is scale invariant
      angle (float): collision angle, if None this will be averaged over 2 pi
    Returns:
      float: probability density on x_grid
    """
    import warnings

    info(10, 'Calculating neutrino energy from beta decay', mother, daughter)

    mass_el = spec_data[20]['mass']
    mass_mo = spec_data[mother]['mass']
    mass_da = spec_data[daughter]['mass']

    Z_mo = spec_data[mother]['charge']
    Z_da = spec_data[daughter]['charge']

    A_mo, _, _ = get_AZN(mother)

    if mother == 100 and daughter == 101:
        # for this channel the masses are already nucleon masses
        qval = mass_mo - mass_da - mass_el
    elif Z_da == Z_mo - 1:  # beta+ decay
        qval = mass_mo - mass_da - 2 * mass_el
    elif Z_da == Z_mo + 1:  # beta- decay
        qval = mass_mo - mass_da
    else:
        raise Exception('Not an allowed beta decay channel: {:} -> {:}'.format(
            mother, daughter))

    # substitute this to the energy grid
    E0 = qval + mass_el
    # NOTE: we subsitute into energy per nucleon here
    Emo = Gamma * mass_mo / A_mo
    E = x_grid * Emo

    # print '------','beta decay','------'
    # print mother
    # print E0
    # print A_mo
    # print Emo

    if angle is None:
        # ctheta = np.linspace(-1, 1, 1000)
        # we use here logspace, as high resolution is mainly needed at small energies
        # otherwise the solution will oscillate at low energy
        ctheta = np.unique(np.concatenate((
                    np.logspace(-8,0,1000) - 1,
                    1 - np.logspace(0,-8,1000),
                )))
    else:
        ctheta = angle

    boost = Gamma * (1 - ctheta)
    Emax = E0 * boost

    E_mesh, boost_mesh = np.meshgrid(E, boost, indexing='ij')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = E_mesh**2 / boost_mesh**5 * (Emax - E_mesh) * np.sqrt(
            (E_mesh - Emax)**2 - boost_mesh**2 * mass_el**2)
    res[E_mesh > Emax] = 0.
    res = np.nan_to_num(res)

    if np.all(res == 0):
        info(10, 'Differential distribution is all zeros for', mother, daughter, 
             'No angle averaging performed!')
    elif angle is None:
        # now average over angle
        res = trapz(res, x=ctheta, axis=1)
        res = res / trapz(res, x=x_grid)
    else:
        res = res[:, 0]
        res = res / trapz(res, x=x_grid)

    return res


def nu_from_beta_decay_old(x_grid, mother, daughter):
    """
    Energy distribution of a neutrinos from beta-decays of mother to daughter

    Args:
      x (float): energy fraction transferred to the secondary
      mother (int): id of mother
      daughter (int): id of daughter
    Returns:
      float: probability density at x
    """

    info(10, 'Calculating neutrino energy from beta decay', mother, daughter)

    mass_el = spec_data[20]['mass']
    mass_mo = spec_data[mother]['mass']
    mass_da = spec_data[daughter]['mass']

    Z_mo = spec_data[mother]['charge']
    Z_da = spec_data[daughter]['charge']

    print(mother, daughter)
    if mother == 100 and daughter == 101:
        # for this channel the masses are already nucleon masses
        qval = mass_mo - mass_da - mass_el
    elif Z_da == Z_mo + 1:  # beta+ decay
        qval = mass_mo - mass_da - 2 * mass_el
    elif Z_da == Z_mo - 1:  # beta- decay
        qval = mass_mo - mass_da
    else:
        raise Exception('Not an allowed beta decay channel: {:} -> {:}'.format(
            mother, daughter))

    E0 = qval + mass_el
    print('Qval', qval, 'E0', E0)
    ye = mass_el / E0
    y_grid = x_grid * mass_mo / 2 / E0

    # norm factor, nomalizing the formula to 1
    norm = 1. / 60. * (np.sqrt(1. - ye**2) * (2 - 9 * ye**2 - 8 * ye**4) +
                       15 * ye**4 * np.log(ye / (1 - np.sqrt(1 - ye**2))))

    cond = y_grid < 1 - ye
    print((1 - ye) * 2 * E0 / mass_mo)
    yshort = y_grid[cond]

    result = np.zeros(y_grid.shape)
    # factor for substitution y -> x
    subst = mass_mo / 2 / E0

    # total formula
    result[cond] = subst / norm * yshort**2 * (1 - yshort) * np.sqrt(
        (1 - yshort)**2 - ye**2)

    result[x_grid > 1] *= 0.

    return result
