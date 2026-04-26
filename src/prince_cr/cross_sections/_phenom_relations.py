"""Module implementing relations from paper

Functions implementing formulas from paper 
http://stacks.iop.org/1402-4896/49/i=3/a=004
and others, to construct the inclusive cross
sections.
"""
import itertools
import sys
from collections import Counter
from os.path import join
from pickle import load

from numpy import array, exp, inf, linspace, sum

from prince_cr.util import get_AZN
from prince_cr.data import spec_data
from prince_cr.config import config

# JH: This file gave some linter errors for me, disabled for now
# pylint: skip-file

# class mydict(dict):
#     def __getitem__(self, key):
#         if key not in self:
#             self.__setitem__(key, 0)
            
#         return dict.__getitem__(self, key)

# listing functions to create species tables
def list_species_by_mass(Amax, tau=inf):
    '''Returns a dictionary with the species stable enough
    to be produced in spallation.
    '''
    species = {}
    for nuc in sorted([k for k in spec_data.keys() if isinstance(k, int)]):
        if (nuc < 100) or (spec_data[nuc]['lifetime'] < tau):
            continue
        At, _, _ = get_AZN(nuc)

        if At in species:
            species[At].append(nuc)
        else:
            species[At] = [nuc]

    return species


def partitions(n):
    """Returns all integer combinations that add up to n
    and no summand is higher than 4
    """	
    # base case of recursion: zero is the sum of the empty list
    if n == 0:
        yield []
        return
        
    # modify partitions of n-1 to form partitions of n
    for p in partitions(n-1):
        yield [1] + p
        if p and (len(p) < 2 or p[1] > p[0]):
            if p[0] < 4:
                yield [p[0] + 1] + p[1:]


def combinations(x, y):
    '''Returns possible combinations of nuclei with mass A<=4
    such that they contain x protons and y neutrons.
    '''
    ncos_id = int((x + y)*100 + x)
    mass_partitions = partitions(x + y)

    for mass_partition in mass_partitions:
        mass_partition = [f for f in mass_partition if f > 1]
        species = [species_by_mass[Af] for Af in mass_partition if Af in species_by_mass]
        for c in list(itertools.product(*species)):
            _, z, n = get_AZN(sum(c))
            if (z <= x) and (n <= y):
                yield int(y-n)*(100, ) + int(x-z)*(101, ) + c


def residual_multiplicities():
    '''Makes a dictionary where the ncos id with x,y number of protons and 
    neutrons emitted are the keys, and the multiplicities of species between A=1-4
    are given, normalized such that they add up to x protons and y neutrons.
    This is a function to precompute the table which is used to find the multiplicities
    of the empirical photomeson model.
    '''
    spalled_nucleons = []
    for Am in species_by_mass:
        for mother in species_by_mass[Am]:
            for A_big_frag in range(Am/2, Am-1):
                for big_frag in species_by_mass[A_big_frag]:
                    if big_frag % 100 > mother % 100:
                        continue
                    spalled_frag = mother - big_frag
                    spalled_nucleons.append(spalled_frag)
                    # print spalled_nucleons[-1]
    residual_list = {}
    count = 0
    last = 0
    print('Completed.... ', 0)
    cant = float(len(set(spalled_nucleons)))
    for tot in set(spalled_nucleons):
        count+=1
        # print '--', tot, '--', '{:3.3f}'.format(count/cant)
        if int(100 * count / cant) >= last + 5:
            print('Completed.... ', int(100 * count / cant))
            last += 5
        _, x, y = get_AZN(tot)
        counts = Counter([e for elem in combinations(x, y) for e in elem])
        suma = 0
        for k, v in counts.items():
            suma += k * v
        for k in counts:
            counts[k] *= tot/float(suma)
        residual_list[tot] = counts.copy()

    return residual_list

# local lookup tables for efficiency
# species_by_mass = list_species_by_mass(56, config['tau_dec_threshold'])
species_by_mass = list_species_by_mass(56, 0.)
# resmul = residual_multiplicities()
small_frags_relative_yields_filename = join(config.data_dir, 'small_frags_relative_yields.pkl')
with open(small_frags_relative_yields_filename, 'rb') as f:
    # it's faster to pickle.load a precomputed resmul
    resmul = load(f,encoding='latin1')


#### empirical relations from reference...
def cs_gpi(A):
    """Cross section for pion photoproduction averaged over 
    E[.14, 1.] GeV
    
    Average cross section values obtained by Monte Carlo
    simulations, for nuclei of different masses, starting at
    7Li. It includes the production of pions of all types.
    
    Arguments:
        A {int} -- Number of nucleons in the target nucleus
    
    Returns:
        float -- Mean cross section in miliibarn
    """
    return 0.027 * A**.847


def cs_gn(A):
    """Cross section for A(g,n)X averaged over E[.3, 1.] GeV
    
    Returns cross section of photoneutron production averaged
    over the energy range [.3, 1.] GeV, in milibarn units.
    
    Arguments:
        A {int} -- Nucleon number of the target nucleus
    """

    return 0.104 * A**0.81


def xm(A):
    """Returns the maximum number of emmited neutrons in a 
    A(g,xn)X reaction
    
    See function cs_gxn()
    
    Arguments:
        A {int} -- 
    
    Returns:
        int -- Maximum number of neutrons to be produced
    """
    return int(1.4 * A**.457)


def cs_gxn(A, x=2):
    """Cross section for A(g,xn)X averaged over E[.2, 1.] GeV
    
    Returns cross section of photoneutrons production (x > 1)
    averaged over the energy range [.3, 1.] GeV, in milibarn 
    units. The number of neutrons x is checked to be lower than
    xm (see function definition xm).
    
    Arguments:
        A {int} -- Nucleon number of the target nucleus
        x {int} -- Number of neutrons produced (1 < x < xm)
    """

    if 1 < x < xm(A):
        k = 37. * A ** -.924
        return 0.187 * A**0.684 * exp(-k * (x - 1)**1.25)
    else:
        return 0


def cs_gxn_all(A):
    """Cross section for A(g,xn)X averaged over E[.2, 1.] GeV
    
    Returns cross section of photoneutrons production (x > 1)
    averaged over the energy range [.3, 1.] GeV, in milibarn 
    units. The number of neutrons x is checked to be lower than
    xm (see function definition xm).
    
    Arguments:
        A {int} -- Nucleon number of the target nucleus
        x {int} -- Number of neutrons produced (1 < x < xm)
    """
    cs_gxn_summed = 0
    for xi in range(2, xm(A)):
        cs_gxn_summed += cs_gxn(A, xi)

    return cs_gxn_summed


def cs_gp(Z=1, **kwargs):
    """Cross section for A(g,p)X averaged over E[.3, .8] GeV
    
    Returns cross section of photoproduction of a proton averaged
    over the energy range [.3, .8] GeV, in milibarn units.
    
    Arguments:
        Z {int} -- Number of protons of the target nucleus
        A {int} -- (optional) 
    """
    if 'A' in kwargs:
        return 0.078 * kwargs['A']**0.5
    else:
        return 0.115 * Z**0.5


def cs_gSp(Z, A, x=1, y=1):
    """Cross section for spallation averaged over E[.2, 1.] GeV
    
    Returns cross section of photoproduction of multiple
    protons and neutrons in a spallation procees, averaged
    over the energy range [.2, 1.] GeV, in milibarn units.
    
    Arguments:
        Z {int} -- Number of protons of the target nucleus
        A {int} -- Number of nucleons of the target nucleus
                   A <= 90
        x {int} -- Number of protons produced. Default is 1.
                   1 <= x <= Z/2
        y {int} -- Number of neutron produced. Default is 1.
                   1 <= y <= (A - Z)/2

    """
    K = 0.466
    a = float(Z) / (A - Z)
    C = 2.3 * a - 1.044
    E = 446. / A
    if E < 21.:
        cs_M = 15.7 / E**1.356
    elif 21. < E:
        cs_M = 0.248
    if E < 10.:
        B = 3.03 / E**1.06
    elif E > 10.:
        B = 0.25

    return cs_M * exp(-B * (x - 1) - K*(x - C*a*y)**2)


def cs_gSp_all(Z, A):
    """Cross section summed for all possible spallation events
    """	
    mother = 100*A + Z
    cs_tot = 0
    for A_big_frag in range(A/2, A-1):
        for big_frag in species_by_mass[A_big_frag]:
            _, x, y = get_AZN(mother - big_frag)
            spalled_id = 100*(x+y) + x
            
            if (x < 1) or (y < 1):
                # in spallation at least a neutron and proton escape
                continue

            cs_frag = cs_gSp(Z, A, x, y)
            cs_tot += cs_frag

    return cs_tot


def cs_gSp_all_inA(A):
    """Cross section summed for all possible spallation events
    """
    n = 0
    cs_summed = 0
    cs_vals = []
    if A in species_by_mass:
        for nuc in species_by_mass[A]:
            A, Z, N = get_AZN(nuc)        
            cs_summed = cs_gSp_all(Z, A)
            cs_vals.append(cs_gSp_all(Z, A))
            n += 1
    
    # return cs_summed / n
    return max(cs_vals)


def cs_tot(A):
    '''Determines the norm for the empirical formulas
    such that the addition of inclusive cross sections
    does not fluctuate with A. Renormalization value 
    depends on the mass and is set to the mean of the
    total empirical cross section per nucleon over 
    a range of A in A=4-55.
    '''
    csp = cs_gp(A=A)
    cspi = cs_gpi(A)
    csn = cs_gn(A)
    csxn = cs_gxn_all(A)
    csSpal = cs_gSp_all_inA(A)
    
    cstot = csp + cspi + csn + csxn + csSpal

    return cstot * cs_norm[A]


#### inclusive cross sections derived from relations above, and related functions

def partition_probability(partition, A, beta=.1):
    """Gives partitions probabilities as exp(-beta*Ei)
    where Ei is the sum of binding energies of the components of the partition.
    
    Given a partition (list of mass fragments {A_j}), the sets of all possible
    species per mass {S^A_k} are taken. Then all possible combinations corresponding
    to the partition are built C_l={S^j_m} and the combination is given a probability
    based on it's energy: P_l = exp(-beta*E_l). The probablities of all combinations
    are normalized to unity (P_l = P_l / sum(P_l))
    
    Arguments:
        partition {[list]} -- list of [A_k] masses into which the original nucleus was split

    Returns:
        combinations -- a list of possible nuclei that make up the A partition given
        yields -- the probabilities of each of the partitions, obtained by a statistical
            argument (i.e. canonical ensemble )
    """

    species = [species_by_mass[Af] for Af in partition if Af in species_by_mass]
    combinations = list(itertools.product(*species))

    yields = []
    for combination in combinations:
        Ei = A
        for nuc in combination:
            Ei -= spec_data[nuc]['mass']
        
        yields.append(exp(-beta*Ei))

    tot_yields = sum(yields)
    yields = [y/tot_yields for y in yields]
    
    return combinations, yields


def partition_probability(combinations, A, beta=.1):
    """Gives partitions probabilities as exp(-beta*Ei)
    where Ei is the sum of binding energies of the components of the partition.
    
    Given a partition (list of mass fragments {A_j}), the sets of all possible
    species per mass {S^A_k} are taken. Then all possible combinations corresponding
    to the partition are built C_l={S^j_m} and the combination is given a probability
    based on it's energy: P_l = exp(-beta*E_l). The probablities of all combinations
    are normalized to unity (P_l = P_l / sum(P_l))
    
    Arguments:
        partition {[list]} -- list of [A_k] masses into which the original nucleus was split

    Returns:
        combinations -- a list of possible nuclei that make up the A partition given
        yields -- the probabilities of each of the partitions, obtained by a statistical
            argument (i.e. canonical ensemble )
    """

    yields = []
    for combination in combinations:
        Ei = A
        for nuc in combination:
            Ei -= spec_data[nuc]['mass']
        
        yields.append(exp(-beta*Ei))

    tot_yields = sum(yields)
    yields = [y/tot_yields for y in yields]
    
    return combinations, yields


def gxn_multiplicities(mother):
    """Multiplicities for multineutron emission A(g,xn)X
    
    Arguments:
        A {int} -- Nucleon number of the target nucleus
    """
    cs_sum = 0
    cs_gxn_incl = {100:0}
    Am, _, _ = get_AZN(mother)

    for xi in range(2, xm(Am)):
        cs = cs_gxn(Am, xi)
        cs_gxn_incl[100] += xi*cs
        cs_gxn_incl[mother - xi*100] = cs

        cs_sum += cs

    if cs_sum == 0:
        cs_sum = inf

    for dau in cs_gxn_incl:
        if cs_gxn_incl[dau] != 0:
            cs_gxn_incl[dau] /= cs_sum
    
    return cs_gxn_incl


def spallation_multiplicities(mother):
    '''Calculates the inclusive cross sections of all fragments
    of mothter species (moter is a neucos id) for spallation
    '''
    Am, Zm, _ = get_AZN(mother)
    
    incl_tab = {}
    cs_sum = 0
    for A_big_frag in range(Am//2, Am-1):
        for big_frag in species_by_mass[A_big_frag]:
            _, x, y = get_AZN(mother - big_frag)
            spalled_id = 100*(x+y) + x
            
            if (x < 1) or (y < 1):
                # in spallation at least a neutron and proton escape
                continue
            
            cs_frag = cs_gSp(Zm, Am, x, y)
            cs_sum += cs_frag  # sum of all cross sections to normalize incl_tab
            
            if big_frag in incl_tab:
                incl_tab[big_frag] += cs_frag
            else:
                incl_tab[big_frag] = cs_frag

            # get low fragment incl_tab from using Counter on a prepared list with x, y outputs
            for dau in resmul[spalled_id]:
                if dau in incl_tab:
                    incl_tab[dau] += cs_frag * resmul[spalled_id][dau]
                else:
                    incl_tab[dau] = cs_frag * resmul[spalled_id][dau]
    for dau in incl_tab:
        incl_tab[dau] /= cs_sum  # all spallation cross section should match total spallation cross section

    return incl_tab


def cs_Rincl(Z, A, yields):
    """Returns incl of residual production
    
    [description]
    
    Arguments:
        Z {[type]} -- [description]
        A {[type]} -- [description]
    """
    Amax = max(yields.keys())
    nuclist = [100, 101]
    csilist = [cs_nincl(Z, A), cs_pincl(Z, A)]
    sub_frags = {}
    for nz in range(0, int(Z / 2.) + 1):
        for nn in range(0, int((A - Z) / 2.) + 1):
            Ared = min(Amax, nn + nz)

            # print nn, nz, A-Z, Z
            if nn + nz == 0:
                # add pion contribution
                nuclist += [A*100 + Z,
                            A*100 + Z - 1,
                            A*100 + Z + 1]
                csilist += [cs_gpi(A) / 3,] * 3
                continue

            cs_incl = 0
            new_frag = 0
            if (nn == 1) and (nz == 0):
                cs_incl = cs_gn(A)
            elif (nn > 1) and (nz == 0):
                cs_incl = cs_gxn(A, nn)
            elif (nn == 0) and (nz == 1):
                cs_incl = cs_gp(Z)
            elif (nn >= 1) and (nz >= 1):
                cs_incl = cs_gSp(Z, A, nz, nn)
                new_frag = 100 * Ared + Ared/2 - nz
                if new_frag > 0:
                    csilist.append(cs_incl)
                    nuclist.append(new_frag)				
                sub_frags = yields[Ared]

            if cs_incl > 0:
                csilist.append(cs_incl)
                nuclist.append(100 * (A - nn - nz) + Z - nz)
                if sub_frags:
                    suma = 0
                    for nuc, val in sub_frags.items():
                        Af, _, _ = get_AZN(nuc)
                        suma += Af * val
                    norm = Ared / suma

                    for nuc, val in sub_frags.items():
                        if nuc in nuclist:
                            csilist[nuclist.index(nuc)] += val * norm * cs_incl
                        else:
                            csilist.append(val * norm * cs_incl)
                
    return nuclist, csilist


def multiplicity_table(mother):
    '''Returns a dict with the multiplicities
    for all fragments from mother is contained.
    The differentence with spallation_inclusive is that
    here all processes are contained
    '''
    gxn_mult = gxn_multiplicities(mother)
    sp_mult = spallation_multiplicities(mother)
    
    Am, Zm, _ = get_AZN(mother)

    cspi = cs_gpi(Am)
    csp = cs_gp(A=Am)
    csn = cs_gn(Am)
    csxn = cs_gxn_all(Am)
    cs_tot = .28 * Am
    csSp = cs_tot - (cspi + csp + csn + csxn)

    multiplicities = {100: 1.*csn/cs_tot,
                      101: 1.*csp/cs_tot,
                      mother - 100: 1.*csn/cs_tot,
                      mother - 101: 1.*csp/cs_tot,}

    for dau, mult in gxn_mult.items():
        if dau in multiplicities:
            multiplicities[dau] += mult * csxn / cs_tot
        else:
            multiplicities[dau] = mult * csxn / cs_tot

    for dau, mult in sp_mult.items():
        if dau in multiplicities:
            multiplicities[dau] += mult * csSp / cs_tot
        else:
            multiplicities[dau] = mult * csSp / cs_tot

    return multiplicities


def main():
    pass
    # resmul = residual_multiplicities()
    print(list(resmul.keys()))

if __name__ == '__main__':
    main()
