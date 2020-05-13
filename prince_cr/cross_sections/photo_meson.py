
from os.path import join

import numpy as np

from prince_cr.data import spec_data
from prince_cr.util import info, bin_widths, get_AZN
import prince_cr.config as config

from .base import CrossSectionBase
class SophiaSuperposition(CrossSectionBase):
    """ Cross sections generated using the Sophia event generator for protons and neutrons.
    Includes redistribution functions into secondaries

    Cross section for Nuclei are built as a superposition:
    csec_(Z,A) = Z * csec_P + (A - Z) * csec_N

    WARNING: This will not check, if the requested nucleus is existent in nature!
    """

    def __init__(self, *args, **kwargs):
        # Tell the interpolator that this model contains the necessary features
        # for energy redistribution functions
        self.supports_redistributions = True
        CrossSectionBase.__init__(self)
        self._load()

    def _load(self):
        from prince_cr.data import db_handler
        info(2, "Load tabulated cross sections")
        photo_nuclear_tables = db_handler.photo_meson_db('SOPHIA')
        info(2, "Loading SOPHIA cross sections from file.")

        egrid = photo_nuclear_tables["energy_grid"]
        xbins = photo_nuclear_tables["xbins"]
        info(2, "Egrid loading finished")

        # Integer idices of mothers and inclusive channels are stored
        # in first column(s)
        pid_nonel = photo_nuclear_tables["inel_mothers"]
        pids_incl = photo_nuclear_tables["mothers_daughters"]

        # the rest of the line denotes the crosssection on the egrid in mbarn,
        # which is converted here to cm^2
        nonel_raw = photo_nuclear_tables["inelastic_cross_sctions"]
        incl_raw = photo_nuclear_tables["fragment_yields"]

        info(2, "Data file loading finished")

        self._egrid_tab = egrid
        self.cs_proton_grid = nonel_raw[pid_nonel==101].flatten()
        self.cs_neutron_grid = nonel_raw[pid_nonel==100].flatten()

        self.xbins = xbins
        self.redist_proton = {}
        self.redist_neutron = {}
        for (mo, da), csgrid in zip(pids_incl, incl_raw):
            if mo == 101:
                self.redist_proton[da] = csgrid
            elif mo == 100:
                self.redist_neutron[da] = csgrid
            else:
                raise Exception(f'Sophia model should only contain protons and neutrons, but has mother id {mo}')

        # set up inclusive differential channels for protons and neutron
        # The model can return both, integrated over x and redistributed.
        for da in sorted(self.redist_proton):
            self.incl_diff_idcs.append((101, da))
            #self.incl_idcs.append((101, da))
        for da in sorted(self.redist_neutron):
            self.incl_diff_idcs.append((100, da))
            #self.incl_idcs.append((100, da))

        # For more convenient generation of trivial redistribution matrices when joining
        self.redist_shape = (self.xbins.shape[0], self._egrid_tab.shape[0])
        self.set_range()

    def generate_incl_channels(self, mo_indices):
        """Generates indices for all allowed channels given mo_indices
        
        Args:
            mo_indices (list of ints): list of indices for mother nuclei

        Returns:
           Returns:
            list of tuples: list of allowed channels given as (mo_idx, da_idx)
        """
        incl_channels = []
        #return incl_channels

        # This model is a Superposition of protons and neutrons, so we need all respective daughters
        for idx in mo_indices:
            # if idx > 200:
            #     continue
            # add all daughters that are allowed for protons
            for da in sorted(self.redist_proton):
                incl_channels.append((idx, da))
            # add all daughters that are allowed for neutrons
            for da in sorted(self.redist_neutron):
                incl_channels.append((idx, da))
            # add residual nucleus to channel lists
            # NOTE: Including residual nuclei that are not in the disintegration model
            #       Caused some problems, therefore we ignore them here. The effect is
            #       minor though, as photo-meson is subdominant
            # for da in [idx - 101, idx - 100]:
            #    if da > 199:
            #        incl_channels.append((idx, da))

        self.incl_diff_idcs = sorted(list(set(self.incl_diff_idcs + incl_channels)))

        return incl_channels

    def nonel(self, mother):
        r"""Returns non-elastic cross section.

        Absorption cross section of `mother`, which is
        the total minus elastic, or in other words, the inelastic
        cross section.

        Args:
            mother (int): Mother nucleus(on)

        Returns:
           Returns:
            (numpy.array, numpy.array): self._egrid_tab (:math:`\epsilon_r`),
            nonelastic (total) cross section in :math:`cm^{-2}`
        """

        # now interpolate these as Spline
        _, Z, N = get_AZN(mother)

        # the nonelastic crosssection is just a superposition of
        # the proton/neutron number
        cgrid = Z * self.cs_proton_grid + N * self.cs_neutron_grid
        return self.egrid, cgrid[self._range]

    def incl(self, mother, daughter):
        r"""Returns inclusive cross section.

        Inclusive cross section for daughter in photo-nuclear
        interactions of `mother`.

        Args:
            mother (int): Mother nucleus(on)
            daughter (int): Daughter nucleus(on)

        Returns:
            (numpy.array, numpy.array): self._egrid_tab (:math:`\epsilon_r`),
            inclusive cross section in :math:`cm^{-2}`
        """

        _, Z, N = get_AZN(mother)

        if daughter <= 101:
            # raise Exception('Boost conserving cross section called ' +
            #                 'for redistributed particle')
            from scipy.integrate import trapz

            _, cs_diff = self.incl_diff(mother, daughter)
            cs_incl = trapz(cs_diff, x=self.xcenters,
                            dx=bin_widths(self.xbins), axis=0)
            return self.egrid, cs_incl[self._range]

        elif daughter >= 200 and daughter not in [mother - 101, mother - 100]:
            info(10, 'mother, daughter', mother, daughter, 'out of range')
            return self.egrid[[0, -1]], np.array([0., 0.])

        if daughter in [mother - 101]:
            cgrid = Z * self.cs_proton_grid
            # created incl. diff. index for all particle created in p-gamma
            for da in self.redist_proton:
                self.incl_diff_idcs.append((mother, da))
            return self.egrid, cgrid[self._range]
        elif daughter in [mother - 100]:
            cgrid = N * self.cs_neutron_grid
            # created incl. diff. channel index for all particle created in n-gamma
            for da in self.redist_neutron:
                self.incl_diff_idcs.append((mother, da))
            return self.egrid, cgrid[self._range]
        else:
            raise Exception(
                'Channel {:} to {:} not allowed in this superposition model'.
                format(mother, daughter))

    def incl_diff(self, mother, daughter):
        r"""Returns inclusive differential cross section.

        Inclusive differential cross section for daughter in photo-nuclear
        interactions of `mother`. Only defined, if the daughter is distributed
        in :math:`x_{\rm L} = E_{da} / E_{mo}`

        Args:
            mother (int): Mother nucleus(on)
            daughter (int): Daughter nucleus(on)

        Returns:
            (numpy.array, numpy.array, numpy.array): :math:`\epsilon_r` grid,
            :math:`x` grid, differential cross section in :math:`{\rm cm}^{-2}`
        """

        _, Z, N = get_AZN(mother)

        if daughter > 101:
            raise Exception(
                'Redistribution function requested for boost conserving particle'
            )
        csec_diff = None
        # TODO: File shall contain the functions in .T directly
        #   JH: I left it like this on purpose, since the raw data is ordered more consistently
        #       i.e. redist.shape = cs_nonel.shape + xbins.shape
        #       The ordering should rather be changed in the rest of the code
        if daughter in self.redist_proton:
            csec_diff = self.redist_proton[daughter].T * Z

        if daughter in self.redist_neutron:
            cgrid = N * self.cs_neutron_grid
            if np.any(csec_diff):
                csec_diff += self.redist_neutron[daughter].T * N
            else:
                csec_diff = self.redist_neutron[daughter].T * N

        return self.egrid, csec_diff[:, self._range]



class EmpiricalModel(SophiaSuperposition):
    """Photomeson model based on empirical relations and experimental data.
    More info:  L. Morejon et al JCAP11(2019)007
    """

    def __init__(self, *args, **kwargs):
        """Initialize object based on a universal function, and a 
        scaling function.
        """
        SophiaSuperposition.__init__(self)

        self._load_universal_function()

        self._load_pion_function()

        self._fill_multiplicity()

        # self._optimize_and_generate_index() # without this works!!

    def A_eff(self, A, x):
        """Returns the effective A using precomputed scaling coefficients based on
        experimental data.
        """
        from scipy.interpolate import interp1d

        def sigm(x, shift=0., gap=1, speed=1, base=0., rising=False):
            """Models a general sigmoid with multiple parameters.

            Parameters:
            -----------
            x: x values, argument
            shift: middle point, inflection point
            gap: maximal value - minimal value
            speed: controls the speed of the change, 
            base: minimal value
            """
            sigmoid = 1. /(1 + np.exp(- speed * (x - shift)))
            
            if rising:
                return gap * sigmoid + base
            else:
                return gap*( 1. - sigmoid) + base

        mass_scaling_file = join(config.data_dir, 'scaling_lines')
        nsc = np.load(mass_scaling_file, allow_pickle=True,encoding='latin1')

        # defining the transistion smoothing function s 
        # emid = np.array(range(600, 630, 10) + range(680, 720, 10))
        # ymid = np.where(emid < 650, np.interp(emid, nsc[0], nsc[1]), .66)
        # s = interp1d(emid, ymid, 'slinear')

        y = []
        for xi in x:
            if xi < 7:
                y.append(sigm(xi, .7, .099, .7, .942))
            elif xi < 300:
                y.append(np.interp(xi, nsc[0], nsc[1]))
            else:
                y.append(sigm(xi, 430, .2, .01, .66))

        return A**np.array(y)

    def fade(self, cs1, cs2, indices=None):
        """Smoothes the transition from cs1 to cs2
        Uses a sigmoid to smothen the transition between differential 
        cross sections cs1 and cs2 in the energy range defined by the
        indices.
        """
        if indices is None:
            return cs1
            
        x = -100 * np.ones_like(cs1)
        x[..., indices[-1]:] = 100
        x[..., indices] = np.linspace(-5, 5, len(indices))
        
        def sigmoid(x):
            return 1./(1 + np.exp(-x))

        return cs1[..., :]*(1 - sigmoid(x)) + cs2[..., :]*sigmoid(x)

    def _load_universal_function(self):
        """Returns the universal function on a fixed energy range
        """
        from pickle import load as pickle_load
        from scipy.interpolate import UnivariateSpline

        uf_file = join(config.data_dir, 'universal-spline.pkl')
        with open(uf_file, 'rb') as f:
            tck = pickle_load(f,encoding='latin1') 

        self.univ_spl = UnivariateSpline._from_tck(tck)

    def _load_pion_function(self):
        """Returns the universal function on a fixed energy range
        """
        from pickle import load as pickle_load
        from scipy.interpolate import UnivariateSpline

        uf_file = join(config.data_dir, 'pion_spline.pkl')
        with open(uf_file, 'rb') as f:
            tck = pickle_load(f,encoding='latin1') 

        self.pion_spl = UnivariateSpline._from_tck(tck)

    def _fill_multiplicity(self, *args, **kwargs):
        """Populates all tabs (_nonel_tab, _incl_tab, _incl_diff_tab) so they can work
        with the _optimize_indices() method of the base class (CrossSectionBase).
        """
        from ._phenom_relations import multiplicity_table

        self._nonel_tab = {100:(), 101:()}
        
        for mom in self._nonel_tab:
            for dau in [2, 3, 4, 100, 101]:
                self._incl_diff_tab[mom, dau] = ()
                
        new_multiplicity = {}
        nuclides = sorted([k for k in spec_data.keys() if isinstance(k, int)])
        for mom in nuclides:
            A, _, _ = get_AZN(mom)
            if (mom < 101) or (A > config.max_mass) or \
                isinstance(mom, str) or (spec_data[mom]['lifetime'] < config.tau_dec_threshold):
                continue
            mults = multiplicity_table(mom)
            # dau_list, csincl_list = zip(*((k, v) for k, v in mults.iteritems()))
            
            self._nonel_tab[mom] = ()
            for dau in [2, 3, 4, 100, 101]:
                self._incl_diff_tab[mom, dau] = SophiaSuperposition.incl_diff(self, mom, dau)[1]
            
            for dau, mult in mults.items():
                new_multiplicity[mom, dau] = mult
                self._incl_tab[mom, dau] = np.array([])
            
        self.multiplicity = new_multiplicity

    def nonel(self, mother):
        """Computes nonelasatic as A * universal_funtion below 1.9GeV
        and as SophiaSuperposition with given scaling betond 1.9GeV
        """
        e_max = 1.2  # prefixed based on data
        e_scale = .3  # prefixed based on data
        A, _, _ = get_AZN(mother)
        egrid, csnonel = SophiaSuperposition.nonel(self, mother)

        csnonel[egrid <= e_max] = A * self.univ_spl(egrid[egrid <= e_max])
        csnonel[egrid > e_scale] = (csnonel * self.A_eff(A, egrid)/A)[egrid > e_scale]

        return egrid, csnonel

    def incl(self, mother, daughter):
        """Computes inclusive from nonel * M with M is
        multiplicity value stored in internal table
        """
        from scipy.integrate import trapz

        if daughter <= config.redist_threshold_ID:
            _, cs_diff = self.incl_diff(mother, daughter)
            cs_incl = trapz(cs_diff, x=self.xcenters,
                            dx=bin_widths(self.xbins), axis=0)
            return self.egrid, cs_incl[self._range]
        elif (mother, daughter) in self.multiplicity:
            egrid, cs_nonel = self.nonel(mother)
            cs_nonel *= self.multiplicity[mother, daughter]
            return egrid, cs_nonel
        else:
            return SophiaSuperposition.incl(self, mother, daughter)

    def incl_diff(self, mother, daughter):
        """Uses corresponding method from SophiaSuperposition class,
        adding nonel to increase multiplicity by one
        """
        egrid, cs_diff = SophiaSuperposition.incl_diff(self, mother,
                                                       daughter)
        if (mother, daughter) in self.multiplicity:
            xw = self.xwidths[-1]  # accounting for bin width
            _, cs_nonel = self.nonel(mother)
            cs_diff[-1, :] += \
                self.multiplicity[mother, daughter] * cs_nonel / xw
        elif (mother > 101) and (daughter in [2, 3, 4]):  # if it's a pion rescale to A^2/3
            def superposition_incl(mother, daughter):
                from scipy.integrate import trapz
                _, Z, N = get_AZN(mother)
                cs_diff = self.redist_proton[daughter].T * Z * self.cs_proton_grid + \
                    self.redist_neutron[daughter].T * N * self.cs_neutron_grid
                cs_incl = trapz(cs_diff, x=self.xcenters,
                                dx=bin_widths(self.xbins), axis=0)
                return cs_incl[self._range]

            def superposition_multiplicities(mother ,daughter):
                _, Z, N = get_AZN(mother)
                cs_incl = superposition_incl(mother, daughter)
                cs_nonel = Z * self.cs_proton_grid + N * self.cs_neutron_grid
                return cs_incl / cs_nonel[self._range]

            Am, _, _ = get_AZN(mother)
            cs_diff *= float(Am)**(-1/3.)  # ... rescaling SpM to A^2/3

            cs_incl_pi0_sophia = superposition_incl(mother, daughter) * float(Am)**(-1/3.)
            cs_incl_pi0_data = 1e-30*self.pion_spl(egrid * 1e3)*Am**(2./3)

            M_pi = superposition_multiplicities(mother, daughter)
            M_pi0 = superposition_multiplicities(mother, 4)

            renorm = M_pi / M_pi0 * cs_incl_pi0_data / cs_incl_pi0_sophia
            cs_diff_renormed = cs_diff * renorm
            cs_diff_renormed = self.fade(cs_diff, cs_diff_renormed, range(32)) # hardcoded index, found manually
            cs_diff = self.fade(cs_diff_renormed, cs_diff, range(55, 95)) # hardcoded index, found manually

            # # additional correction to pion scaling high energies, after paper was corrected
            # def sigm(x, shift=0., gap=1, speed=1, base=0., rising=False):
            #     """Models a general sigmoid with multiple parameters.

            #     Parameters:
            #     -----------
            #     x: x values, argument
            #     shift: middle point, inflection point
            #     gap: maximal value - minimal value
            #     speed: controls the speed of the change, 
            #     base: minimal value
            #     """
            #     sigmoid = 1. /(1 + np.exp(- speed * (x - shift)))
                
            #     if rising:
            #         return gap * sigmoid + base
            #     else:
            #         return gap*( 1. - sigmoid) + base

            alpha_plus = np.where(egrid <= 1., 2./3, 1 - np.exp( -4./7*(egrid - 1)**.5)/3)
            cs_diff *= Am**(alpha_plus - 2./3)

        return egrid, cs_diff
