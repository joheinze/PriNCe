import numpy as np
from os.path import join
import prince_cr.config as config
from prince_cr.cross_sections.photo_meson import SophiaSuperposition
from prince_cr.data import spec_data

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