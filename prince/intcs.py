from abc import ABCMeta, abstractmethod
from os.path import isfile, join

import numpy as np

from prince.util import get_AZN, get_interp_object, info, load_or_convert_array
from prince_config import config

class CrossSectionBase(object):
    """Base class for cross section interfaces to tabulated models.

    The class is abstract and it is not inteded to be instatiated.
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        # Tuple, defining min and max energy cuts on the grid
        self._range = None
        # Energy grid, as defined in files
        self._egrid_tab = None
        # Grid in y (defined as self.egrid/2)
        self._ygrid_tab = None
        # Dictionary of incl. cross sections on egrid, indexed by (mother, daughter)
        self._incl_tab = None
        # Dictionary of nonel. cross sections on egrid, indexed by (mother)
        self._nonel_tab = None
        # List of available mothers for nonel cross sections
        self.nonel_idcs = None
        # List of available (mothers,daughter) reactions in incl. cross sections
        self.incl_idcs = None
        # Dictionary of (mother, daughter) reactions for each mother
        self.reactions = None

        # Dictionary of reponse function interpolators
        self.resp_nonel_intp = {}
        self.resp_incl_intp = {}

    def set_range(self, e_min=None, e_max=None):
        """Set energy range within which to return tabulated data.

        Args:
            e_min (float): minimal energy in GeV
            e_max (float): maximal energy in GeV
        """
        if e_min is None:
            e_min = np.min(self._egrid_tab)
        if e_max is None:
            e_max = np.max(self._egrid_tab)

        info(2, "Setting range to {0:3.2e} - {1:3.2e}".format(e_min, e_max))
        self._range = np.where((self._egrid_tab >= e_min) & (self._egrid_tab <=
                                                             e_max))[0]
        info(2, "Range set to {0:3.2e} - {1:3.2e}".format(
            np.min(self._egrid_tab[self._range]),
            np.min(self._egrid_tab[self._range])))

    def egrid(self):
        """Returns energy grid of the tabulated data in selected range.

        Returns:
            (numpy.array): Energy grid in GeV
        """

        return self._egrid_tab[self._range]

    def ygrid(self):
        """Returns grid of y values (average CMS energy in reponse functions).

        Returns:
            (numpy.array): y grid in GeV
        """

        return self._ygrid_tab[1:]

    def _gen_channel_index(self):
        """Construct a list of mothers and (mother, daughter) indices"""

        self.nonel_idcs = sorted(self._nonel_tab.keys())
        self.incl_idcs = sorted(self._incl_tab.keys())
        self.reactions = {}

        for mo, da in self.incl_idcs:
            if get_AZN(da)[0] > get_AZN(mo)[0]:
                raise Exception(
                    'Daughter {0} heavier than mother {1}. Physics??'.format(
                        da, mo))
            if mo not in self.reactions:
                self.reactions[mo] = {}
            self.reactions[mo] = (mo, da)

    def nonel(self, mother):
        """Returns non-elastic cross section.

        Absorption cross section of `mother`, which is
        the total minus elastic, or in other words, the inelastic
        cross section.

        Args:
            mother (int): Mother nucleus(on)

        Returns:
            (numpy.array): Inclusive cross section in :math:`cm^{-2}`
                           on self._egrid_tab
        """

        if mother not in self._nonel_tab:
            raise Exception('Mother', mother, 'unknown')

        return self._nonel_tab[mother][self._range]

    def incl(self, mother, daughter):
        """Returns inclusive cross section.

        Inclusive cross section for daughter in photo-nuclear
        interactions of `mother`.

        Args:
            mother (int): Mother nucleus(on)
            daughter (int): Daughter nucleus(on)

        Returns:
            (numpy.array): Inclusive cross section in :math:`cm^{-2}`
                           on self._egrid_tab
        """

        if (mother, daughter) not in self._incl_tab:
            raise Exception(
                '({0},{1}) combination not in inclusive cross sections'.format(
                    mother, daughter))

        return self._incl_tab[(mother, daughter)][self._range]

    def response_function(self, mother, daughter=None):
        """Reponse function :math:`f(y)` or :math:`g(y)` as
        defined in the note.

        Returns :math:`f(y)` or :math:`g(y)` if a daughter
        index is provided.

        Args:
            mother (int): mother nucleus(on)
            daughter (int, optional): daughter nucleus(on)

        Returns:
            (numpy.array) Reponse function on self._ygrid_tab
        """
        from scipy import integrate

        if self._range[0] != 0:
            raise Exception('Response functions can not computed ' +
                            'if a minimal energy is defined in range.')

        cross_section = self.nonel(mother)
        if daughter != None:
            cross_section = self.incl(mother, daughter)

        integral = integrate.cumtrapz(
            self.egrid() * cross_section, x=self.egrid())

        return integral / (2 * self._ygrid_tab[1:]**2)

    def precomp_response_func(self):
        self.resp_nonel_intp = {}
        for mother in self.nonel_idcs:
            self.resp_nonel_intp[mother] = get_interp_object(
                self._ygrid_tab[1:], self.response_function(mother))

        self.resp_incl_intp = {}
        for mother in self.incl_idcs:
            self.resp_incl_intp[(mother, daughter)] = get_interp_object(
                self._ygrid_tab[1:], self.response_function(mother, daughter))


class CrossSectionInterpolator(CrossSectionBase):
    """Concatenates and interpolates cross section models.

    """

    def __init__(self, model_list):
        """The constructor takes a list of models in the following format::

            $ model_list = [(e_threshold, m_class, m_args),
                            (e_threshold, m_class, m_args)]

        The arguments m1 or m2 are classes derived from
        :class:`CrossSectionBase`. `e_threshold_` is the
        minimal energy for above which a model is used. The maximum
        energy until which a model class is used, is the threshold of the
        next one. m_args are optional arguments passed to the
        constructor of `m_class`.

        Args:
            model_list (list): format as specified above
        """
        CrossSectionBase.__init__(self)

        self._concatenate_models(model_list)

    def _concatenate_models(self, model_list):

        info(1, "Attempt to combine", len(model_list), "models.")

        nmodels = len(model_list)
        m_ranges = []
        grid_list = []

        m_objects = []
        for im, (e_thr, mclass, margs) in enumerate(model_list):
            csm_inst = mclass(*margs)
            if im < nmodels - 1:
                csm_inst.set_range(e_thr, model_list[im + 1][0])
            else:
                csm_inst.set_range(e_thr)

            m_objects.append(csm_inst)

        self._egrid_tab = np.concatenate(
            [m_objects[i].egrid() for i in range(nmodels)])
        self._ygrid_tab = self._egrid_tab / 2.
        self._nonel_tab = {}
        self._incl_tab = {}
        self.set_range()

        info(3, 'Concatenating nonelastic cross sections')
        for mother in m_objects[0].nonel_idcs:
            self._nonel_tab[mother] = np.concatenate(
                [m_objects[i].nonel(mother) for i in range(nmodels)])

        info(3, 'Concatenating inclusive cross sections')
        for mother, daughter in m_objects[0].incl_idcs:
            self._incl_tab[(mother, daughter)] = np.concatenate(
                [m_objects[i].incl(mother, daughter) for i in range(nmodels)])

        self._gen_channel_index()
        # now also precompute the response function
        # self.precomp_response_func()


class SophiaSuperposition(CrossSectionBase):
    """ Cross sections generated using the Sophia event generator for protons and neutrons.
    Data available from 10 MeV to 10^10 GeV
    """

    def __init__(self, *args, **kwargs):
        CrossSectionBase.__init__(self)
        self._load()

    def _load(self):
        info(2, "Loading SOPHIA cross sections from file.")

        self._egrid_tab, self.cs_proton_grid, self.cs_neutron_grid = \
        load_or_convert_array(
            'sophia_csec', delimiter=',', unpack=True)
        self.cs_proton_grid *= 1e-30
        self.cs_neutron_grid *= 1e-30
        self._ygrid_tab = self._egrid_tab / 2.

        self.set_range()

    def nonel(self, mother):
        """Returns non-elastic cross section.

        Absorption cross section of `mother`, which is
        the total minus elastic, or in other words, the inelastic
        cross section.

        Args:
            mother (int): Mother nucleus(on)

        Returns:
            (numpy.array): Inclusive cross section in :math:`cm^{-2}`
                           on self._egrid_tab
        """
        # now interpolate these as Spline
        _, Z, N = get_AZN(mother)

        # the nonelastic crosssection is just a superposition of
        # the proton/neutron number
        cgrid = Z * self.cs_proton_grid + N * self.cs_neutron_grid
        return cgrid[self._range]

    def incl(self, mother, daughter):
        """Returns inclusive cross section.

        Inclusive cross section for daughter in photo-nuclear
        interactions of `mother`.

        Args:
            mother (int): Mother nucleus(on)
            daughter (int): Daughter nucleus(on)

        Returns:
            (numpy.array): Inclusive cross section in :math:`cm^{-2}`
                           on self._egrid_tab
        """

        _, Z, N = get_AZN(mother)

        if daughter not in [101, 100, mother - 101, mother - 100]:
            info(5, 'mother, daughter', mother, daughter, 'out of range')
            return np.zeros_like(self.cs_proton_grid)[self._range]

        if daughter in [101, mother - 101]:
            cgrid = Z * self.cs_proton_grid
            return cgrid[self._range]
        elif daughter in [100, mother - 100]:
            cgrid = N * self.cs_neutron_grid
            return cgrid[self._range]
        else:
            raise Exception('Should not happen.')


class NeucosmaFileInterface(CrossSectionBase):
    """Tabulated disintegration cross sections from Peanut or TALYS.
    Data available from 1 MeV to 1 GeV"""

    def __init__(self, model_prefix='peanut', *args, **kwargs):
        CrossSectionBase.__init__(self)
        self._load(model_prefix)

    def _load(self, model_prefix):

        cspath = config['data_dir']

        info(2, "Load tabulated cross sections")
        # The energy grid is given in MeV, so we convert to GeV
        egrid = load_or_convert_array(
            model_prefix + "_egrid", dtype='float') * 1e-3
        info(2, "Egrid loading finished")

        # Load tables from files
        _nonel_tab = load_or_convert_array(model_prefix + "_IAS_nonel")
        _incl_tab = load_or_convert_array(model_prefix + "_IAS_incl_i_j")

        # Integer idices of mothers and inclusive channels are stored
        # in first column(s)
        pid_nonel = _nonel_tab[:, 0].astype('int')
        pids_incl = _incl_tab[:, 0:2].astype('int')

        # the rest of the line denotes the crosssection on the egrid in mbarn,
        # which is converted here to cm^2
        nonel_raw = _nonel_tab[:, 1:] * 1e-27
        incl_raw = _incl_tab[:, 2:] * 1e-27
        info(2, "Data file loading finished")

        # Now write the raw data into a dict structure
        _nonel_tab = {}
        for pid, csgrid in zip(pid_nonel, nonel_raw):
            _nonel_tab[pid] = csgrid
        _incl_tab = {}

        # mo = mother, da = daughter
        for (mo, da), csgrid in zip(pids_incl, incl_raw):
            _incl_tab[mo, da] = csgrid

        self._egrid_tab = egrid
        self._ygrid_tab = egrid / 2
        self._nonel_tab = _nonel_tab
        self._incl_tab = _incl_tab
        # Set initial range to whole egrid
        self.set_range()

        self._gen_channel_index()

        info(2, "Finished initialization")


if __name__ == "__main__":
    pass
