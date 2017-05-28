"""The module contains everything to handle cross section interfaces."""

from abc import ABCMeta, abstractmethod
from os.path import isfile, join

import numpy as np

from prince.util import *
from prince_config import config, spec_data


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
        # Dictionary of incl. cross sections on egrid, indexed by (mother, daughter)
        self._incl_tab = None
        # Dictionary of nonel. cross sections on egrid, indexed by (mother)
        self._nonel_tab = None
        # List of available mothers for nonel cross sections
        self.nonel_idcs = []
        # List of available (mothers,daughter) reactions in incl. cross sections
        self.incl_idcs = []
        # List of all known particles (after optimization)
        self.known_species = []
        # List of all known inclusive channels (after optimization)
        self.known_channels = []
        # Dictionary of (mother, daughter) reactions for each mother
        self.reactions = {}

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
            np.max(self._egrid_tab[self._range])))

    def egrid(self):
        """Returns energy grid of the tabulated data in selected range.

        Returns:
            (numpy.array): Energy grid in GeV
        """

        return self._egrid_tab[self._range]

    def _optimize_and_generate_index(self):
        """Construct a list of mothers and (mother, daughter) indices.

        Args:
            just_reactions (bool): If True then fill just the reactions index.
        """

        # Integrate out short lived processes and leave only stable particles
        # in the databases
        self._optimize_channels()

        self.nonel_idcs = sorted(self._nonel_tab.keys())
        self.incl_idcs = sorted(self._incl_tab.keys())

        for mo, da in self.incl_idcs:
            if da > 100 and get_AZN(da)[0] > get_AZN(mo)[0]:
                raise Exception(
                    'Daughter {0} heavier than mother {1}. Physics??'.format(
                        da, mo))
            if mo not in self.reactions:
                self.reactions[mo] = []
                self.known_species.append(mo)
            elif (mo, da) not in self.reactions[mo]:
                # Make sure it's a unique list to avoid unnecessary loops
                self.reactions[mo].append((mo, da))
                self.known_channels.append((mo, da))
                self.known_species.append(da)

        self.known_species = sorted(list(set(self.known_species)))
        self.known_channels = sorted(list(set(self.known_channels)))

    def _optimize_channels(self):
        """Follows decay chains until all inclusive reactions point to
        stable final state particles.

        The "tau_dec_threshold" parameter in the config controls the
        definition of stable. Nuclei for which no the decays are
        unkown, will be forced to beta-decay until they reach a stable
        element.
        """
        # The new dictionary that will replace _incl_tab
        new_incl_tab = {}
        threshold = config["tau_dec_threshold"]
        # How to indent debug printout for recursion
        dbg_indent = lambda lev: 4 * lev * "-" + ">" if lev else ""

        info(2, "Integrating out species with lifetime smaller than",
             threshold)
        info(3,
             ("Before optimization, the number of known primaries is {0} with "
              + "in total {1} inclusive channels").format(
                  len(self._nonel_tab), len(self._incl_tab)))

        def follow_chain(first_mo, da, value, reclev):
            """Recursive function to follow decay chains until all
            final state particles are stable"""

            if da not in spec_data:
                info(
                    3,
                    dbg_indent(reclev),
                    'daughter {0} unknown, forcing beta decay. Not Implemented yet!!'.
                    format(da))
                return
                # dict_add(new_incl_tab,(mo,da)) TODO something here

            # Daughter is stable. Add it to the new dictionary and terminate
            # recursion
            if spec_data[da]["lifetime"] >= threshold:
                info(10,
                     dbg_indent(reclev),
                     'daughter {0} stable. Adding to ({1}, {2})'.format(
                         da, first_mo, da))

                dict_add(new_incl_tab, (first_mo, da), value)
                return

            # ..otherwise follow decay products of this daughter, tracking the
            # original mother particle. The cross section (value) is reduced by
            # the branching ratio into this partcular channel
            for br, daughters in spec_data[da]["branchings"]:
                info(10,
                     dbg_indent(reclev),
                     ("{3} -> {0:4d} -> {2:4.2f}: {1}").format(
                         da, ", ".join(map(str, daughters)), br, first_mo))

                for chained_daughter in daughters:
                    # Follow each secondary and increment the recursion level by one
                    follow_chain(first_mo, chained_daughter, br * value,
                                 reclev + 1)

        # Launch the reduction for each inclusive channel
        for (mo, da), value in self._incl_tab.items():
            if mo in spec_data and spec_data[mo]["lifetime"] >= threshold:
                follow_chain(mo, da, value, 0)
            else:
                info(
                    10,
                    "Primary species {0} does not fulfill stability criteria.".
                    format(mo))
                # Remove mother from _nonel as well, since those particles will
                # not live long enough for propagation
                if mo in self._nonel_tab:
                    _ = self._nonel_tab.pop(mo)

        # Overwrite the old dictionary
        self._incl_tab = new_incl_tab
        info(3, "The number of channels after optimization is",
             len(self._incl_tab))

        info(3,
             ("After optimization, the number of known primaries is {0} with "
              + "in total {1} inclusive channels").format(
                  len(self._nonel_tab), len(self._incl_tab)))

    def nonel_scale(self, mother, scale='A'):
        """Returns the nonel cross section scaled by `scale`.

        Convenience funtion for plotting, where it is important to
        compare the cross section per nucleon.

        Args:
            mother (int): Mother nucleus(on)
            scale (float): If `A` then nonel/A is returned, otherwise
                           scale can be any float.

        Returns:
            (numpy.array, numpy.array): Tuple of Energy grid in GeV,
                                        scale * inclusive cross section
                                        in :math:`cm^{-2}`
        """

        egr, cs = self.nonel(mother)

        if scale == 'A':
            scale = 1. / get_AZN(mother)[0]

        return egr, scale * cs

    def incl_scale(self, mother, daughter, scale='A'):
        """Same as :func:`~intcs.CrossSectionBase.nonel_scale`,
        just for inclusive cross sections.
        """

        egr, cs = self.incl(mother, daughter)

        if scale == 'A':
            scale = 1. / get_AZN(mother)[0]

        return egr, scale * cs

    def response_function_scale(self, mother, daughter=None, scale='A'):
        """Same meaning as :func:`~intcs.CrossSectionBase.nonel_scale`,
        just for response functions.
        """

        ygr, cs = self.response_function(mother, daughter)

        if scale == 'A':
            scale = 1. / get_AZN(mother)[0]

        return ygr, scale * cs

    def nonel(self, mother):
        """Returns non-elastic cross section.

        Absorption cross section of `mother`, which is
        the total minus elastic, or in other words, the inelastic
        cross section.

        Args:
            mother (int): Mother nucleus(on)

        Returns:
            (numpy.array, numpy.array): Tuple of Energy grid in GeV, inclusive cross
                                        section in :math:`cm^{-2}`
        """

        if mother not in self._nonel_tab:
            raise Exception('Mother {0} unknown.'.format(mother))
            # return self.egrid()[[0, -1]], self._nonel_tab[(
            #     mother)][self._range][[0, -1]]
        if isinstance(self._nonel_tab[mother], tuple):
            return self._nonel_tab[mother]
        else:
            return self.egrid(), self._nonel_tab[mother][self._range]

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

            # return self.egrid()[[0, -1]], self._incl_tab[(
            #     mother, daughter)][self._range][[0, -1]]

        # If _nonel_tab contains tuples of (egrid, cs) return tuple
        # otherwise return (egrid(), cs) in range defined by self.range
        if isinstance(self._incl_tab[(mother, daughter)], tuple):
            return self._incl_tab[(mother, daughter)]
        return self.egrid(), self._incl_tab[(mother, daughter)][self._range]

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

        egrid, cross_section = None, None

        if daughter != None:
            egrid, cross_section = self.incl(mother, daughter)
        else:
            egrid, cross_section = self.nonel(mother)

        ygrid = egrid[1:] / 2.

        integral = integrate.cumtrapz(egrid * cross_section, x=egrid)

        return ygrid, integral / (2 * ygrid**2)

    def _precomp_response_func(self):
        """Interpolate each response function and store interpolators.

        Uses :func:`prince.util.get_interp_object` as interpolator.
        This might result in too many knots and can be subject to
        future optimization.
        """

        info(2, 'Computing interpolators for response functions')
        self.resp_nonel_intp = {}
        for mother in self.nonel_idcs:
            self.resp_nonel_intp[mother] = get_interp_object(
                *self.response_function(mother))

        self.resp_incl_intp = {}
        for mother, daughter in self.incl_idcs:
            self.resp_incl_intp[(mother, daughter)] = get_interp_object(
                *self.response_function(mother, daughter))


class CrossSectionInterpolator(CrossSectionBase):
    """Joins and interpolates cross section models.

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
        # References to model instances to be joined
        self.model_refs = None
        self._join_models(model_list)

    def _join_models(self, model_list):

        info(1, "Attempt to join", len(model_list), "models.")

        nmodels = len(model_list)
        m_ranges = []
        grid_list = []

        self.model_refs = []
        # Construct instances of models and set ranges where they are valid
        for im, (e_thr, mclass, margs) in enumerate(model_list):
            csm_inst = mclass(*margs)
            if im < nmodels - 1:
                csm_inst.set_range(e_thr, model_list[im + 1][0])
            else:
                csm_inst.set_range(e_thr)

            self.model_refs.append(csm_inst)

        # Create a list of nonel and incl cross section in all models
        self.nonel_idcs = sorted(
            list(set(sum([m.nonel_idcs for m in self.model_refs], []))))
        self._nonel_tab = {}
        for mo in self.nonel_idcs:
            self._nonel_tab[mo] = self._join_nonel(mo)

        self.incl_idcs = sorted(
            list(set(sum([m.incl_idcs for m in self.model_refs], []))))
        self._incl_tab = {}
        for mo, da in self.incl_idcs:
            self._incl_tab[(mo, da)] = self._join_incl(mo, da)

        self._optimize_and_generate_index()

        # now also precompute the response function
        self._precomp_response_func()

    def _join_nonel(self, mother):
        """Returns the non-elastic cross section of the joined models.
        """

        info(5, 'Joining nonelastic cross sections for', mother)

        egr = []
        nonel = []
        for mod in self.model_refs:
            e, cs = mod.nonel(mother)
            egr.append(e)
            nonel.append(cs)

        return np.concatenate(egr), np.concatenate(nonel)

    def _join_incl(self, mother, daughter):
        """Returns joined incl cross sections."""

        info(5, 'Joining inclusive cross sections for channel', (mother,
                                                                 daughter))

        egr = []
        incl = []
        for mod in self.model_refs:
            e, cs = mod.incl(mother, daughter)
            egr.append(e)
            incl.append(cs)

        return np.concatenate(egr), np.concatenate(incl)


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
        # Take each n-th row to reduce the nuber of datapoints
        nr = config["sophia_grid_skip"]
        self._egrid_tab, self.cs_proton_grid, self.cs_neutron_grid = \
            self._egrid_tab[::nr], self.cs_proton_grid[::nr], self.cs_neutron_grid[::nr]
        self.cs_proton_grid *= 1e-30
        self.cs_neutron_grid *= 1e-30

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
        return self.egrid(), cgrid[self._range]

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
            info(10, 'mother, daughter', mother, daughter, 'out of range')
            return self.egrid()[[0, -1]], np.array([0., 0.])

        if daughter in [101, mother - 101]:
            cgrid = Z * self.cs_proton_grid
            return self.egrid(), cgrid[self._range]
        elif daughter in [100, mother - 100]:
            cgrid = N * self.cs_neutron_grid
            return self.egrid(), cgrid[self._range]
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
        self._nonel_tab = _nonel_tab
        self._incl_tab = _incl_tab
        # Set initial range to whole egrid
        self.set_range()

        self._optimize_and_generate_index()

        info(2, "Finished initialization")


if __name__ == "__main__":
    pass
