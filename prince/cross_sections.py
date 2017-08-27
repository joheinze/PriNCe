"""The module contains everything to handle cross section interfaces."""

from abc import ABCMeta, abstractmethod
from os.path import isfile, join

import numpy as np

from prince.util import *
from prince_config import config, spec_data
import prince.decays as decs

# TODO:
# - CrossSectionInterpolator._join_incl_diff() does currently not work properly for inclusive differential crossections
#   there are two problems
#     - the class combines the channel indices from all models,
#       however sophia does not provide these, and still introduces indices for lighter particles
#     - we need a way to combine channels where one models provides an x distribution, while the other does not.
#       The problem arrises, as incl refers to x = 1 while incl_diff is distributed over bins with bin_center < 1


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
        # Dictionary of nonel. cross sections on egrid, indexed by (mother)
        self._nonel_tab = None
        # Dictionary of incl. cross sections on egrid, indexed by (mother, daughter)
        self._incl_tab = None
        # Dictionary of incl. diff. cross sections on egrid, indexed by (mother, daughter)
        self._incl_diff_tab = None
        # List of available mothers for nonel cross sections
        self.nonel_idcs = []
        # List of available (mothers,daughter) reactions in incl. cross sections
        self.incl_idcs = []
        # List of available (mothers,daughter) reactions in incl. diff. cross sections
        self.incl_diff_idcs = []
        # Common grid in x (the redistribution variable)
        self.xbins = None

        # Flag, which tells if the model supports secondary redistributions
        self.supports_redistributions = False
        # List of all known particles (after optimization)
        self.known_species = []
        # List of all boost conserving inclusive channels (after optimization)
        self.known_bc_channels = []
        # List of all differential inclusive channels (after optimization)
        self.known_diff_channels = []
        # Dictionary of (mother, daughter) reactions for each mother
        self.reactions = {}

        # Class name of the model
        self.mname = self.__class__.__name__

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

    @property
    def egrid(self):
        """Returns energy grid of the tabulated data in selected range.

        Returns:
            (numpy.array): Energy grid in GeV
        """

        return self._egrid_tab[self._range]

    def is_differential(self, mother, daughter):
        """Returns true if the model supports redistributions and requested
        mother/daughter combination should return non-zero redistribution matrices.

        Args:
            mother (bool): Neucosma ID of mother particle
            daughter (bool): Neucosma ID of daughter particle

        Returns:
            (bool): ``True`` if the model has this particular redistribution function
        """

        if not self.supports_redistributions:
            return False
        if (daughter <= config["redist_threshold_ID"] or
            (mother, daughter) in self.incl_diff_idcs):
            info(10, 'Daughter requires redistribution.', mother, daughter)
            return True
        info(10, 'Daughter conserves boost.', mother, daughter)
        return False

    def _optimize_and_generate_index(self):
        """Construct a list of mothers and (mother, daughter) indices.

        Args:
            just_reactions (bool): If True then fill just the reactions index.
        """

        # Integrate out short lived processes and leave only stable particles
        # in the databases
        self._optimize_channels()

        # Go through all three cross section/response function categories
        # index contents in the ..known..variable
        self.nonel_idcs = sorted(self._nonel_tab.keys())
        self.incl_idcs = sorted(self._incl_tab.keys())
        self.incl_diff_idcs = sorted(self._incl_diff_tab.keys())
        self.reactions = {}

        for mo, da in self.incl_idcs:
            if da > 100 and get_AZN(da)[0] > get_AZN(mo)[0]:
                raise Exception(
                    'Daughter {0} heavier than mother {1}. Physics??'.format(
                        da, mo))

            if mo not in self.reactions:
                self.reactions[mo] = []
                self.known_species.append(mo)

            if (mo, da) not in self.reactions[mo]:
                # Make sure it's a unique list to avoid unnecessary loops
                self.reactions[mo].append((mo, da))
                self.known_bc_channels.append((mo, da))
                self.known_species.append(da)

        for mo, da in self.incl_diff_idcs:
            if da > 100 and get_AZN(da)[0] > get_AZN(mo)[0]:
                raise Exception(
                    'Daughter {0} heavier than mother {1}. Physics??'.format(
                        da, mo))

            if mo not in self.reactions:
                self.reactions[mo] = []
                self.known_species.append(mo)

            if (mo, da) not in self.reactions[mo]:
                # Make sure it's a unique list to avoid unnecessary loops
                self.reactions[mo].append((mo, da))
                self.known_diff_channels.append((mo, da))
                self.known_species.append(da)

        # Remove duplicates
        self.known_species = sorted(list(set(self.known_species)))
        self.known_bc_channels = sorted(list(set(self.known_bc_channels)))
        self.known_diff_channels = sorted(list(set(self.known_diff_channels)))

        # Count numbers of channels for statistics
        # Count number of incl channels for activated nuclear species
        # n_incl = np.sum([
        #     len(self.reactions[mother])
        #     for mother in self.spec_man.known_species if mother >= 100
        # ])

    def _optimize_channels(self):
        """Follows decay chains until all inclusive reactions point to
        stable final state particles.

        The "tau_dec_threshold" parameter in the config controls the
        definition of stable. Nuclei for which no the decays are
        unkown, will be forced to beta-decay until they reach a stable
        element.
        """
        # TODO: check routine, how to avoid empty channels and
        # mothers with zero nonel cross sections

        # The new dictionary that will replace _incl_tab
        new_incl_tab = {}
        new_dec_diff_tab = {}
        threshold = config["tau_dec_threshold"]

        # How to indent debug printout for recursion
        dbg_indent = lambda lev: 4 * lev * "-" + ">" if lev else ""

        info(2, "Integrating out species with lifetime smaller than",
             threshold)
        info(3,
             ("Before optimization, the number of known primaries is {0} with "
              + "in total {1} inclusive channels").format(
                  len(self._nonel_tab), len(self._incl_tab)))

        xcenters = bin_centers(self.xbins)
        xwidths = bin_widths(self.xbins)
        # The x array for the convolution matrix. Repeat xbins times
        # the x array linewise.
        # | x_0 x_1 .... x_k |
        # | x_0 x_1 .... x_k |
        # ...
        # Decay distribution will be read out on that

        xconv = np.tile(xcenters,(len(xcenters),1))[:,0]
        xwconv = np.tile(xwidths,(len(widths),1))[:,0]

        def convolve_with_decay_distribution(diff_dist, mother, daughter):
            r"""Computes the prompt decay xdist by convolving the x distribution
            of the unstable particle with the decay product distribution.

            :math:`\frac{{\rm d}N^{A\gamma \to \mu}}{{\rm d}x_j} = 
            \sum_{i=0}^{N_x}~\Delta x_i 
            \frac{{\rm d}N^{A\gamma \to \pi}}{{\rm d} x_i}~
            \frac{{\rm d}N^{\pi \to \mu}}{{\rm d} x_j}`
            """
            
            dec_dist = decs.get_decay_matrix(mother, daughter, xcenters)


        def follow_chain(first_mo, da, value, reclev):
            """Recursive function to follow decay chains until all
            final state particles are stable.ABCMeta
            
            The result is saved in two dictionaries; one for the boost
            conserving inclusive channels and the other one collects
            channels with meson or lepton decay products, which will
            need special care due to energy redistributions of these
            secondaries.
            """

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
                if (self.supports_redistributions and da < 100):
                    # If the daughter is a meson or lepton, it needs special treatment
                    # since it comes from decays with energy redistributions
                    dict_add(new_dec_diff_tab, (first_mo, da), value)
                else:
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

        # TODO: Add convolution with decay redistribution here. 
        # Loop individually over the diff channels and find convolve it with the
        # decay matrices to obtained chained distributions 

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
        # Reduce also the incl_diff_tab by removing the unknown mothers. At this stage
        # of the code, the particles with redistributions are 
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
        """Same as :func:`~cross_sections.CrossSectionBase.nonel_scale`,
        just for inclusive cross sections.
        """

        egr, cs = self.incl(mother, daughter)

        if scale == 'A':
            scale = 1. / get_AZN(mother)[0]

        return egr, scale * cs

    def response_function_scale(self, mother, daughter=None, scale='A'):
        """Same meaning as :func:`~cross_sections.CrossSectionBase.nonel_scale`,
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

        if isinstance(self._nonel_tab[mother], tuple):
            return self._nonel_tab[mother]
        else:
            return self.egrid, self._nonel_tab[mother][self._range]

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

        # If _nonel_tab contains tuples of (egrid, cs) return tuple
        # otherwise return (egrid, cs) in range defined by self.range

        if isinstance(self._incl_tab[(mother, daughter)], tuple):
            return self._incl_tab[(mother, daughter)]
        return self.egrid, self._incl_tab[(mother, daughter)][self._range]

    def incl_diff(self, mother, daughter):
        """Returns inclusive cross section.

        Inclusive differential cross section for daughter in photo-nuclear
        interactions of `mother`. Only defined, if the daughter is distributed 
        in :math:`x = E_{da} / E_{mo}`

        Args:
            mother (int): Mother nucleus(on)
            daughter (int): Daughter nucleus(on)

        Returns:
            (numpy.array): Inclusive cross section in :math:`cm^{-2}`
                           on self._egrid_tab
        """

        if (mother, daughter) not in self._incl_diff_tab:
            raise Exception(
                '({0},{1}) combination not in inclusive cross sections'.format(
                    mother, daughter))

        # If _nonel_tab contains tuples of (egrid, cs) return tuple
        # otherwise return (egrid, cs) in range defined by self.range

        if isinstance(self._incl_diff_tab[(mother, daughter)], tuple):
            return self._incl_diff_tab[(mother, daughter)]
        return self.egrid, self._incl_diff_tab[(mother,
                                                daughter)][:, self._range]

    def response_function(self, mother, daughter=None):
        """Reponse function :math:`f(y)` or :math:`g(y)` as
        defined in the note.

        Returns :math:`f(y)` or :math:`g(y)` if a daughter
        index is provided. If the inclusive channel has a redistribution,
        :math:`h(x,y)` will be returned

        Args:
            mother (int): mother nucleus(on)
            daughter (int, optional): daughter nucleus(on)

        Returns:
            (numpy.array) Reponse function on self._ygrid_tab
        """
        from scipy import integrate

        egrid, cross_section = None, None

        if daughter is not None:
            if self.is_differential(mother, daughter):
                egrid, cross_section = self.incl_diff(mother, daughter)
            elif (mother, daughter) in self.incl_idcs:
                egrid, cross_section = self.incl(mother, daughter)
            else:
                raise Exception(
                    'Unknown inclusive channel {:} -> {:} for this model'.
                    format(mother, daughter))
        else:
            egrid, cross_section = self.nonel(mother)

        ygrid = egrid[1:] / 2.

        # note that cumtrapz works also for 2d-arrays and will integrate along axis = 1
        integral = integrate.cumtrapz(egrid * cross_section, x=egrid)

        return ygrid, integral / (2 * ygrid**2)

    def _precomp_response_func(self):
        """Interpolate each response function and store interpolators.

        Uses :func:`prince.util.get_interp_object` as interpolator.
        This might result in too many knots and can be subject to
        future optimization.
        """

        info(2, 'Computing interpolators for response functions')
        info(5, 'Nonelastic response functions f(y)')
        self.resp_nonel_intp = {}
        for mother in self.nonel_idcs:
            self.resp_nonel_intp[mother] = get_interp_object(
                *self.response_function(mother))
        info(5, 'Inclusive (boost conserving) response functions g(y)')
        self.resp_incl_intp = {}
        for mother, daughter in self.incl_idcs:
            self.resp_incl_intp[(mother, daughter)] = get_interp_object(
                *self.response_function(mother, daughter))
        info(5, 'Inclusive (redistributed) response functions h(y)')
        self.resp_incl_diff_intp = {}
        for mother, daughter in self.incl_diff_idcs:
            ygr, rfunc = self.response_function(mother, daughter)
            self.resp_incl_diff_intp[(mother, daughter)] = get_2Dinterp_object(
                0.5 * (self.xbins[1:] + self.xbins[:-1]), ygr, rfunc)


class CrossSectionInterpolator(CrossSectionBase):
    """Joins and interpolates between cross section models.

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

        # Number of modls to join
        nmodels = len(model_list)
        # Energy ranges of each model
        m_ranges = []
        # grid_list = []

        self.model_refs = []
        # Construct instances of models and set ranges where they are valid
        for imo, (e_thr, mclass, margs) in enumerate(model_list):

            # Create instance of a model, passing the provided args
            csm_inst = mclass(*margs)

            if imo < nmodels - 1:
                # If not the highest energy model set both energy limits
                csm_inst.set_range(e_thr, model_list[imo + 1][0])
            else:
                # For the highest energy model, use only minimal energy
                csm_inst.set_range(e_thr)

            # If at least one of the models support redistributions, construct the
            # Interpolator class with redistributions
            if not self.supports_redistributions:
                self.supports_redistributions = csm_inst.supports_redistributions

            # Save reference
            self.model_refs.append(csm_inst)

        # Create a unique list of nonel cross sections from
        # the combination of all models
        self.nonel_idcs = sorted(
            list(set(sum([m.nonel_idcs for m in self.model_refs], []))))

        # For each ID interpolate the cross sections over entire energy range
        self._nonel_tab = {}
        for mo in self.nonel_idcs:
            self._nonel_tab[mo] = self._join_nonel(mo)

        # Create a unique list of inclusive channels from the combination
        # of all models
        self.incl_idcs_all = sorted(
            list(set(sum([m.incl_idcs for m in self.model_refs], []))))

        # Collect the channels, that need redistribution functions in a
        # separate list. Put channels that conserve boost into the normal
        # incl_idcs.
        for mo, da in self.incl_idcs_all:
            if da <= config["redist_threshold_ID"]:
                info(10, 'Daughter has redistribution function', mo, da)
                self.incl_diff_idcs.append((mo, da))
            else:
                info(10, 'Mother and daughter conserve boost', mo, da)
                self.incl_idcs.append((mo, da))

        # Join the boost conserving channels
        self._incl_tab = {}
        for mo, da in self.incl_idcs:
            self._incl_tab[(mo, da)] = self._join_incl(mo, da)

        # Join the redistribution channels
        self._incl_diff_tab = {}
        for mo, da in self.incl_diff_idcs:
            self._incl_diff_tab[(mo, da)] = self._join_incl_diff(mo, da)

        self._optimize_and_generate_index()

        # now also precompute the response function
        self._precomp_response_func()

    def _join_nonel(self, mother):
        """Returns the non-elastic cross section of the joined models.
        """

        info(5, 'Joining nonelastic cross sections for', mother)

        egrid = []
        nonel = []
        for model in self.model_refs:
            e, csec = model.nonel(mother)
            egrid.append(e)
            nonel.append(csec)

        return np.concatenate(egrid), np.concatenate(nonel)

    def _join_incl(self, mother, daughter):
        """Returns joined incl cross sections."""

        info(5, 'Joining inclusive cross sections for channel', (mother,
                                                                 daughter))
        egrid = []
        incl = []
        for model in self.model_refs:
            e, csec = model.incl(mother, daughter)
            egrid.append(e)
            incl.append(csec)

        return np.concatenate(egrid), np.concatenate(incl)

    def _join_incl_diff(self, mother, daughter):
        """Returns joined incl diff cross sections.

        The function assumes the same `x` bins for all models.
        """

        info(5, 'Joining inclusive differential cross sections for channel',
             (mother, daughter))

        egrid = []
        incl_diff = []

        # Get an x grid from a model which supports it
        for model in self.model_refs:
            if model.supports_redistributions:
                self.xbins = model.xbins
                break
        if self.xbins is None:
            raise Exception('Redistributions requested but none of the ' +
                            'models supports it')

        for model in self.model_refs:
            egr, csec = None, None
            if model.is_differential(mother, daughter):
                egr, xb, csec = model.incl_diff(mother, daughter)
                if config["debug_level"] > 1:
                    assert np.alltrue(
                        self.xbins == xb), 'Unequal x bins. Aborting...'
                info(10, model.mname, mother, daughter, 'is differential.')

            elif (mother, daughter) in model.incl_idcs:
                # try to use incl and extend by zeros
                egr, csec_1d = model.incl(mother, daughter)
                # no x-distribution given, so x = 1
                csec = np.zeros((self.xbins.shape[0] - 1, csec_1d.shape[0]))
                csec[-1, :] = csec_1d
                info(10, model.mname, mother, daughter,
                     'not differential, x=1.')
            else:
                raise Exception('Why does this happen? Should not...', mother,
                                daughter)

            egrid.append(egr)
            incl_diff.append(csec)

        return np.concatenate(egrid), np.concatenate(incl_diff, axis=1)


class SophiaSuperposition(CrossSectionBase):
    """ Cross sections generated using the Sophia event generator for protons and neutrons.
    Includes redistribution functions into secondaries
    """

    def __init__(self, *args, **kwargs):
        CrossSectionBase.__init__(self)
        # Tell the interpolator that this model contains the necessary features
        # for energy redistribution functions
        self.supports_redistributions = True
        self._load()

    def _load(self):
        info(2, "Loading SOPHIA cross sections from file.")

        # load the crossection from file
        self._egrid_tab, self.cs_proton_grid, self.cs_neutron_grid = \
        load_or_convert_array(
            'sophia_crosssec', delimiter=',', unpack=True)

        epsr_grid, self.xbins, self.redist_proton, self.redist_neutron = np.load(
            join(config["data_dir"], config["redist_fname"]))

        # check if crosssection and redistribution are defined on the same grid,
        # other wise interpolate crosssection
        if epsr_grid.shape != self._egrid_tab.shape or np.any(
                epsr_grid != self._egrid_tab):
            info(1, "Adjusting cross section by interpolation.")
            self.cs_proton_grid = np.interp(epsr_grid, self._egrid_tab,
                                            self.cs_proton_grid)
            self.cs_neutron_grid = np.interp(epsr_grid, self._egrid_tab,
                                             self.cs_neutron_grid)
            self._egrid_tab = epsr_grid

        # sophia crossections are in mubarn; convert here to cm^2
        self.cs_proton_grid *= 1e-30
        self.cs_neutron_grid *= 1e-30

        # set up inclusive differential channels for protons and neutron
        for da in sorted(self.redist_proton):
            self.incl_diff_idcs.append((101, da))
        for da in sorted(self.redist_neutron):
            self.incl_diff_idcs.append((100, da))

        # For more convenient generation of trivial redistribution matrices when joining
        self.redist_shape = (self.xbins.shape[0], self._egrid_tab.shape[0])

        self.set_range()

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
            raise Exception('Boost conserving cross section called ' +
                            'for redistributed particle')

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
        if daughter in self.redist_proton:
            cgrid = Z * self.cs_proton_grid
            csec_diff = self.redist_proton[daughter].T * cgrid

        if daughter in self.redist_neutron:
            cgrid = N * self.cs_neutron_grid
            if np.any(csec_diff):
                csec_diff += self.redist_neutron[daughter].T * cgrid
            else:
                csec_diff = self.redist_neutron[daughter].T * cgrid

        return self.egrid, self.xbins, csec_diff[:, self._range]


class NeucosmaFileInterface(CrossSectionBase):
    """Tabulated disintegration cross sections from Peanut or TALYS.
    Data available from 1 MeV to 1 GeV"""

    def __init__(self, model_prefix='peanut', *args, **kwargs):
        CrossSectionBase.__init__(self)
        self.supports_redistributions = False
        self._load(model_prefix)
        self._optimize_and_generate_index()

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

        # mo = mother, da = daughter
        _incl_tab = {}
        for (mo, da), csgrid in zip(pids_incl, incl_raw):
            _incl_tab[mo, da] = csgrid

        self._egrid_tab = egrid
        self._nonel_tab = _nonel_tab
        self._incl_tab = _incl_tab
        # Set initial range to whole egrid
        self.set_range()
        info(2, "Finished initialization")


if __name__ == "__main__":
    pass
