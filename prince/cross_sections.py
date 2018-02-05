"""The module contains everything to handle cross section interfaces."""

from abc import ABCMeta, abstractmethod
from os.path import isfile, join

import numpy as np

from prince.util import *
import prince.decays as decs
from prince_config import config, spec_data

# ToDo:
# - CompositeCrossSection._join_incl_diff() does currently not work properly for inclusive differential crossections
#     - the class combines the channel indices from all models,
#       however sophia does not provide these, and still introduces indices for lighter particles


class CrossSectionBase(object):
    """Base class for cross section interfaces to tabulated models.

    The class is abstract and it is not inteded to be instantiated.
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        # Tuple, defining min and max energy cuts on the grid
        self._range = None
        # Energy grid, as defined in files
        self._egrid_tab = None
        # Dictionary of nonel. cross sections on egrid, indexed by (mother)
        self._nonel_tab = {}
        # Dictionary of incl. cross sections on egrid, indexed by (mother, daughter)
        self._incl_tab = {}
        # Dictionary of incl. diff. cross sections on egrid, indexed by (mother, daughter)
        self._incl_diff_tab = {}
        # List of available mothers for nonel cross sections
        self.nonel_idcs = []
        # List of available (mothers,daughter) reactions in incl. cross sections
        self.incl_idcs = []
        # List of available (mothers,daughter) reactions in incl. diff. cross sections
        self.incl_diff_idcs = []
        # Common grid in x (the redistribution variable)
        self.xbins = None

        # Flag, which tells if the model supports secondary redistributions
        if not hasattr(self, 'supports_redistributions'):
            self.supports_redistributions = None  # JH: to differ from explicitly set False
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

    @property
    def xcenters(self):
        """Returns centers of the grid in x.

        Returns:
            (numpy.array): x grid
        """

        return 0.5 * (self.xbins[1:] + self.xbins[:-1])
    
    @property
    def xwidths(self):
        """Returns bin widths of the grid in x.

        Returns:
            (numpy.array): x widths
        """

        return self.xbins[1:] - self.xbins[:-1]

    @property
    def resp(self):
        """Return ResponseFunction corresponding to this cross section
        Will only create the Response function once. 
        """
        if not hasattr(self, '_resp'):
            info(2, 'First Call, creating instance of ResponseFunction now')
            self._resp = ResponseFunction(self)
        return self._resp

    def is_differential(self, mother, daughter):
        """Returns true if the model supports redistributions and requested
        mother/daughter combination should return non-zero redistribution matrices.

        Args:
            mother (bool): Neucosma ID of mother particle
            daughter (bool): Neucosma ID of daughter particle

        Returns:
            (bool): ``True`` if the model has this particular redistribution function
        """
        # info(10, mother, daughter, " asking for redist")
        # if not self.supports_redistributions:
        #     info(10, mother, daughter, " model doesn't support redist")
        #     return False
        if (daughter <= config["redist_threshold_ID"] or
            (mother, daughter) in self.incl_diff_idcs):
            info(60, 'Daughter requires redistribution.', mother, daughter)
            return True
        info(60, 'Daughter conserves boost.', mother, daughter)
        return False

    def _update_indices(self):
        """Updates the list of indices according to entries in the
        _tab variables"""

        self.nonel_idcs = sorted(self._nonel_tab.keys())
        self.incl_idcs = sorted(self._incl_tab.keys())
        self.incl_diff_idcs = sorted(self._incl_diff_tab.keys())

    def _optimize_and_generate_index(self):
        """Construct a list of mothers and (mother, daughter) indices.

        Args:
            just_reactions (bool): If True then fill just the reactions index.
        """

        # Integrate out short lived processes and leave only stable particles
        # in the databases
        self._reduce_channels()

        # Go through all three cross section categories
        # index contents in the ..known..variable
        self.reactions = {}

        self._update_indices()

        for mo, da in self.incl_idcs:
            if da >= 100 and get_AZN(da)[0] > get_AZN(mo)[0]:
                raise Exception(
                    'Daughter {0} heavier than mother {1}. Physics??'.format(
                        da, mo))

            if mo not in self.reactions:
                self.reactions[mo] = []
                self.known_species.append(mo)

            if (mo, da) not in self.reactions[mo]:
                # Make sure it's a unique list
                self.reactions[mo].append((mo, da))
            if self.is_differential(mo, da):
                # Move the distributions which are expected to be differential
                # to _incl_diff_tab
                self._incl_diff_tab[(
                    mo,
                    da)] = self._arange_on_xgrid(self._incl_tab.pop((mo, da)))
                info(10, "Channel {0} -> {1} forced to be differential.")
            else:
                self.known_bc_channels.append((mo, da))
                self.known_species.append(da)

        for mo, da in self._incl_diff_tab.keys():
            if da >= 100 and get_AZN(da)[0] > get_AZN(mo)[0]:
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

        for sp in self.known_species:
            if sp >= 100 and (sp, sp) not in self.known_diff_channels:
                self.known_bc_channels.append((mo, mo))
            if (mo, mo) not in self.reactions[mo]:
                self.reactions[mo].append((mo, mo))

        # Make sure the indices are up to date
        self._update_indices()

        # Count numbers of channels for statistics
        # Count number of incl channels for activated nuclear species
        # n_incl = np.sum([
        #     len(self.reactions[mother])
        #     for mother in self.spec_man.known_species if mother >= 100
        # ])

    def _reduce_channels(self):
        """Follows decay chains until all inclusive reactions point to
        stable final state particles.

        The "tau_dec_threshold" parameter in the config controls the
        definition of stable. Unstable nuclei for which no decay channels
        are known, will be forced to beta-decay until they reach a stable
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

        if self.xbins is None:
            info(
                4,
                'Model does not provide a native xbins. Assuming JH special sophia',
                'binning.')
            self.xbins = SophiaSuperposition().xbins

        bc = self.xcenters
        bw = bin_widths(self.xbins)
        # The x_mu/x_pi grid
        # dec_grid = np.fromfunction(
        #     lambda j, i: 10**(np.log10(bc[1] / bc[0]) * (j - i)), (len(bc),
        #                                                            len(bc)))
        
        dec_grid = np.outer(bc,1/bc)
        
        dec_bins = np.outer(self.xbins,1/bc)
        dec_bins_lower = dec_bins[:-1]
        dec_bins_upper = dec_bins[1:]

        # dec_grid[dec_grid > 1.] *= 0.
        # The differential element dx_mu/x_pi
        int_scale = np.tile(bw / bc, (len(bc), 1))

        def convolve_with_decay_distribution(diff_dist, mother, daughter,
                                             branching_ratio):
            r"""Computes the prompt decay xdist by convolving the x distribution
            of the unstable particle with the decay product distribution.

            :math:`\frac{{\rm d}N^{A\gamma \to \mu}}{{\rm d}x_j} = 
            \sum_{i=0}^{N_x}~\Delta x_i 
            \frac{{\rm d}N^{A\gamma \to \pi}}{{\rm d} x_i}~
            \frac{{\rm d}N^{\pi \to \mu}}{{\rm d} x_j}`
            """
            # dec_dist = int_scale * decs.get_decay_matrix(
            #     mother, daughter, dec_grid)
            dec_dist = int_scale * decs.get_decay_matrix_bin_average(
                mother, daughter, dec_bins_lower, dec_bins_upper)
                
            info(20, 'convolving with decay dist', mother, daughter)
            # Handle the case where table entry is (energy_grid, matrix)
            if not isinstance(diff_dist, tuple):
                return branching_ratio * dec_dist.dot(diff_dist)
            else:
                return diff_dist[0], branching_ratio * dec_dist.dot(
                    diff_dist[1])

        def follow_chain(first_mo, da, csection, reclev):
            """Recursive function to follow decay chains until all
            final state particles are stable.
            
            The result is saved in two dictionaries; one for the boost
            conserving inclusive channels and the other one collects
            channels with meson or lepton decay products, which will
            need special care due to energy redistributions of these
            secondaries.
            """

            info(10, dbg_indent(reclev), 'Entering with', first_mo, da)

            if da not in spec_data:
                info(
                    3,
                    dbg_indent(reclev),
                    'daughter {0} unknown, forcing beta decay. Not Implemented yet!!'.
                    format(da))
                return

            # Daughter is stable. Add it to the new dictionary and terminate
            # recursion
            if spec_data[da]["lifetime"] >= threshold:
                if self.is_differential(None, da):
                    # If the daughter is a meson or lepton, use the dictionary for
                    # differential channels
                    info(
                        20,
                        dbg_indent(reclev),
                        'daughter {0} stable and differential. Adding to ({1}, {2})'.
                        format(da, first_mo, da))
                    dict_add(new_dec_diff_tab, (first_mo, da), csection)
                else:
                    info(20,
                         dbg_indent(reclev),
                         'daughter {0} stable. Adding to ({1}, {2})'.format(
                             da, first_mo, da))
                    dict_add(new_incl_tab, (first_mo, da), csection)
                return

            # ..otherwise follow decay products of this daughter, tracking the
            # original mother particle (first_mo). The cross section (csection) is
            # reduced by the branching ratio (br) of this particular channel
            for br, daughters in spec_data[da]["branchings"]:
                info(10,
                     dbg_indent(reclev),
                     ("{3} -> {0:4d} -> {2:4.2f}: {1}").format(
                         da, ", ".join(map(str, daughters)), br, first_mo))

                for chained_daughter in daughters:
                    # Follow each secondary and increment the recursion level by one
                    if self.is_differential(None, chained_daughter):
                        info(10, 'daughter', chained_daughter, 'of', da,
                             'is differential')
                        follow_chain(first_mo, chained_daughter,
                                     convolve_with_decay_distribution(
                                         self._arange_on_xgrid(csection), da,
                                         chained_daughter, br), reclev + 1)
                    else:
                        follow_chain(first_mo, chained_daughter, br * csection,
                                     reclev + 1)

        # Remove all unstable particles from the dictionaries
        for mother in sorted(self._nonel_tab.keys()):
            if mother not in spec_data or spec_data[mother]["lifetime"] < threshold:
                info(
                    20,
                    "Primary species {0} does not fulfill stability criteria.".
                    format(mother))
                _ = self._nonel_tab.pop(mother)
        # Only stable (interacting) mother particles are left
        self._update_indices()

        for (mother, daughter) in self.incl_idcs:

            if mother not in self.nonel_idcs:
                info(30,
                     "Removing {0}/{1} from incl, since mother not stable ".
                     format(mother, daughter))
                _ = self._incl_tab.pop((mother, daughter))

            elif self.is_differential(mother, daughter):
                # Move the distributions which are expected to be differential
                # to _incl_diff_tab
                self._incl_diff_tab[(
                    mother, daughter)] = self._arange_on_xgrid(
                        self._incl_tab.pop((mother, daughter)))

        self._update_indices()

        for (mother, daughter) in self.incl_diff_idcs:

            if mother not in self.nonel_idcs:
                info(
                    30,
                    "Removing {0}/{1} from diff incl, since mother not stable ".
                    format(mother, daughter))
                _ = self._incl_diff_tab.pop((mother, daughter))

        self._update_indices()

        # Launch the reduction for each inclusive channel
        for (mo, da), value in self._incl_tab.items():
            follow_chain(mo, da, value, 0)

        for (mo, da), value in self._incl_diff_tab.items():
            follow_chain(mo, da, value, 0)

        # Overwrite the old incl dictionary
        self._incl_tab = new_incl_tab
        # Overwrite the old incl_diff dictionary
        self._incl_diff_tab = new_dec_diff_tab
        # Reduce also the incl_diff_tab by removing the unknown mothers. At this stage
        # of the code, the particles with redistributions are
        info(3,
             ("After optimization, the number of known primaries is {0} with "
              + "in total {1} inclusive channels").format(
                  len(self._nonel_tab),
                  len(self._incl_tab) + len(self._incl_diff_tab)))

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

        egr, csection = self.nonel(mother)

        if scale == 'A':
            scale = 1. / get_AZN(mother)[0]

        return egr, scale * csection

    def incl_scale(self, mother, daughter, scale='A'):
        """Same as :func:`~cross_sections.CrossSectionBase.nonel_scale`,
        just for inclusive cross sections.
        """

        egr, csection = self.incl(mother, daughter)

        if scale == 'A':
            scale = 1. / get_AZN(mother)[0]

        return egr, scale * csection

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
                self.__class__.__name__ +
                '({0},{1}) combination not in inclusive differential cross sections'.
                format(mother, daughter))

        # If _nonel_tab contains tuples of (egrid, cs) return tuple
        # otherwise return (egrid, cs) in range defined by self.range

        if isinstance(self._incl_diff_tab[(mother, daughter)], tuple):
            return self._incl_diff_tab[(mother, daughter)]
        return self.egrid, self._incl_diff_tab[(mother,
                                                daughter)][:, self._range]

    def _arange_on_xgrid(self, incl_cs):
        """Returns the inclusive cross section on an xgrid at x=1."""

        egr, cs = None, None

        if isinstance(incl_cs, tuple):
            egr, cs = incl_cs
        else:
            cs = incl_cs

        nxbins = len(self.xbins) - 1
        if len(cs.shape) > 1 and cs.shape[0] != nxbins:
            raise Exception(
                'One dimensional cross section expected, instead got',
                cs.shape, '\n', cs)
        elif len(cs.shape) == 2 and cs.shape[0] == nxbins:
            info(20, 'Supplied 2D distribution seems to be distributed in x.')
            if isinstance(incl_cs, tuple):
                return egr, cs
            return cs

        csec = np.zeros((nxbins, cs.shape[0]))
        csec[-1, :] = cs / self.xwidths[-1]
        # print 'Warning! Test division by bin width here!'
        if isinstance(incl_cs, tuple):
            return egr, csec
        return csec


class CompositeCrossSection(CrossSectionBase):
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
        # of all models, no matter if diff or not. The rearrangement
        # is performed in the next steps
        self.incl_idcs_all = sorted(
            list(set(sum([m.incl_idcs for m in self.model_refs], []))))
        self.incl_idcs_all += sorted(
            list(set(sum([m.incl_diff_idcs for m in self.model_refs], []))))

        # Collect the channels, that need redistribution functions in a
        # separate list. Put channels that conserve boost into the normal
        # incl_idcs.
        self.incl_diff_idcs = []
        self.incl_idcs = []
        for mother, daughter in self.incl_idcs_all:
            if self.is_differential(mother, daughter):
                info(10, 'Daughter has redistribution function', mother,
                     daughter)
                self.incl_diff_idcs.append((mother, daughter))
            else:
                info(10, 'Mother and daughter conserve boost', mother,
                     daughter)
                self.incl_idcs.append((mother, daughter))

        # Join the boost conserving channels
        self._incl_tab = {}
        for mother, daughter in self.incl_idcs:
            self._incl_tab[(mother, daughter)] = self._join_incl(
                mother, daughter)

        # Join the redistribution channels
        self._incl_diff_tab = {}
        for mother, daughter in self.incl_diff_idcs:
            self._incl_diff_tab[(mother, daughter)] = self._join_incl_diff(
                mother, daughter)

        self._update_indices()
        self._optimize_and_generate_index()

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
            if config["debug_level"] > 1:
                if not np.allclose(self.xbins, model.xbins):
                    raise Exception('Unequal x bins. Aborting...',
                                    self.xbins.shape, model.xbins)
            if (mother, daughter) in model.incl_diff_idcs:
                egr, csec = model.incl_diff(mother, daughter)
                info(10, model.mname, mother, daughter, 'is differential.')

            elif (mother, daughter) in model.incl_idcs:
                # try to use incl and extend by zeros
                egr, csec_1d = model.incl(mother, daughter)
                print mother, daughter, csec_1d.shape
                # no x-distribution given, so x = 1
                csec = self._arange_on_xgrid(csec_1d)
                info(1, model.mname, mother, daughter,
                     'not differential, x=1.')
            else:
                info(5, 'Model', model.mname, 'does not provide cross',
                     'sections for channel {0}/{1}. Setting to zero.'.format(
                         mother, daughter))
                # Tried with reduced energy grids to save memory, but
                # matrix addition in decay chains becomes untrasparent
                # egr = np.array((model.egrid[0], model.egrid[-1]))
                # csec = np.zeros((len(self.xbins) - 1, 2))
                egr = model.egrid
                csec = np.zeros((len(self.xbins) - 1, model.egrid.size))

            egrid.append(egr)
            incl_diff.append(csec)

        return np.concatenate(egrid), np.concatenate(incl_diff, axis=1)


class SophiaSuperposition(CrossSectionBase):
    """ Cross sections generated using the Sophia event generator for protons and neutrons.
    Includes redistribution functions into secondaries
    """

    def __init__(self, *args, **kwargs):
        # Tell the interpolator that this model contains the necessary features
        # for energy redistribution functions
        self.supports_redistributions = True
        CrossSectionBase.__init__(self)
        self._load()

    def _load(self):
        info(2, "Loading SOPHIA cross sections from file.")
        info(5, "File used:",join(config["data_dir"], config["redist_fname"]))
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

        return self.egrid, csec_diff[:, self._range]


class TabulatedCrossSection(CrossSectionBase):
    """Tabulated disintegration cross sections from Peanut or TALYS.
    Data available from 1 MeV to 1 GeV"""

    def __init__(self, model_prefix='peanut_IAS', max_mass=None, *args, **kwargs):
        self.supports_redistributions = False
        if max_mass is None:
            self.max_mass = config["max_mass"]
        CrossSectionBase.__init__(self)
        self._load(model_prefix)
        self._optimize_and_generate_index()

    def _load(self, model_prefix):

        cspath = config['data_dir']

        info(2, "Load tabulated cross sections")
        # The energy grid is given in MeV, so we convert to GeV
        egrid = load_or_convert_array(
            model_prefix + "_egrid.dat", dtype='float') * 1e-3
        info(2, "Egrid loading finished")

        # Load tables from files
        _nonel_tab = load_or_convert_array(model_prefix + "_nonel.dat")
        _incl_tab = load_or_convert_array(model_prefix + "_incl_i_j.dat")

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
            # TODO: dirty workarround, pass max mass to config
            # to delete heavier particle from crosssection
            if get_AZN(pid)[0] > self.max_mass:
                continue
            _nonel_tab[pid] = csgrid

        # If proton and neutron cross sections are not in contained
        # in the files, set them to 0. Needed for TALYS and CRPropa2
        for pid in [101, 100]:
            if pid not in _nonel_tab:
                _nonel_tab[pid] = np.zeros_like(egrid)

        # mo = mother, da = daughter
        _incl_tab = {}
        for (mo, da), csgrid in zip(pids_incl, incl_raw):
            # TODO: dirty workarround, pass max mass to config
            # to delete heavier particle from crosssection
            if get_AZN(mo)[0] > self.max_mass:
                continue
            _incl_tab[mo, da] = csgrid

        self._egrid_tab = egrid
        self._nonel_tab = _nonel_tab
        self._incl_tab = _incl_tab
        # Set initial range to whole egrid
        self.set_range()
        info(2, "Finished initialization")

class NEUCOSMACrossSection(CrossSectionBase):
    """Class to import cross sections from a NEUCOSMA file
    """
    def __init__(self, NEUCOSMA_filename = '160513_XDIS_PSB-SS_syst.dat', max_mass=None, *args, **kwargs):
        if max_mass is None:
            self.max_mass = config["max_mass"]
        CrossSectionBase.__init__(self)

        import os.path as path
        filepath = path.join(config['raw_data_dir'],'cross_sections_NeuCosmA',NEUCOSMA_filename)
        self._load_NEUCOSMA_file(filepath)
        self._optimize_and_generate_index()

    @property
    def resp_data(self):
        """Return ResponseFunction corresponding to this cross section
        but based on data loaded from NEUCOSMA file
        Will only create the Response function once.
        """
        if not hasattr(self, '_resp_data'):
            info(2, 'First Call, creating instance of ResponseFunctionLoaded now')
            self._resp_data = ResponseFunctionLoaded(self)
        return self._resp_data

    def _load_NEUCOSMA_file(self, filename):
        """Loads a txt file with format as define in the internal note.

        Args:
            filename (string): name of the ile including path

        Returns:
            (filename1, filename2) Two pickled dictionaries saved on the
            same directory as `filename` which contain:
            filename1: a dictionary indexed (mother, daughter) where the
            the g function and the multiplicity are stored.
            filename2: a dictionary indexed (mother) where the f function
            and the total inelasticcross section are stored.
        """

        # The file format is as following (by column):
        # 1. parent id
        # 2. daughter id
        # 3. systematic flag (currently ignored)
        # 4. log10(E [GeV]) (E or eps_r or y depending on corresp. column)
        # 5. g_ij(y) [mubarn = 10^-30 cm^2]
        # 6. M_ij(eps_r)
        # 7. f_i(y) [mubarn = 10^-30 cm^2]
        # 8. sigma_i(eps_r) [mubarn = 10^-30 cm^2]
    

        with open(filename) as f:
            text_data = f.readlines()

        # We need the following: sigma_i, sigma_ij = M_ij * sigma_i

        mo, da = (int(l) for l in text_data[0].split()[:2])
        cs_nonel, cs_incl = {}, {}
        e, g, mu, f, cs = (), (), (), (), ()

        neucosma_data = {}
        neucosma_data['f'] = {}
        neucosma_data['g'] = {}
        neucosma_data['m'] = {}

        for line in text_data:
            m, d, _, e_k, g_ijk, m_ijk, f_ik, cs_ik = line.split()

            m, d = int(m), int(d)
            if d == da:
                e += (float(e_k), )
                g += (float(g_ijk), )
                mu += (float(m_ijk), )
                f += (float(f_ik), )
                cs += (float(cs_ik), )
            else:
                neucosma_data['g'][mo, da] = np.array(g)  # stored in mubarn
                neucosma_data['m'][mo, da] = np.array(mu)
                # Factor 1e30 below, for conversion to cm-2
                cs_incl[mo, da] = np.array(mu) * np.array(cs) * 1e-30
                # reset values of lists
                da = d
                if m != mo:
                    neucosma_data['f'][mo] = np.array(f)
                    cs_nonel[mo] = np.array(cs) * 1e-30   # conversion to cm-2
                    mo = m
                e, g, mu, f, cs = (float(e_k), ), (float(g_ijk), ), \
                                  (float(m_ijk), ), (float(f_ik), ),\
                                  (float(cs_ik), )
        
        # Do not forget the last mother:
        neucosma_data['f'][mo] = np.array(f)
        cs_nonel[mo] = np.array(cs) * 1e-30   # conversion to cm-2

        neucosma_data['g'][mo, da] = np.array(g)  # stored in mubarn
        neucosma_data['m'][mo, da] = np.array(mu)
        # Factor 1e30 below, for conversion to cm-2
        cs_incl[mo, da] = np.array(mu) * np.array(cs) * 1e-30

        # If proton and neutron cross sections are not in contained
        # in the files, set them to 0. Needed for TALYS and CRPropa2 and PSB
        for pid in [101, 100]:
            if pid not in cs_nonel:
                cs_nonel[pid] = np.zeros_like(e)

        print 'known species after loading NeuCosmA file:'
        print np.sort(cs_nonel.keys())

        # storing f,g,m data from NEUCOSMA file
        self._NEUCOSMA_data = neucosma_data
        self._egrid_tab = 10**np.array(e)
        self._nonel_tab = cs_nonel
        self._incl_tab = cs_incl
        # Set initial range to whole egrid
        self.set_range()
        # info(2, "Finished initialization")

class ResponseFunction(object):
    """Redistribution Function based on Crossection model
    """

    def __init__(self, cross_section):
        self.cross_section = cross_section

        self.xcenters = cross_section.xcenters

        # Copy indices from CrossSection Model
        self.nonel_idcs = cross_section.nonel_idcs
        self.incl_idcs = cross_section.incl_idcs
        self.incl_diff_idcs = cross_section.incl_diff_idcs

        # Dictionary of reponse function interpolators
        self.nonel_intp = {}
        self.incl_intp = {}
        self.incl_diff_intp = {}

        self._precompute_interpolators()

    # forward is_differential() to CrossSectionBase
    # that might break in the future...
    def is_differential(self, mother, daughter):
        return CrossSectionBase.is_differential(self, mother, daughter)

    def get_full(self, mother, daughter, ygrid, xgrid=None):
        """Return the full response function :math:`f(y) + g(y) + h(x,y)`
        on the grid that is provided. xgrid is ignored if `h(x,y)` not in the channel.
        """
        if xgrid is not None and ygrid.shape != xgrid.shape:
            raise Exception('ygrid and xgrid do not have the same shape!!')
        if get_AZN(mother)[0] < get_AZN(daughter)[0]:
            info(
                3,
                'WARNING: channel {:} -> {:} with daughter heavier than mother!'.
                format(mother, daughter))

        res = np.zeros(ygrid.shape)

        if (mother, daughter) in self.incl_intp:
            res += self.incl_intp[(mother, daughter)](ygrid)
        elif (mother, daughter) in self.incl_diff_intp:
            #incl_diff_res = self.incl_diff_intp[(mother, daughter)](
            #    xgrid, ygrid, grid=False)
            #if mother == 101:
            #    incl_diff_res = np.where(xgrid < 0.9, incl_diff_res, 0.)
            #res += incl_diff_res
            #if not(mother == daughter):
                res += self.incl_diff_intp[(mother, daughter)].inteval(
                    xgrid, ygrid, grid=False)

        if mother == daughter and mother in self.nonel_intp:
            # nonel cross section leads to absorption, therefore the minus
            if xgrid is None:
                res -= self.nonel_intp[mother](ygrid)
            else:
                diagonal = xgrid == 1.
                res[diagonal] -= self.nonel_intp[mother](ygrid[diagonal])

        return res

    def get_channel(self, mother, daughter=None):
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

        cs_model = self.cross_section
        egrid, cross_section = None, None

        if daughter is not None:
            if (mother, daughter) in self.incl_diff_idcs:
                egrid, cross_section = cs_model.incl_diff(mother, daughter)
            elif (mother, daughter) in self.incl_idcs:
                egrid, cross_section = cs_model.incl(mother, daughter)
            else:
                raise Exception(
                    'Unknown inclusive channel {:} -> {:} for this model'.
                    format(mother, daughter))
        else:
            egrid, cross_section = cs_model.nonel(mother)

    # note that cumtrapz works also for 2d-arrays and will integrate along axis = 1
        integral = integrate.cumtrapz(egrid * cross_section, x=egrid)
        ygrid = egrid[1:] / 2.

        return ygrid, integral / (2 * ygrid**2)

    def get_channel_scale(self, mother, daughter=None, scale='A'):
        """Returns the reponse function scaled by `scale`.

        Convenience funtion for plotting, where it is important to
        compare the cross section/response function per nucleon.

        Args:
            mother (int): Mother nucleus(on)
            scale (float): If `A` then nonel/A is returned, otherwise
                           scale can be any float.

        Returns:
            (numpy.array, numpy.array): Tuple of Energy grid in GeV,
                                        scale * inclusive cross section
                                        in :math:`cm^{-2}`
        """

        ygr, cs = self.get_channel(mother, daughter)

        if scale == 'A':
            scale = 1. / get_AZN(mother)[0]

        return ygr, scale * cs

    def _precompute_interpolators(self):
        """Interpolate each response function and store interpolators.

        Uses :func:`prince.util.get_interp_object` as interpolator.
        This might result in too many knots and can be subject to
        future optimization.
        """

        info(2, 'Computing interpolators for response functions')

        info(5, 'Nonelastic response functions f(y)')
        self.nonel_intp = {}
        for mother in self.nonel_idcs:
            self.nonel_intp[mother] = get_interp_object(
                *self.get_channel(mother))

        info(5, 'Inclusive (boost conserving) response functions g(y)')
        self.incl_intp = {}
        for mother, daughter in self.incl_idcs:
            self.incl_intp[(mother, daughter)] = get_interp_object(
                *self.get_channel(mother, daughter))

        info(5, 'Inclusive (redistributed) response functions h(y)')
        self.incl_diff_intp = {}
        for mother, daughter in self.incl_diff_idcs:
            ygr, rfunc = self.get_channel(mother, daughter)
            self.incl_diff_intp[(mother, daughter)] = get_2Dinterp_object(
                self.xcenters, ygr, rfunc, self.cross_section.xbins)

class ResponseFunctionLoaded(ResponseFunction):
    """Redistribution Function based on Crossection model.
    Operates on values of (f, g, h) loaded from  a NEAUCOSMA file
    """

    def __init__(self, cross_section):
        ResponseFunction.__init__(self, cross_section)

    def _precompute_interpolators(self):
        """Interpolate each response function and store interpolators.

        Uses :func:`prince.util.get_interp_object` as interpolator.
        This might result in too many knots and can be subject to
        future optimization.
        """
        cs_model = self.cross_section

        info(2, 'Computing interpolators for response functions')

        info(5, 'Nonelastic response functions f(y)')
        self.nonel_intp = {}
        for mother in self.nonel_idcs:
            self.nonel_intp[mother] = get_interp_object(
                cs_model.egrid, cs_model._NEUCOSMA_data['f'][mother])

        info(5, 'Inclusive (boost conserving) response functions g(y)')
        self.incl_intp = {}
        for mother, daughter in self.incl_idcs:
            self.incl_intp[(mother, daughter)] = get_interp_object(
                cs_model.egrid, cs_model._NEUCOSMA_data['g'][mother, daughter])

        info(5, 'Inclusive (redistributed) response functions h(y): not implemented')
        self.incl_diff_intp = {}
        # for mother, daughter in self.incl_diff_idcs:
        #     ygr, rfunc = self.get_channel(mother, daughter)
        #     self.incl_diff_intp[(mother, daughter)] = get_2Dinterp_object(
        #         self.xcenters, ygr, rfunc)



if __name__ == "__main__":
    pass
