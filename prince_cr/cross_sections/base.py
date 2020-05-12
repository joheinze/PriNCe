"""The module contains everything to handle cross section interfaces."""

from abc import ABCMeta, abstractmethod
from os.path import isfile, join

import numpy as np

import prince_cr.decays as decs
from prince_cr.data import spec_data
from prince_cr.util import (
    get_2Dinterp_object, get_interp_object, info, bin_widths, get_AZN)
import prince_cr.config as config

class CrossSectionBase(object, metaclass=ABCMeta):
    """Base class for cross section interfaces to tabulated models.

    The class is abstract and it is not inteded to be instantiated.

    Child Classes either define the tables:
        self._egrid_tab
        self._nonel_tab
        self._incl_tab
        self._incl_diff

    Or directly reimplememnt the functions
        nonel(self, mother, daughter)
        incl(self, mother, daughter)
        incl_diff(self, mother, daughter)

    The flag self.supports_redistributions = True/False should be set
    To tell the class to include/ignore incl_diff
    """
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

        info(5, "Setting range to {0:3.2e} - {1:3.2e}".format(e_min, e_max))
        self._range = np.where((self._egrid_tab >= e_min)
                               & (self._egrid_tab <= e_max))[0]
        info(
            2, "Range set to {0:3.2e} - {1:3.2e}".format(
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
            from .response import ResponseFunction
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
        if (daughter <= config.redist_threshold_ID
                or (mother, daughter) in self.incl_diff_idcs):
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

    def generate_incl_channels(self, mo_indices):
        """Generates indices for all allowed channels given mo_indices
            Note: By default this returns an empty list,
                  meant to be overwritten in cases where 
                  the child class needs to dynamically generate indices
        
        Args:
            mo_indices (list of ints): list of indices for mother nuclei

        Returns:
           Returns:
            list of tuples: list of allowed channels given as (mo_idx, da_idx)
        """
        incl_channels = []

        return incl_channels

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
                self._incl_diff_tab[(mo, da)] = self._arange_on_xgrid(
                    self._incl_tab.pop((mo, da)))
                info(10, "Channel {0} -> {1} forced to be differential.")
            else:
                self.known_bc_channels.append((mo, da))
                self.known_species.append(da)

        for mo, da in list(self._incl_diff_tab.keys()):
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
        from prince_cr.util import AdditiveDictionary
        # TODO: check routine, how to avoid empty channels and
        # mothers with zero nonel cross sections

        # The new dictionary that will replace _incl_tab
        new_incl_tab = AdditiveDictionary()
        new_dec_diff_tab = AdditiveDictionary()

        threshold = config.tau_dec_threshold

        # How to indent debug printout for recursion
        dbg_indent = lambda lev: 4 * lev * "-" + ">" if lev else ""

        info(2, "Integrating out species with lifetime smaller than",
             threshold)
        info(3, (
            "Before optimization, the number of known primaries is {0} with " +
            "in total {1} inclusive channels").format(len(self._nonel_tab),
                                                      len(self._incl_tab)))

        if self.xbins is None:
            info(
                4,
                'Model does not provide a native xbins. Assuming JH special sophia',
                'binning.')
            from .photo_meson import SophiaSuperposition
            self.xbins = SophiaSuperposition().xbins

        bc = self.xcenters
        bw = bin_widths(self.xbins)
        # The x_mu/x_pi grid
        # dec_grid = np.fromfunction(
        #     lambda j, i: 10**(np.log10(bc[1] / bc[0]) * (j - i)), (len(bc),
        #                                                            len(bc)))

        # dec_grid = np.outer(bc, 1 / bc)

        dec_bins = np.outer(self.xbins, 1 / bc)
        dec_bins_lower = dec_bins[:-1]
        dec_bins_upper = dec_bins[1:]

        # dec_grid[dec_grid > 1.] *= 0.
        # The differential element dx_mu/x_pi
        int_scale = np.tile(bw / bc, (len(bc), 1))

        from functools import lru_cache
        @lru_cache(maxsize=512, typed=False)
        def decay_cached(mother,daughter):
            dec_dist = int_scale * decs.get_decay_matrix_bin_average(
                mother, daughter, dec_bins_lower, dec_bins_upper)
            
            return dec_dist

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
            dec_dist = decay_cached(mother,daughter)

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
                    3, dbg_indent(reclev),
                    'daughter {0} unknown, forcing beta decay. Not Implemented yet!!'
                    .format(da))
                return

            # Daughter is stable. Add it to the new dictionary and terminate
            # recursion
            if spec_data[da]["lifetime"] >= threshold:
                if self.is_differential(None, da):
                    # If the daughter is a meson or lepton, use the dictionary for
                    # differential channels
                    info(
                        20, dbg_indent(reclev),
                        'daughter {0} stable and differential. Adding to ({1}, {2})'
                        .format(da, first_mo, da))
                    new_dec_diff_tab[(first_mo, da)] = csection
                else:
                    info(
                        20, dbg_indent(reclev),
                        'daughter {0} stable. Adding to ({1}, {2})'.format(
                            da, first_mo, da))
                    new_incl_tab[(first_mo, da)] = csection
                return

            # ..otherwise follow decay products of this daughter, tracking the
            # original mother particle (first_mo). The cross section (csection) is
            # reduced by the branching ratio (br) of this particular channel
            for br, daughters in spec_data[da]["branchings"]:
                info(10, dbg_indent(reclev),
                     ("{3} -> {0:4d} -> {2:4.2f}: {1}").format(
                         da, ", ".join(map(str, daughters)), br, first_mo))

                for chained_daughter in daughters:
                    # Follow each secondary and increment the recursion level by one
                    if self.is_differential(None, chained_daughter):
                        info(10, 'daughter', chained_daughter, 'of', da,
                             'is differential')
                        follow_chain(
                            first_mo, chained_daughter,
                            convolve_with_decay_distribution(
                                self._arange_on_xgrid(csection), da,
                                chained_daughter, br), reclev + 1)
                    else:
                        follow_chain(first_mo, chained_daughter, br * csection,
                                     reclev + 1)

        # Remove all unstable particles from the dictionaries
        for mother in sorted(self._nonel_tab.keys()):
            if mother not in spec_data or spec_data[mother][
                    "lifetime"] < threshold:
                info(
                    20,
                    "Primary species {0} does not fulfill stability criteria.".
                    format(mother))
                _ = self._nonel_tab.pop(mother)
        # Only stable (interacting) mother particles are left
        self._update_indices()

        for (mother, daughter) in self.incl_idcs:

            if mother not in self.nonel_idcs:
                info(
                    30, "Removing {0}/{1} from incl, since mother not stable ".
                    format(mother, daughter))
                _ = self._incl_tab.pop((mother, daughter))

            elif self.is_differential(mother, daughter):
                # Move the distributions which are expected to be differential
                # to _incl_diff_tab
                self._incl_diff_tab[(mother,
                                     daughter)] = self._arange_on_xgrid(
                                         self._incl_tab.pop(
                                             (mother, daughter)))

        self._update_indices()

        for (mother, daughter) in self.incl_diff_idcs:

            if mother not in self.nonel_idcs:
                info(
                    30,
                    "Removing {0}/{1} from diff incl, since mother not stable "
                    .format(mother, daughter))
                _ = self._incl_diff_tab.pop((mother, daughter))

        self._update_indices()

        # Launch the reduction for each inclusive channel
        for (mo, da), value in list(self._incl_tab.items()):
            #print mo, da, value
            #print '---'*30
            follow_chain(mo, da, value, 0)

        for (mo, da), value in list(self._incl_diff_tab.items()):
            #print mo, da, value
            #print '---'*30
            follow_chain(mo, da, value, 0)

        # Overwrite the old incl dictionary
        self._incl_tab = dict(new_incl_tab)
        # Overwrite the old incl_diff dictionary
        self._incl_diff_tab = dict(new_dec_diff_tab)
        # Reduce also the incl_diff_tab by removing the unknown mothers. At this stage
        # of the code, the particles with redistributions are
        info(
            3,
            ("After optimization, the number of known primaries is {0} with " +
             "in total {1} inclusive channels").format(
                 len(self._nonel_tab),
                 len(self._incl_tab) + len(self._incl_diff_tab)))
        info(2, f'Cache used for decays, {decay_cached.cache_info()}') # pylint:disable=no-value-for-parameter

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

        from scipy.integrate import trapz

        if (mother, daughter) in self._incl_diff_tab:
            # Return the integral of the differential for the inclusive
            egr_incl, cs_diff = self.incl_diff(mother, daughter)
            # diff_mat = diff_mat.transpose()
            cs_incl = trapz(cs_diff,
                            x=self.xcenters,
                            dx=bin_widths(self.xbins),
                            axis=0)

            if isinstance(self._incl_diff_tab[(mother, daughter)], tuple):
                return egr_incl, cs_incl

            return self.egrid, cs_incl[self._range]

        elif (mother, daughter) not in self._incl_tab:
            raise Exception(
                self.__class__.__name__ + '::'
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
                '({0},{1}) combination not in inclusive differential cross sections'
                .format(mother, daughter))

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
        # NOTE: The factor 2 in the following line is a workarround to account for the latter linear interpolation
        #       This is needed because linear spline integral will result in a trapz,
        #       which has half the area of the actual first bin
        corr_factor = 2 * self.xwidths[-1] / (self.xcenters[-1] -
                                              self.xcenters[-2])
        csec[-1, :] = cs / self.xwidths[-1] * corr_factor
        info(
            4,
            'Warning! Workaround to account for linear interpolation in x, factor 2 added!'
        )
        if isinstance(incl_cs, tuple):
            return egr, csec
        return csec

    def multiplicities(self, mother, daughter):
        '''Return the multiplicities from either the inclusive channels, or the
        differential ones integrated by x, as a function of Energy.
        '''
        egrid_incl, cs_incl = self.incl(mother, daughter)
        egrid_nonel, cs_nonel = self.nonel(mother)

        if egrid_incl.shape != egrid_nonel.shape:
            raise Exception('Problem with different grid shapes')

        multiplicities = cs_incl / np.where(cs_nonel == 0, np.inf, cs_nonel)

        return egrid_nonel, multiplicities


if __name__ == "__main__":
    pass
