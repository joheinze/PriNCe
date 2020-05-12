
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
            # cgrid = N * self.cs_neutron_grid
            if np.any(csec_diff):
                csec_diff += self.redist_neutron[daughter].T * N
            else:
                csec_diff = self.redist_neutron[daughter].T * N

        return self.egrid, csec_diff[:, self._range]
