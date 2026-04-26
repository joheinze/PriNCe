
from os.path import join

import numpy as np

from prince_cr.util import info, get_AZN
import prince_cr.config as config

from .base import CrossSectionBase


class TabulatedCrossSection(CrossSectionBase):
    """Interface class to read tabulated disintegration cross sections
        Data is expected to be files with names:
        (model_prefix)_egrid.data, (model_prefix)_nonel.data, (model_prefix)_incl_i_j.data 
    
        Data available for Peanut and TALYS with :
        model_prefix = 'PEANUT_IAS' and model_prefix = 'CRP2_TALYS'
        Data available from 1 MeV to 1 GeV
    """

    def __init__(self,
                 model_prefix='PEANUT_IAS',
                 *args,
                 **kwargs):
        self.supports_redistributions = False
        config.max_mass = kwargs.pop('max_mass', config.max_mass)
        CrossSectionBase.__init__(self)
        self._load(model_prefix)
        self._optimize_and_generate_index()

    def _load(self, model_prefix):
        from prince_cr.data import db_handler
        info(2, "Load tabulated cross sections")
        # The energy grid is given in MeV, so we convert to GeV
        photo_nuclear_tables = db_handler.photo_nuclear_db(model_prefix)

        egrid = photo_nuclear_tables["energy_grid"]
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

        # Now write the raw data into a dict structure
        _nonel_tab = {}
        for pid, csgrid in zip(pid_nonel, nonel_raw):
            if get_AZN(pid)[0] > config.max_mass:
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
            if get_AZN(mo)[0] > config.max_mass:
                continue
            _incl_tab[mo, da] = csgrid

        self._egrid_tab = egrid
        self._nonel_tab = _nonel_tab
        self._incl_tab = _incl_tab
        # Set initial range to whole egrid
        self.set_range()
        info(2, "Finished initialization")


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
        #print self.model_refs[1].incl_diff_idcs

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

        # Add dynamically generated indices to each model
        newincl = sorted(
            list(
                set(
                    sum([
                        m.generate_incl_channels(self.nonel_idcs)
                        for m in self.model_refs
                    ], []))))
        self.incl_idcs_all = sorted(
            list(set(sum([newincl, self.incl_idcs_all], []))))

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

        # Join the Egrid
        self._egrid_tab = np.concatenate([m.egrid for m in self.model_refs])
        self.set_range()

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

        #return np.concatenate(nonel)
        return np.concatenate(egrid), np.concatenate(nonel)

    def _join_incl(self, mother, daughter):
        """Returns joined incl cross sections."""

        info(5, 'Joining inclusive cross sections for channel',
             (mother, daughter))
        egrid = []
        incl = []

        for model in self.model_refs:
            e, csec = model.incl(mother, daughter)
            egrid.append(e)
            incl.append(csec)
        #print np.concatenate(egrid), np.concatenate(incl)
        #print '---'*30
        #return np.concatenate(incl)
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
            if config.debug_level > 1:
                if not np.allclose(self.xbins, model.xbins):
                    raise Exception('Unequal x bins. Aborting...',
                                    self.xbins.shape, model.xbins)
            if (mother, daughter) in model.incl_diff_idcs:
                egr, csec = model.incl_diff(mother, daughter)
                info(10, model.mname, mother, daughter, 'is differential.')

            elif (mother, daughter) in model.incl_idcs:
                # try to use incl and extend by zeros
                egr, csec_1d = model.incl(mother, daughter)
                print(mother, daughter, csec_1d.shape)
                # no x-distribution given, so x = 1
                csec = self._arange_on_xgrid(csec_1d)
                info(1, model.mname, mother, daughter,
                     'not differential, x=1.')
            else:
                info(
                    5, 'Model', model.mname, 'does not provide cross',
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

        #return np.concatenate(incl_diff, axis=1)
        return np.concatenate(egrid), np.concatenate(incl_diff, axis=1)
