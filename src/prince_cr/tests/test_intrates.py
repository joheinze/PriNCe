# Test whether cross section are correctly created.

import numpy as np

import prince_cr.config as config
from prince_cr import cross_sections, photonfields, core

config.x_cut = 1e-4
config.x_cut_proton = 1e-2
config.tau_dec_threshold = np.inf
config.max_mass = 14
config.debug_level = 0 # suppress info statements

pf = photonfields.CombinedPhotonField(
                [photonfields.CMBPhotonSpectrum, 
                photonfields.CIBGilmore2D])
cs = cross_sections.CompositeCrossSection([(0., cross_sections.TabulatedCrossSection, ('CRP2_TALYS',)),
                                        (0.14, cross_sections.SophiaSuperposition, ())])

import unittest
class TestCsec(unittest.TestCase):

    def test_kernel_1(self):
        prince_run_talys = core.PriNCeRun(max_mass = 4, photon_field=pf, cross_sections=cs)
        self.assertEqual(prince_run_talys.int_rates._batch_matrix.shape, (88344, 72))
        self.assertEqual(prince_run_talys.int_rates._batch_rows.shape, (88344,))
        self.assertEqual(prince_run_talys.int_rates._batch_cols.shape, (88344,))
        self.assertEqual(prince_run_talys.int_rates._batch_vec.shape, (88344,))

    def test_kernel_4(self):
        prince_run_talys = core.PriNCeRun(max_mass = 1, photon_field=pf, cross_sections=cs)
        self.assertEqual(prince_run_talys.int_rates._batch_matrix.shape, (17528, 72))
        self.assertEqual(prince_run_talys.int_rates._batch_rows.shape, (17528,))
        self.assertEqual(prince_run_talys.int_rates._batch_cols.shape, (17528,))
        self.assertEqual(prince_run_talys.int_rates._batch_vec.shape, (17528,))

    def test_kernel_14(self):
        prince_run_talys = core.PriNCeRun(max_mass = 14, photon_field=pf, cross_sections=cs)
        self.assertEqual(prince_run_talys.int_rates._batch_matrix.shape, (287664, 72))
        self.assertEqual(prince_run_talys.int_rates._batch_rows.shape, (287664,))
        self.assertEqual(prince_run_talys.int_rates._batch_cols.shape, (287664,))
        self.assertEqual(prince_run_talys.int_rates._batch_vec.shape, (287664,))

if __name__ == '__main__':
    unittest.main()