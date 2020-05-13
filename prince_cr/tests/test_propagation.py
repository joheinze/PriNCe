# Test whether cross section are correctly created.

import numpy as np

import prince_cr.config as config
from prince_cr import cross_sections, photonfields, core
from prince_cr.solvers import UHECRPropagationSolverBDF
from prince_cr.cr_sources import AugerFitSource

config.x_cut = 1e-4
config.x_cut_proton = 1e-2
config.tau_dec_threshold = np.inf
config.max_mass = 14
config.debug_level = 0  # suppress info statements

pf = photonfields.CombinedPhotonField(
    [photonfields.CMBPhotonSpectrum, photonfields.CIBGilmore2D])
cs = cross_sections.CompositeCrossSection([
    (0., cross_sections.TabulatedCrossSection, ('CRP2_TALYS', )),
    (0.14, cross_sections.SophiaSuperposition, ())
])

prince_run_talys = core.PriNCeRun(max_mass=14,
                                  photon_field=pf,
                                  cross_sections=cs)

import unittest


class TestCsec(unittest.TestCase):
    def test_propagation(self):
        solver = UHECRPropagationSolverBDF(initial_z=1.,
                                           final_z=0.,
                                           prince_run=prince_run_talys,
                                           enable_pairprod_losses=True,
                                           enable_adiabatic_losses=True,
                                           enable_injection_jacobian=True,
                                           enable_partial_diff_jacobian=True)

        solver.add_source_class(
            AugerFitSource(prince_run_talys,
                           norm=1e-50,
                           params={
                               101: (0.96, 10**9.68, 20.),
                               402: (0.96, 10**9.68, 50.),
                               1407: (0.96, 10**9.68, 30.)
                           }))
        solver.solve(dz=1e-3, verbose=False, progressbar=False)

        self.assertEqual(solver.known_species, prince_run_talys.spec_man.known_species)

if __name__ == '__main__':
    unittest.main()