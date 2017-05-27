"""Contains solvers, i.e. integrators, kernels, steppers, for PriNCe."""

import numpy as np
from prince.cosmology import H
from prince.util import info, pru


class ExtragalacticSpace(object):
    def __init__(self, initial_z, final_z, prince_run):
        self.initial_z = initial_z
        self.final_z = final_z
        self.list_of_sources = []
        self.FGmat = prince_run.FGmat
        self.struct_mat = prince_run.struct_mat
        self.coupling_mat = np.zeros_like(self.struct_mat)

        self.prince_run = prince_run
        self.spec_man = prince_run.spec_man
        self.egrid = prince_run.egrid
        self.dim_cr = prince_run.int_rates.dim_cr
        self.dim_ph = prince_run.int_rates.dim_ph
        self.int_rates = self.prince_run.int_rates

        self.targ_vec = prince_run.int_rates.photon_vector

        self.state = np.zeros(prince_run.dim_states)

        self.solution_vector = []
        self._init_vode()

    def add_source_class(self, source_instance):
        self.list_of_sources.append()

    def get_solution(self, nco_id, redshift=0.):
        sp = self.spec_man.ncoid2sref[nco_id]
        return self.r.y[sp.lidx():sp.uidx()]
        # return self.solution_vector[-1][sp.lidx():sp.uidx()]

    def set_initial_condition(self, spectrum, nco_id):
        sp = self.spec_man.ncoid2sref[nco_id]
        self.state *= 0.
        self.state[sp.lidx():sp.uidx()] = spectrum
        # Initial value
        self.r.set_initial_value(self.state, self.initial_z)

    def injection(self, z):
        """This needs to return the injection rate
        at each redshift value z"""

        return np.sum([s.injection(z) for s in self.list_of_sources])

    def energy_loss_legth_Mpc(self, nco_id, z):
        sp = self.spec_man.ncoid2sref[nco_id]
        rate = self.FGmat[sp.lidx():sp.uidx(),
                          sp.lidx("ph"):sp.uidx("ph")].dot(self.targ_vec(z))
        return (1. / rate) * pru.cm2Mpc

    def update_coupling_mat(self, z):
        # info(3, 'Updating coupling matrix at redshift', z)
        sref = self.spec_man.ncoid2sref
        for tup in self.int_rates.matrix.keys():
            if not isinstance(tup, tuple):
                continue
            mo, da = tup
            self.coupling_mat[sref[da].lidx():sref[da].uidx(), sref[mo].lidx():
                              sref[mo].uidx()] = np.diag(
                                  self.int_rates.fg_submat((mo, da), z))
            # for colsel in range(self.spec_man.nspec):

            # i0, i1 = colsel * self.dim_cr, (colsel + 1) * self.dim_cr
            # p0, p1 = colsel * self.dim_ph, (colsel + 1) * self.dim_ph

            # self.coupling_mat[:, i0:i1] = (
            #     self.struct_mat[:, i0:i1].T *
            #     self.FGmat[:, p0:p1].dot(self.targ_vec(z, 1))).T

    def eqn_deriv(self, z, state, *args):
        # print z
        dldz = -1. / ((1. + z) * H(z) * pru.cm2sec)
        # state[state < 1e-100] *= 0.
        r = dldz * self.coupling_mat.dot(state)
        # print r
        # r[r < 1e-100] = 0.
        return r

    def _init_vode(self):
        from scipy.integrate import ode

        ode_params = {
            'name': 'vode',
            'method': 'adams',
            'nsteps': 2000,
            'rtol': 0.01,
            'max_step': 0.1,
            # 'with_jacobian': False
        }

        # Setup solver
        self.r = ode(self.eqn_deriv).set_integrator(**ode_params)

    def solve(self):
        from time import time
        dz = -1e-5
        now = time()
        info(2, 'Starting integration.')
        while self.r.successful() and (self.r.t + dz) > self.final_z:
            # info(3, "Integrating at z={0}".format(self.r.t))
            self.update_coupling_mat(self.r.t)
            self.r.integrate(self.r.t + dz)
        self.r.integrate(self.final_z)
        info(2, 'Integration completed in {0} s'.format(time() - now))
