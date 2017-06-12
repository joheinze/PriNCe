"""Contains solvers, i.e. integrators, kernels, steppers, for PriNCe."""

import numpy as np
from prince.cosmology import H
from prince.util import info, pru, get_AZN


class UHECRPropagationSolver(object):
    def __init__(self, initial_z, final_z, prince_run):
        self.initial_z = initial_z
        self.final_z = final_z

        self.prince_run = prince_run
        self.egrid = prince_run.egrid
        self.dim_cr = prince_run.int_rates.dim_cr
        self.dim_ph = prince_run.int_rates.dim_ph
        self.int_rates = self.prince_run.int_rates

        self.targ_vec = prince_run.int_rates.photon_vector

        self.state = np.zeros(prince_run.dim_states)

        self.solution_vector = []
        self.list_of_sources = []
        self._init_vode()

    def add_source_class(self, source_instance):
        self.list_of_sources.append(source_instance)

    def get_energy_density(self, nco_id):
        from scipy.integrate import trapz

        A, _, _ = get_AZN(nco_id)
        return trapz(A * self.egrid * self.get_solution(nco_id),
                     A * self.egrid)

    def get_solution(self, nco_id, redshift=0.):
        sp = self.prince_run.spec_man.ncoid2sref[nco_id]
        return self.r.y[sp.lidx():sp.uidx()]
        # return self.solution_vector[-1][sp.lidx():sp.uidx()]

    def set_initial_condition(self, spectrum, nco_id):
        sp = self.prince_run.spec_man.ncoid2sref[nco_id]
        self.state *= 0.
        self.state[sp.lidx():sp.uidx()] = spectrum
        # Initial value
        self.r.set_initial_value(self.state, self.initial_z)
        self.update_coupling_mat(self.initial_z)

    def injection(self, z):
        """This needs to return the injection rate
        at each redshift value z"""

        return np.sum([s.injection(z) for s in self.list_of_sources])

    # def energy_loss_legth_Mpc(self, nco_id, z):
    #     sp = self.prince_run.spec_man.ncoid2sref[nco_id]
    #     rate = self.FGmat[sp.lidx():sp.uidx(),
    #                       sp.lidx("ph"):sp.uidx("ph")].dot(self.targ_vec(z))
    #     return (1. / rate) * pru.cm2Mpc

    def update_coupling_mat(self, z):
        info(5, 'Updating coupling matrix at redshift', z)
        self.nonel_rate, self.coupling_mat = self.int_rates.get_coupling_mat(z)

    def eqn_deriv(self, z, state, *args):

        state[state < 1e-50] *= 0.
        dldz = -1. / ((1. + z) * H(z) * pru.cm2sec)
        # print self.nonel_rate.shape, self.state.shape
        r = dldz * (-self.nonel_rate * state + self.coupling_mat.dot(state) + self.injection(z))
        return r

    def _init_vode(self):
        from scipy.integrate import ode

        ode_params = {
            'name': 'vode',
            'method': 'bdf',
            'nsteps': 10000,
            'rtol': 0.1,
            'max_step': 0.01,
            # 'with_jacobian': False
        }

        # Setup solver
        self.r = ode(self.eqn_deriv).set_integrator(**ode_params)
        self.r.stiff = 1

    def solve(self):
        from time import time
        dz = -1e-2
        now = time()
        info(2, 'Starting integration.')
        while self.r.successful() and (self.r.t + dz) > self.final_z:
            info(3, "Integrating at z={0}".format(self.r.t))
            self.update_coupling_mat(self.r.t)
            self.r.integrate(self.r.t + dz)

        if not self.r.successful():
            raise Exception('Integration failed. Change integrator setup and retry.')

        self.r.integrate(self.final_z)

        info(2, 'Integration completed in {0} s'.format(time() - now))
