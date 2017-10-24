"""Contains solvers, i.e. integrators, kernels, steppers, for PriNCe."""

import numpy as np
from prince.cosmology import H
from prince.util import info, PRINCE_UNITS, get_AZN


class UHECRPropagationSolver(object):
    def __init__(self, initial_z, final_z, prince_run,
                 enable_cont_losses=True):
        self.initial_z = initial_z
        self.final_z = final_z

        self.prince_run = prince_run
        self.spec_man = prince_run.spec_man

        self.enable_cont_losses = enable_cont_losses

        self.egrid = prince_run.egrid
        self.egr_state = np.tile(self.egrid, prince_run.spec_man.nspec)
        self.dim_cr = prince_run.int_rates.dim_cr
        self.dim_ph = prince_run.int_rates.dim_ph
        self.int_rates = self.prince_run.int_rates
        self.continuous_loss_rates = self.prince_run.continuous_losses

        self.targ_vec = prince_run.int_rates.photon_vector

        self.state = np.zeros(prince_run.dim_states)
        self.dim_states = prince_run.dim_states
        self.last_z = self.initial_z

        self.solution_vector = []
        self.list_of_sources = []
        self._init_vode()

    def dldz(self, z):
        return -1. / ((1. + z) * H(z) * PRINCE_UNITS.cm2sec)

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

    def set_initial_condition(self, spectrum=None, nco_id=None):
        self.state *= 0.
        if spectrum is not None and nco_id is not None:
            sp = self.prince_run.spec_man.ncoid2sref[nco_id]
            self.state[sp.lidx():sp.uidx()] = spectrum
        # Initial value
        self.r.set_initial_value(self.state, self.initial_z)
        self._update_jacobian(self.initial_z)
        self.last_z = self.initial_z

    def injection(self, dz, z):
        """This needs to return the injection rate
        at each redshift value z"""
        f = self.dldz(z) * dz / PRINCE_UNITS.cm2sec
        return f * np.sum(
            [s.injection_rate(z) for s in self.list_of_sources], axis=0)

    def _update_jacobian(self, z):
        info(5, 'Updating jacobian matrix at redshift', z)
        self.continuous_losses = self.continuous_loss_rates.loss_vector(z)
        self.jacobian = self.int_rates.get_hadr_jacobian(z)
        self.djacobian = None

    def eqn_jac(self, z, state, *jac_args):
        self.ncallsj += 1
        if self.djacobian is None:
            self.djacobian = self.jacobian.todense()
        return self.dldz(z) * self.djacobian

    def semi_lagrangian(self, delta_z, z, state):
        conloss = self.continuous_losses * delta_z * self.dldz(z)
        for spec in self.spec_man.species_refs:
            lidx, uidx = spec.lidx(), spec.uidx()
            state[lidx:uidx] = np.interp(
                self.egrid, self.egrid - conloss[lidx:uidx], state[lidx:uidx])
        return state

    def conloss_deriv(self, z, state, delta_z=1e-2):
        conloss = self.continuous_losses * delta_z * self.dldz(z)
        conloss_deriv = np.zeros_like(state)
        for spec in self.spec_man.species_refs:
            lidx, uidx = spec.lidx(), spec.uidx()
            sup = np.interp(self.egrid, self.egrid + conloss[lidx:uidx],
                            state[lidx:uidx])
            sd = np.interp(self.egrid, self.egrid - conloss[lidx:uidx],
                           state[lidx:uidx])
            conloss_deriv[lidx:uidx] = (sup - sd) / (2. * delta_z)
        return -conloss_deriv

    def eqn_deriv(self, z, state, *args):
        self.ncallsf += 1
        # self._update_jacobian(self.r.t)
        r = self.jacobian.dot(self.dldz(z) * state) + self.injection(1., z)
        # r = self.injection(1., z)
        return r

    def _init_vode(self):
        from scipy.integrate import ode
        from scipy.sparse import csc_matrix
        self._update_jacobian(self.initial_z)
        self.ncallsf = 0
        self.ncallsj = 0

        ode_params_lsodes = {
            'name': 'lsodes',
            'method': 'bdf',
            # 'nsteps': 10000,
            'rtol': 1e-4,
            'atol': 1e5,
            'ndim': self.dim_states,
            'nnz': self.jacobian.nnz,
            'csc_jacobian': csc_matrix(self.jacobian.todense()),
            # 'max_order_s': 5,
            # 'with_jacobian': True
        }

        ode_params_vode = {
            'name': 'vode',
            'method': 'bdf',
            'nsteps': 10000,
            'rtol': 1e-1,
            'atol': 1e10,
            # 'order': 5,
            'max_step': 0.2,
            'with_jacobian': False
        }

        ode_params_lsoda = {
            'name': 'lsoda',
            'nsteps': 10000,
            'rtol': 0.2,
            'max_order_ns': 5,
            'max_order_s': 2,
            'max_step': 0.2,
            # 'first_step': 1e-4,
        }

        ode_params = ode_params_lsodes

        # Setup solver

        self.r = ode(self.eqn_deriv, self.eqn_jac).set_integrator(**ode_params)
        # self.r = ode(self.eqn_deriv).set_integrator(**ode_params)

    def solve(self):
        from time import time
        extended_output = False
        dz = -1e-2
        now = time()
        self.r.set_initial_value(np.zeros(self.dim_states), self.r.t)

        info(2, 'Starting integration.')

        while self.r.successful() and (self.r.t + dz) > self.final_z:
            info(3, "Integrating at z={0}".format(self.r.t))
            self._update_jacobian(self.r.t)
            stepin = time()

            self.r.integrate(self.r.t + dz)

            print 'step took', time() - stepin
            print 'At t =', self.r.t
            print 'jacobian calls', self.ncallsj
            print 'function calls', self.ncallsf
            if extended_output:
                NST = self.r._integrator.iwork[10]
                NFE = self.r._integrator.iwork[11]
                NJE = self.r._integrator.iwork[13]
                NLU = self.r._integrator.iwork[20]
                NNZ = self.r._integrator.iwork[19]
                NNZLU = self.r._integrator.iwork[24] + self.r._integrator.iwork[26] + 12
                print 'NST', NST
                print 'NFE', NFE
                print 'NJE', NJE
                print 'NLU', NLU
                print 'NNZLU', NNZLU
                print 'LAST STEP {0:4.3e}'.format(self.r._integrator.rwork[10])
            self.ncallsf = 0
            self.ncallsj = 0

            if self.enable_cont_losses:
                self.r._integrator.call_args[3] = 20
                self.r._y = self.semi_lagrangian(dz, self.r.t, self.r.y)
                # self.r.set_initial_value(
                #     self.semi_lagrangian(dz, self.r.t, self.r.y), self.r.t)

        self.r.integrate(self.final_z)

        if not self.r.successful():
            raise Exception(
                'Integration failed. Change integrator setup and retry.')

        info(2, 'Integration completed in {0} s'.format(time() - now))
