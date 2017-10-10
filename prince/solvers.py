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
        # return self.solution_vector[-1][sp.lidx():sp.uidx()]

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

    # def energy_loss_legth_Mpc(self, nco_id, z):
    #     sp = self.prince_run.spec_man.ncoid2sref[nco_id]
    #     rate = self.FGmat[sp.lidx():sp.uidx(),
    #                       sp.lidx("ph"):sp.uidx("ph")].dot(self.targ_vec(z))
    #     return (1. / rate) * PRINCE_UNITS.cm2Mpc

    def _update_jacobian(self, z):
        info(5, 'Updating jacobian matrix at redshift', z)
        self.continuous_losses = self.continuous_loss_rates.loss_vector(z)
        self.sp_jacobian = self.int_rates.get_hadr_jacobian(z)
        self.jacobian = self.sp_jacobian.todense()

    def eqn_jac(self, z, state):
        return self.dldz(z) * self.jacobian

    def semi_lagrangian(self, delta_z, z, state):

        conloss = self.continuous_losses * delta_z * self.dldz(z)
        for s in self.spec_man.species_refs:
            lidx, uidx = s.lidx(), s.uidx()
            state[lidx:uidx] = np.interp(
                self.egrid, self.egrid + conloss[lidx:uidx], state[lidx:uidx])

        return state

    def eqn_deriv(self, z, state, *args):
        state[state < 1e-50] *= 0.

        r = self.dldz(z) * self.sp_jacobian.dot(state)
        return r

    def _init_vode(self):
        from scipy.integrate import ode

        ode_params_vode = {
            'name': 'vode',
            'method': 'bdf',
            'nsteps': 10000,
            'rtol': 1e-1,
            'atol':1e10,
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

        ode_params = ode_params_vode

        # Setup solver
        self.r = ode(self.eqn_deriv).set_integrator(**ode_params)

    def solve(self):
        from time import time
        dz = -2e-2
        now = time()
        self.r.set_initial_value(
            self.injection(dz, self.r.t), self.r.t)

        info(2, 'Starting integration.')
        while self.r.successful() and (self.r.t + dz) > self.final_z:
            info(3, "Integrating at z={0}".format(self.r.t))

            #print self.r.y

            self._update_jacobian(self.r.t)
            self.r.integrate(self.r.t + dz)

            #print self.r.y

            if self.enable_cont_losses:
                self.r.set_initial_value(
                    self.semi_lagrangian(dz, self.r.t, self.r.y) +
                    self.injection(dz, self.r.t), self.r.t)

            # reduce step size at local redshift
            #if self.r.t <= 0.1:
            #    dz = -1e-2

        if not self.r.successful():
            raise Exception(
                'Integration failed. Change integrator setup and retry.')

        self.r.integrate(self.final_z)

        info(2, 'Integration completed in {0} s'.format(time() - now))


class UHECRPropagationSolverSimple(UHECRPropagationSolver):
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
        self.last_z = self.initial_z

        self.solution_vector = []
        self.list_of_sources = []

    def eqn_jac(self, state, z):
        return self.dldz(z) * self.jacobian

    def eqn_deriv(self, state, z, *args):
        #state[state < 1e-50] *= 0.

        r = self.dldz(z) * self.sp_jacobian.dot(state)
        return r

    def get_solution(self, nco_id, redshift=0.):
        sp = self.prince_run.spec_man.ncoid2sref[nco_id]
        return self.state[sp.lidx():sp.uidx()]

    def set_initial_condition(self, spectrum=None, nco_id=None):
        self.state *= 0.
        if spectrum is not None and nco_id is not None:
            sp = self.prince_run.spec_man.ncoid2sref[nco_id]
            self.state[sp.lidx():sp.uidx()] = spectrum
        # Initial value
        self._update_jacobian(self.initial_z)
        self.last_z = self.initial_z

    def solve(self):
        from scipy.integrate import odeint
        from time import time
        dz = -2e-2
        now = time()
        info(2, 'Starting integration.')

        f = self.eqn_deriv
        jac = self.eqn_jac
        ode_params = {
            #'mxstep': 10000,
            #'rtol': 0.2,
            #'hmax': 0.2,
            #'Dfun': jac,
        }

        while self.last_z + dz > self.final_z:
            info(3, "Integrating at z={0}".format(self.last_z))

            self._update_jacobian(self.last_z)
            self.jacobian = self.sp_jacobian.todense()

            print self.last_z
            print self.state

            step = np.array([self.last_z, self.last_z + dz])
            state_old, state_new = odeint(f, self.state, step, **ode_params)
            
            self.state = state_new
            self.last_z += dz

            print self.last_z
            print self.state

            if self.enable_cont_losses:
                self.state = (self.semi_lagrangian(dz, self.last_z, self.state)
                              + self.injection(dz, self.last_z))

        step = np.array([self.last_z, self.final_z])
        state_old, state_new = odeint(f, self.state, step, **ode_params)
        self.state = state_new
        self.last_z += self.last_z + dz

        info(2, 'Integration completed in {0} s'.format(time() - now))


class UHECRPropagationSolverSparse(UHECRPropagationSolverSimple):
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
        self.last_z = self.initial_z

        self.solution_vector = []
        self.list_of_sources = []

    def eqn_jac(self, state, z):
        return self.dldz(z) * self.sp_jacobian

    def eqn_deriv(self, state, z, *args):
        state[state < 1e-50] *= 0.

        r = self.dldz(z) * self.sp_jacobian.dot(state)
        return r

    def solve(self):
        from odesparse import odeints
        from time import time
        dz = -2e-2
        now = time()
        info(2, 'Starting integration.')

        f = self.eqn_deriv
        jac = self.eqn_jac
        ode_params = {
            #'mxstep': 10000,
            #'rtol': 0.2,
            #'hmax': 0.2,
            #'JPat': jac,
            'full_output': 1,
        }

        while self.last_z + dz > self.final_z:
            info(3, "Integrating at z={0}".format(self.last_z))

            self._update_jacobian(self.last_z)
            step = np.array([self.last_z, self.last_z + dz])
            print step
            (state_old, state_new), infodic = odeints(f, self.state, step, **ode_params)
            print (state_old, state_new)
            print infodic
            self.state = state_new
            self.last_z += dz

            if self.enable_cont_losses:
                self.state = (self.semi_lagrangian(dz, self.last_z, self.state)
                              + self.injection(dz, self.last_z))

        step = np.array([self.last_z, self.final_z])
        state_old, state_new = odeints(f, self.state, step, **ode_params)
        self.state = state_new
        self.last_z += self.last_z + dz

        info(2, 'Integration completed in {0} s'.format(time() - now))