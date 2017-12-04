"""Contains solvers, i.e. integrators, kernels, steppers, for PriNCe."""

import numpy as np
from prince.cosmology import H
from prince.util import info, PRINCE_UNITS, get_AZN, EnergyGrid
from prince_config import config

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
        self.ewi_state = np.tile(self.prince_run.cr_grid.widths,
                                 prince_run.spec_man.nspec)
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

    def get_solution(self, nco_id):
        """Returns the spectrum in energy per nucleon"""
        sp = self.prince_run.spec_man.ncoid2sref[nco_id]
        return self.egrid, self.r.y[sp.lidx():sp.uidx()]

    def get_solution_scale(self, nco_id, epow=0):
        """Returns the spectrum scaled back to total energy"""
        spec = self.prince_run.spec_man.ncoid2sref[nco_id]
        egrid = spec.A * self.egrid
        return egrid, egrid**epow * self.r.y[
            spec.lidx():spec.uidx()] / spec.A

    def get_solution_group(self, nco_ids, epow=3):
        """Return the summed spectrum (in total energy) for all elements in the range"""
        # Take egrid from first id ( doesn't cover the range for iron for example)
        max_mass = max([s.A for s in self.spec_man.species_refs])
        emin_log, emax_log, nbins = list(config["cosmic_ray_grid"])
        emax_log = np.log10(max_mass * 10**emax_log)
        nbins *= 2
        com_egrid = EnergyGrid(emin_log, emax_log, nbins).grid
        spec = np.zeros_like(com_egrid)
        for pid in nco_ids:
            curr_egrid, curr_spec = self.get_solution_scale(pid, epow=0)
            spec += np.exp(
                np.interp(
                    np.log(com_egrid),
                    np.log(curr_egrid),
                    np.log(curr_spec),
                    left=0.,
                    right=0.))

        return com_egrid, com_egrid**epow * spec

    def set_initial_condition(self, spectrum=None, nco_id=None):
        self.state *= 0.
        if spectrum is not None and nco_id is not None:
            sp = self.prince_run.spec_man.ncoid2sref[nco_id]
            self.state[sp.lidx():sp.uidx()] = spectrum
        # Initial value
        self.r.set_initial_value(self.state, self.initial_z)
        self._update_jacobian(self.initial_z)
        self.last_z = self.initial_z
        self.last_hadr_jac = None

    def injection(self, dz, z):
        """This needs to return the injection rate
        at each redshift value z"""
        f = self.dldz(z) * dz / PRINCE_UNITS.cm2sec
        return f * np.sum(
            [s.injection_rate(z) for s in self.list_of_sources], axis=0)

    def _update_jacobian(self, z):
        info(5, 'Updating jacobian matrix at redshift', z)
        self.continuous_losses = self.continuous_loss_rates.loss_vector(z)
        self.jacobian = self.int_rates.get_hadr_jacobian(z, self.dldz(z))
        self.last_hadr_jac = None

    def eqn_jac(self, z, state, *jac_args):
        self.ncallsj += 1
        if self.last_hadr_jac is None:
            self.last_hadr_jac = self.int_rates.get_dense_hadr_jacobian(
                z, self.dldz(z))

        return self.last_hadr_jac

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
        # r = self.jacobian.dot(state) + self.injection(1., z)
        r = self.jacobian.dot(state) + self.injection(1., z)
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
            'rtol': 1e-2,
            # 'atol': 1e5,
            'ndim': self.dim_states,
            'nnz': self.jacobian.nnz,
            # 'csc_jacobian': self.jacobian.tocsc(),
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

    def solve(self, dz=1e-2, verbose=True, extended_output=False):
        from time import time

        dz = -1 * dz
        now = time()
        self.r.set_initial_value(np.zeros(self.dim_states), self.r.t)

        info(2, 'Starting integration.')

        while self.r.successful() and (self.r.t + dz) > self.final_z:
            if verbose:
                info(3, "Integrating at z={0}".format(self.r.t))

            self._update_jacobian(self.r.t)
            stepin = time()
            self.r.integrate(self.r.t + dz)  #,
            # step=True if self.r.t < self.initial_z else False)
            if verbose:
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
                    print 'LAST STEP {0:4.3e}'.format(
                        self.r._integrator.rwork[10])
            self.ncallsf = 0
            self.ncallsj = 0

            if self.enable_cont_losses:
                self.r._integrator.call_args[3] = 20
                self.r._y = self.semi_lagrangian(dz, self.r.t, self.r.y)

        self.r.integrate(self.final_z)

        if not self.r.successful():
            raise Exception(
                'Integration failed. Change integrator setup and retry.')

        info(2, 'Integration completed in {0} s'.format(time() - now))
