"""Contains solvers, i.e. integrators, kernels, steppers, for PriNCe."""

import numpy as np
from prince.cosmology import H
from prince.util import info, PRINCE_UNITS, get_AZN, EnergyGrid
from prince_config import config
from prince.solvers_energy import DifferentialOperator, ChangCooperSolver, SemiLagrangianSolver

class UHECRPropagationResult(object):
    """Reduced version of solver class, that only holds the result vector and defined add and multiply
    """
    def __init__(self, state, egrid, spec_man):
        self.spec_man = spec_man
        self.egrid = egrid

        self.state = state

    @property
    def known_species(self):
        return self.spec_man.known_species

    def __add__(self, other):
        cumstate = self.state + other.state

        if self.egrid is not other.egrid:
            raise Exception('Cannot add Propagation Results, they are defined in different energy grids!')
        if self.spec_man is not other.spec_man:
            raise Exception('Cannot add Propagation Results, have different species managers!')
        else:
            return UHECRPropagationResult(cumstate, self.egrid, self.spec_man)

    def __mul__(self, number):
        if not np.isscalar(number):
            raise Exception('Can only multiply result by scalar number, got type {:} instead!'.format(type(number)))
        else:
            newstate = self.state * number
            return UHECRPropagationResult(newstate, self.egrid, self.spec_man)

    def get_solution(self, nco_id):
        """Returns the spectrum in energy per nucleon"""
        sp = self.spec_man.ncoid2sref[nco_id]
        return self.egrid, self.state[sp.lidx():sp.uidx()]

    def get_solution_scale(self, nco_id, epow=0):
        """Returns the spectrum scaled back to total energy"""
        spec = self.spec_man.ncoid2sref[nco_id]
        egrid = spec.A * self.egrid
        return egrid, egrid**epow * self.state[
            spec.lidx():spec.uidx()] / spec.A

    def get_solution_group(self, nco_ids, epow=3, egrid=None):
        """Return the summed spectrum (in total energy) for all elements in the range"""
        # Take egrid from first id ( doesn't cover the range for iron for example)
        # create a common egrid or used supplied one
        if egrid is None:
            max_mass = max([s.A for s in self.spec_man.species_refs])
            emin_log, emax_log, nbins = list(config["cosmic_ray_grid"])
            emax_log = np.log10(max_mass * 10**emax_log)
            nbins *= 4
            com_egrid = EnergyGrid(emin_log, emax_log, nbins).grid
        else:
            com_egrid = egrid

        # print com_egrid
        spectrum = np.zeros_like(com_egrid)
        for pid in nco_ids:
            curr_egrid, curr_spec = self.get_solution_scale(pid, epow=0)
            res = np.exp(np.interp(
                np.log(com_egrid),
                np.log(curr_egrid),
                np.log(curr_spec),
                left=np.nan,right=np.nan))
            spectrum += np.nan_to_num(res)

        return com_egrid, com_egrid**epow * spectrum

    def get_lnA(self, nco_ids, egrid=None):
        """Return the average ln(A) as a function of total energy for all elements in the range"""
        # create a common egrid or used supplied one
        if egrid is None:
            max_mass = max([s.A for s in self.spec_man.species_refs])
            emin_log, emax_log, nbins = list(config["cosmic_ray_grid"])
            emax_log = np.log10(max_mass * 10**emax_log)
            nbins *= 4
            com_egrid = EnergyGrid(emin_log, emax_log, nbins).grid
        else:
            com_egrid = egrid
            
        # collect all the spectra in 2d array of dimension     
        spectra = np.zeros((len(nco_ids), com_egrid.size))
        for idx, pid in enumerate(nco_ids):
            curr_egrid, curr_spec = self.get_solution_scale(pid, epow=0)
            res = np.exp(np.interp(
                np.log(com_egrid),
                np.log(curr_egrid),
                np.log(curr_spec),
                left=np.nan,right=np.nan))
            spectra[idx] = np.nan_to_num(res)
        lnA = np.array([np.log(get_AZN(el)[0]) for el in nco_ids])

        # build the average lnA and variance for each energy by weigting with the spectrum
        avg = np.zeros_like(com_egrid)
        var = np.zeros_like(com_egrid)
        for idx, line in enumerate(spectra.T):
            if np.sum(line) == 0.:
                avg[idx] = 0.
                var[idx] = 0.
            else:
                avg[idx] = np.average(lnA, weights=line)
                var[idx] = np.average((lnA - avg[idx])**2, weights=line)

        return com_egrid, avg, var

    def get_energy_density(self, nco_id):
        from scipy.integrate import trapz

        A, _, _ = get_AZN(nco_id)
        return trapz(A * self.egrid * self.get_solution(nco_id), self.egrid)

class UHECRPropagationSolver(object):
    def __init__(self, initial_z, final_z, prince_run,
                 enable_adiabatic_losses=True,
                 enable_pairprod_losses=True,
                 enable_photohad_losses=True,
                 enable_injection_jacobian=True,
                 enable_partial_diff_jacobian=False,
                ):
        self.initial_z = initial_z
        self.final_z = final_z

        self.prince_run = prince_run
        self.spec_man = prince_run.spec_man
        self.egrid = prince_run.egrid
        self.ebins = prince_run.ebins
        #Flags to enable/disable different loss types
        self.enable_adiabatic_losses = enable_adiabatic_losses
        self.enable_pairprod_losses = enable_pairprod_losses
        self.enable_photohad_losses = enable_photohad_losses
        #Flag for Jacobian injection: 
        # if True injection in jacobion
        # if False injection only per dz-step
        self.enable_injection_jacobian = enable_injection_jacobian
        self.enable_partial_diff_jacobian = enable_partial_diff_jacobian
        
        self.had_int_rates = self.prince_run.int_rates
        self.adia_loss_rates_grid = self.prince_run.adia_loss_rates_grid
        self.pair_loss_rates_grid = self.prince_run.pair_loss_rates_grid 
        self.adia_loss_rates_bins = self.prince_run.adia_loss_rates_bins
        self.pair_loss_rates_bins = self.prince_run.pair_loss_rates_bins
        self.intp = None

        self.state = np.zeros(prince_run.dim_states)
        self.result = None
        self.dim_states = prince_run.dim_states

        self.list_of_sources = []

        self._init_solver()
        self.diff_operator = DifferentialOperator(prince_run.cr_grid, prince_run.spec_man.nspec)
        self.semi_lag_solver = SemiLagrangianSolver(prince_run.cr_grid)

    @property
    def known_species(self):
        return self.prince_run.spec_man.known_species

    @property
    def res(self):
        if self.result is None:
            self.result = UHECRPropagationResult(self.state, self.egrid, self.spec_man)
        return self.result

    def add_source_class(self, source_instance):
        self.list_of_sources.append(source_instance)

    def set_initial_condition(self, spectrum=None, nco_id=None):
        self.state *= 0.
        self.result = None
        if spectrum is not None and nco_id is not None:
            sp = self.prince_run.spec_man.ncoid2sref[nco_id]
            self.state[sp.lidx():sp.uidx()] = spectrum
        # Initial value
        self.r.set_initial_value(self.state, self.initial_z)
        self._update_jacobian(self.initial_z)
        self.last_hadr_jac = None

    def dldz(self, z):
        return -1. / ((1. + z) * H(z) * PRINCE_UNITS.cm2sec)

    def injection(self, dz, z):
        """This needs to return the injection rate
        at each redshift value z"""
        f = self.dldz(z) * dz * PRINCE_UNITS.cm2sec
        return f * np.sum(
            [s.injection_rate(z) for s in self.list_of_sources], axis=0)

    def _update_jacobian(self, z):
        info(5, 'Updating jacobian matrix at redshift', z)

        # enable photohadronic losses, or use a zero matrix 
        if self.enable_photohad_losses:
            self.jacobian = self.had_int_rates.get_hadr_jacobian(z, self.dldz(z),force_update=True)
        else:
            self.jacobian = self.had_int_rates.get_hadr_jacobian(z, self.dldz(z),force_update=True)
            self.jacobian.data *= 0.

        self.last_hadr_jac = None

    def semi_lagrangian(self, delta_z, z, state):
        # if no continuous losses are enables, just return the state.
        if not self.enable_adiabatic_losses and not self.enable_pairprod_losses:
                return state

        conloss = np.zeros_like(self.adia_loss_rates_bins.energy_vector)
        if self.enable_adiabatic_losses:
            conloss += self.adia_loss_rates_bins.loss_vector(z)
        if self.enable_pairprod_losses:
            conloss += self.pair_loss_rates_bins.loss_vector(z)
        conloss *= self.dldz(z) * delta_z

        # -------------------------------------------------------------------
        # numpy interpolator
        # -------------------------------------------------------------------
        if config["semi_lagr_method"] == 'intp_numpy':
            for spec in self.spec_man.species_refs:
                lbin, ubin = spec.lbin(), spec.ubin()
                lidx, uidx = spec.lidx(), spec.uidx()
                state[lidx:uidx] = self.semi_lag_solver.interpolate(conloss[lbin:ubin], state[lidx:uidx])
        # -------------------------------------------------------------------
        # local gradient interpolation
        # -------------------------------------------------------------------
        elif config["semi_lagr_method"] == 'gradient':
            for spec in self.spec_man.species_refs:
                lbin, ubin = spec.lbin(), spec.ubin()
                lidx, uidx = spec.lidx(), spec.uidx()
                state[lidx:uidx] = self.semi_lag_solver.interpolate_gradient(conloss[lbin:ubin], state[lidx:uidx])

        # -------------------------------------------------------------------
        # linear lagrange weigts
        # -------------------------------------------------------------------
        elif config["semi_lagr_method"] == 'linear':
            for spec in self.spec_man.species_refs:
                lbin, ubin = spec.lbin(), spec.ubin()
                lidx, uidx = spec.lidx(), spec.uidx()
                state[lidx:uidx] = self.semi_lag_solver.interpolate_linear_weights(conloss[lbin:ubin], state[lidx:uidx])

        # -------------------------------------------------------------------
        # quadratic lagrange weigts
        # -------------------------------------------------------------------
        elif config["semi_lagr_method"] == 'quadratic':
            for spec in self.spec_man.species_refs:
                lbin, ubin = spec.lbin(), spec.ubin()
                lidx, uidx = spec.lidx(), spec.uidx()
                state[lidx:uidx] = self.semi_lag_solver.interpolate_quadratic_weights(conloss[lbin:ubin], state[lidx:uidx])
 
        # -------------------------------------------------------------------
        # cubic lagrange weigts
        # -------------------------------------------------------------------
        elif config["semi_lagr_method"] == 'cubic':
            for spec in self.spec_man.species_refs:
                lbin, ubin = spec.lbin(), spec.ubin()
                lidx, uidx = spec.lidx(), spec.uidx()
                state[lidx:uidx] = self.semi_lag_solver.interpolate_cubic_weights(conloss[lbin:ubin], state[lidx:uidx])

        # -------------------------------------------------------------------
        # 4th order lagrange weigts
        # -------------------------------------------------------------------
        elif config["semi_lagr_method"] == '4th_order':
            for spec in self.spec_man.species_refs:
                lbin, ubin = spec.lbin(), spec.ubin()
                lidx, uidx = spec.lidx(), spec.uidx()
                state[lidx:uidx] = self.semi_lag_solver.interpolate_4thorder_weights(conloss[lbin:ubin], state[lidx:uidx])
 
        # -------------------------------------------------------------------
        # 5th order lagrange weigts
        # -------------------------------------------------------------------
        elif config["semi_lagr_method"] == '5th_order':
            for spec in self.spec_man.species_refs:
                lbin, ubin = spec.lbin(), spec.ubin()
                lidx, uidx = spec.lidx(), spec.uidx()
                state[lidx:uidx] = self.semi_lag_solver.interpolate_5thorder_weights(conloss[lbin:ubin], state[lidx:uidx])
 
        # -------------------------------------------------------------------
        # finite diff euler steps
        # -------------------------------------------------------------------
        elif config["semi_lagr_method"] == 'finite_diff':
            conloss = np.zeros_like(self.adia_loss_rates_grid.energy_vector)
            if self.enable_adiabatic_losses:
                conloss += self.adia_loss_rates_grid.loss_vector(z)
            if self.enable_pairprod_losses:
                conloss += self.pair_loss_rates_grid.loss_vector(z)

            state[:] = state + self.dldz(z) * delta_z * self.diff_operator.operator.dot(conloss * state)

        # -------------------------------------------------------------------
        # if method was not in list before, raise an Expection
        # -------------------------------------------------------------------
        else:
            raise Exception('Unknown semi-lagrangian method ({:})'.format(config["semi_lagr_method"]))

        return state

    def eqn_deriv(self, z, state, *args):
        self.ncallsf += 1
        if self.enable_injection_jacobian and self.enable_partial_diff_jacobian:
            conloss = np.zeros_like(self.adia_loss_rates_grid.energy_vector)
            if self.enable_adiabatic_losses:
                conloss += self.adia_loss_rates_grid.loss_vector(z)
            if self.enable_pairprod_losses:
                conloss += self.pair_loss_rates_grid.loss_vector(z)
            partial_deriv = self.dldz(z) * self.diff_operator.operator.dot(conloss * state)
            r = self.jacobian.dot(state) + self.injection(1., z) + partial_deriv
        elif self.enable_injection_jacobian:
            r = self.jacobian.dot(state) + self.injection(1., z)
        elif self.enable_partial_diff_jacobian:
            conloss = np.zeros_like(self.adia_loss_rates_grid.energy_vector)
            if self.enable_adiabatic_losses:
                conloss += self.adia_loss_rates_grid.loss_vector(z)
            if self.enable_pairprod_losses:
                conloss += self.pair_loss_rates_grid.loss_vector(z)
            partial_deriv = self.dldz(z) * self.diff_operator.operator.dot(conloss * state)
            r = self.jacobian.dot(state) + partial_deriv
        else:
            r = self.jacobian.dot(state)
        return r

    def eqn_jac(self, z, state, *jac_args):
        self.ncallsj += 1
        if self.last_hadr_jac is None:
            if self.enable_photohad_losses:
                    self.last_hadr_jac = self.had_int_rates.get_dense_hadr_jacobian(
                        z, self.dldz(z))
            else:
                    self.last_hadr_jac = self.had_int_rates.get_dense_hadr_jacobian(
                        z, self.dldz(z))
                    self.last_hadr_jac *= 0.

        return self.last_hadr_jac

    def _init_solver(self):
        from scipy.integrate import ode
        from scipy.sparse import csc_matrix
        self._update_jacobian(self.initial_z)
        self.ncallsf = 0
        self.ncallsj = 0

        ode_params = {
            'name': 'lsodes',
            'method': 'bdf',
            # 'nsteps': 10000,
            'rtol': 1e-6,
            # 'atol': 1e5,
            'ndim': self.dim_states,
            'nnz': self.jacobian.nnz,
            # 'csc_jacobian': self.jacobian.tocsc(),
            # 'max_order_s': 5,
            # 'with_jacobian': True
        }

        ode_params = ode_params

        # info(1,'Setting solver with jacobian')
        # self.r = ode(self.eqn_deriv, self.eqn_jac).set_integrator(**ode_params)
        info(1,'Setting solver without jacobian')
        self.r = ode(self.eqn_deriv).set_integrator(**ode_params)

    def solve(self, dz=1e-2, verbose=True, extended_output=False, full_reset=False, progressbar=False):
        from time import time

        stepcount = 0
        dz = -1 * dz
        start_time = time()
        self.r.set_initial_value(np.zeros(self.dim_states), self.r.t)

        if progressbar:
            if progressbar == 'notebook':
                from tqdm import tqdm_notebook as tqdm
            else:
                from tqdm import tqdm
            pbar = tqdm(total=-1*int(self.r.t / dz))
            pbar.update()
        else:
            pbar = None
            
        info(2, 'Starting integration.')
        while self.r.successful() and (self.r.t + dz) > self.final_z:
            if verbose:
                info(3, "Integrating at z = {0}".format(self.r.t))
            step_start = time()
            # --------------------------------------------
            # Solve for hadronic interactions
            # --------------------------------------------
            if verbose:
                print 'Solving hadr losses at t =', self.r.t
            self._update_jacobian(self.r.t)
            self.r.integrate(self.r.t + dz)
            if verbose:
                self.print_step_info(step_start,extended_output=extended_output)
            self.ncallsf = 0
            self.ncallsj = 0
            # --------------------------------------------
            # Apply the injection
            # --------------------------------------------
            if not self.enable_injection_jacobian and not self.enable_partial_diff_jacobian:
                if verbose:
                    print 'applying injection at t =', self.r.t
                if full_reset:
                    self.r.set_initial_value(self.r.y + self.injection(dz, self.r.t), self.r.t)
                else:
                    self.r._integrator.call_args[3] = 20
                    self.r._y += self.injection(dz, self.r.t)

            # --------------------------------------------
            # Apply the semi lagrangian
            # --------------------------------------------
            if not self.enable_partial_diff_jacobian:
                if self.enable_adiabatic_losses or self.enable_pairprod_losses:
                    if verbose:
                        print 'applying semi lagrangian at t =', self.r.t
                    if full_reset:
                        self.r.set_initial_value(self.semi_lagrangian(dz, self.r.t, self.r.y), self.r.t)
                    else:
                        self.r._integrator.call_args[3] = 20
                        self.r._y = self.semi_lagrangian(dz, self.r.t, self.r.y)

            # --------------------------------------------
            # Some last checks and resets
            # --------------------------------------------
            if self.r.t < -1 * dz:
                print 'break at z =', self.r.t
                break

            stepcount += 1
            if pbar is not None:
                pbar.update()
        # self.r.integrate(self.final_z)
        if pbar is not None:
            pbar.close()
        if not self.r.successful():
            raise Exception(
                'Integration failed. Change integrator setup and retry.')

        self.state = self.r.y
        end_time = time()
        info(2, 'Integration completed in {0} s'.format(end_time - start_time))

    def print_step_info(self, step_start,extended_output=False):
        from time import time

        print 'step took', time() - step_start
        print 'At t = ', self.r.t
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


    def solve_euler(self, dz=1e-3, verbose=True, extended_output=False, full_reset=False):
        from time import time

        # stepcount = 0
        dz = -1 * dz
        start_time = time()
        curr_z = self.initial_z
        state = np.zeros(self.dim_states)

        info(2, 'Starting integration.')

        while curr_z + dz >= self.final_z:
            if verbose:
                info(3, "Integrating at z={0}".format(curr_z))
            step_start = time()
            
            # --------------------------------------------
            # Solve for hadronic interactions
            # --------------------------------------------
            if verbose:
                print 'Solving hadr losses at t =', self.r.t
            self._update_jacobian(curr_z)

            state += self.eqn_deriv(curr_z, state)*dz

            # --------------------------------------------
            # Apply the injection 
            # --------------------------------------------
            if not self.enable_injection_jacobian and not self.enable_partial_diff_jacobian:
                if verbose:
                    print 'applying injection at t =', self.r.t
                state += self.injection(dz, curr_z)
            
            # --------------------------------------------
            # Apply the semi lagrangian
            # --------------------------------------------
            if not self.enable_partial_diff_jacobian:
                if self.enable_adiabatic_losses or self.enable_pairprod_losses:
                    if verbose:
                        print 'applying semi lagrangian at t =', self.r.t
                    state= self.semi_lagrangian(dz, curr_z, state)

            # --------------------------------------------
            # Some last checks and resets
            # --------------------------------------------
            if curr_z < -1 * dz:
                print 'break at z =', self.r.t
                break

            curr_z += dz 

        # self.r.integrate(self.final_z)

        end_time = time()
        info(2, 'Integration completed in {0} s'.format(end_time - start_time))
        self.state = state