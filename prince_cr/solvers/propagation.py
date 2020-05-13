"""Contains solvers, i.e. integrators, kernels, steppers, for PriNCe."""

import numpy as np

from prince_cr.cosmology import H
from prince_cr.data import PRINCE_UNITS, EnergyGrid
from prince_cr.util import info
import prince_cr.config as config

from .partial_diff import DifferentialOperator, SemiLagrangianSolver

class UHECRPropagationResult(object):
    """Reduced version of solver class, that only holds the result vector and defined add and multiply
    """

    def __init__(self, state, egrid, spec_man):
        self.spec_man = spec_man
        self.egrid = egrid
        self.state = state

    def to_dict(self):
        dic = {}
        dic['egrid'] = self.egrid
        dic['state'] = self.state
        dic['known_spec'] = self.known_species

        return dic

    @classmethod
    def from_dict(cls, dic):
        egrid = dic['egrid']
        edim = egrid.size
        state = dic['state']
        known_spec = dic['known_spec']

        from ..data import SpeciesManager
        spec_man = SpeciesManager(known_spec, edim)

        return cls(state, egrid, spec_man)

    @property
    def known_species(self):
        return self.spec_man.known_species

    def __add__(self, other):
        cumstate = self.state + other.state

        if not np.array_equal(self.egrid, other.egrid):
            raise Exception(
                'Cannot add Propagation Results, they are defined in different energy grids!'
            )
        if not np.array_equal(self.known_species, other.known_species):
            raise Exception(
                'Cannot add Propagation Results, have different species managers!'
            )
        else:
            return UHECRPropagationResult(cumstate, self.egrid, self.spec_man)

    def __mul__(self, number):
        if not np.isscalar(number):
            raise Exception(
                'Can only multiply result by scalar number, got type {:} instead!'.
                format(type(number)))
        else:
            newstate = self.state * number
            return UHECRPropagationResult(newstate, self.egrid, self.spec_man)

    def get_solution(self, nco_id):
        """Returns the spectrum in energy per nucleon"""
        spec = self.spec_man.ncoid2sref[nco_id]
        return self.egrid, self.state[spec.lidx():spec.uidx()]

    def get_solution_scale(self, nco_id, epow=0):
        """Returns the spectrum scaled back to total energy"""
        spec = self.spec_man.ncoid2sref[nco_id]
        egrid = spec.A * self.egrid
        return egrid, egrid**epow * self.state[spec.lidx():spec.uidx()] / spec.A

    def _check_id_grid(self, nco_ids, egrid):
        # Take egrid from first id ( doesn't cover the range for iron for example)
        # create a common egrid or used supplied one
        if egrid is None:
            max_mass = max([s.A for s in self.spec_man.species_refs])
            emin_log, emax_log, nbins = list(config.cosmic_ray_grid)
            emax_log = np.log10(max_mass * 10**emax_log)
            nbins *= 4
            com_egrid = EnergyGrid(emin_log, emax_log, nbins).grid
        else:
            com_egrid = egrid

        if isinstance(nco_ids, list):
            pass
        elif nco_ids == 'CR':
            nco_ids = [s for s in self.known_species if s >= 100]
        elif nco_ids == 'nu':
            nco_ids = [
                s for s in self.known_species if s in [11, 12, 13, 14, 15, 16]
            ]
        elif nco_ids == 'all':
            nco_ids = self.known_species
        elif isinstance(nco_ids, tuple):
            select, vmin, vmax = nco_ids
            nco_ids = [
                s for s in self.known_species if vmin <= select(s) <= vmax
            ]

        return nco_ids, com_egrid

    def _collect_interpolated_spectra(self, nco_ids, epow, egrid=None):
        """Collect interpolated spectra in a 2D array. Used by
        get_solution_group and get_lnA"""
        nco_ids, com_egrid = self._check_id_grid(nco_ids, egrid)
        
        # collect all the spectra in 2d array of dimension
        spectra = np.zeros((len(nco_ids), com_egrid.size))
        for idx, pid in enumerate(nco_ids):
            curr_egrid, curr_spec = self.get_solution_scale(pid, epow)
            mask = curr_spec > 0.
            if np.count_nonzero(mask) > 0:
                res = np.exp(
                    np.interp(
                        np.log(com_egrid),
                        np.log(curr_egrid[mask]),
                        np.log(curr_spec[mask]),
                        left=np.nan,
                        right=np.nan))
            else:
                res = np.zeros_like(com_egrid)
            spectra[idx] = np.nan_to_num(res)

        return nco_ids, com_egrid, spectra

    def get_solution_group(self, nco_ids, epow=3, egrid=None):
        """Return the summed spectrum (in total energy) for all elements in the range"""
        
        _, com_egrid, spectra = self._collect_interpolated_spectra(nco_ids, epow, egrid)
        spectrum = spectra.sum(axis=0)

        return com_egrid, spectrum

    def get_lnA(self, nco_ids, egrid=None):
        """Return the average ln(A) as a function of total energy for all elements in the range"""
        
        nco_ids, com_egrid, spectra = self._collect_interpolated_spectra(nco_ids, 0, egrid)

        # get the average and variance by using the spectra as weights
        lnA = np.array([np.log(self.spec_man.ncoid2sref[el].A) for el in nco_ids])
        average = (
            lnA[:, np.newaxis] * spectra).sum(axis=0) / spectra.sum(axis=0)
        variance = (lnA[:, np.newaxis]**2 *
                    spectra).sum(axis=0) / spectra.sum(axis=0) - average**2

        return com_egrid, average, variance

    def get_energy_density(self, nco_id):
        from scipy.integrate import trapz

        A = self.spec_man.ncoid2sref[nco_id].A
        return trapz(A * self.egrid * self.get_solution(nco_id), self.egrid)


class UHECRPropagationSolver(object):
    def __init__(
            self,
            initial_z,
            final_z,
            prince_run,
            enable_adiabatic_losses=True,
            enable_pairprod_losses=True,
            enable_photohad_losses=True,
            enable_injection_jacobian=True,
            enable_partial_diff_jacobian=True,
            z_offset=0.,
    ):
        self.initial_z = initial_z + z_offset
        self.final_z = final_z + z_offset
        self.z_offset = z_offset

        self.current_z_rates = None
        self.recomp_z_threshold = config.update_rates_z_threshold

        self.spec_man = prince_run.spec_man
        self.egrid = prince_run.cr_grid.grid
        self.ebins = prince_run.cr_grid.bins
        # Flags to enable/disable different loss types
        self.enable_adiabatic_losses = enable_adiabatic_losses
        self.enable_pairprod_losses = enable_pairprod_losses
        self.enable_photohad_losses = enable_photohad_losses
        # Flag for Jacobian injection:
        # if True injection in jacobion
        # if False injection only per dz-step
        self.enable_injection_jacobian = enable_injection_jacobian
        self.enable_partial_diff_jacobian = enable_partial_diff_jacobian

        self.had_int_rates = prince_run.int_rates
        self.adia_loss_rates_grid = prince_run.adia_loss_rates_grid
        self.pair_loss_rates_grid = prince_run.pair_loss_rates_grid
        self.adia_loss_rates_bins = prince_run.adia_loss_rates_bins
        self.pair_loss_rates_bins = prince_run.pair_loss_rates_bins
        self.intp = None

        self.state = np.zeros(prince_run.dim_states)
        self.result = None
        self.dim_states = prince_run.dim_states

        self.list_of_sources = []

        self.diff_operator = DifferentialOperator(prince_run.cr_grid,
                                                  prince_run.spec_man.nspec).operator
        self.semi_lag_solver = SemiLagrangianSolver(prince_run.cr_grid)

        # Configuration of BLAS backend
        self.using_cupy = False
        if config.has_cupy and config.linear_algebra_backend.lower() == 'cupy':
            self.eqn_derivative = self.eqn_deriv_cupy
            self.using_cupy = True
        elif config.has_mkl and config.linear_algebra_backend.lower() == 'mkl':
            self.eqn_derivative = self.eqn_deriv_mkl
        else:
            self.eqn_derivative = self.eqn_deriv_standard

        # Reset counters
        self.ncallsf = 0
        self.ncallsj = 0

    @property
    def known_species(self):
        return self.spec_man.known_species

    @property
    def res(self):
        if self.result is None:
            self.result = UHECRPropagationResult(self.state, self.egrid,
                                                 self.spec_man)
        return self.result

    def add_source_class(self, source_instance):
        self.list_of_sources.append(source_instance)

    def dldz(self, z):
        return -1. / ((1. + z) * H(z) * PRINCE_UNITS.cm2sec)

    def injection(self, dz, z):
        """This needs to return the injection rate
        at each redshift value z"""
        f = self.dldz(z) * dz * PRINCE_UNITS.cm2sec
        if len(self.list_of_sources) > 1:
            return f * np.sum(
                [s.injection_rate(z) for s in self.list_of_sources], axis=0)
        else:
            return f * self.list_of_sources[0].injection_rate(z)

    def _update_jacobian(self, z):
        info(5, 'Updating jacobian matrix at redshift', z)

        # enable photohadronic losses, or use a zero matrix
        if self.enable_photohad_losses:
            self.jacobian = self.had_int_rates.get_hadr_jacobian(
                z, self.dldz(z), force_update=True)
        else:
            self.jacobian = self.had_int_rates.get_hadr_jacobian(
                z, self.dldz(z), force_update=True)
            self.jacobian.data *= 0.

        self.last_hadr_jac = None

    def semi_lagrangian(self, delta_z, z, state):
        z = z - self.z_offset

        # if no continuous losses are enables, just return the state.
        if not self.enable_adiabatic_losses and not self.enable_pairprod_losses:
            return state

        conloss = np.zeros_like(self.adia_loss_rates_bins.energy_vector)
        if self.enable_adiabatic_losses:
            conloss += self.adia_loss_rates_bins.loss_vector(z)
        if self.enable_pairprod_losses:
            conloss += self.pair_loss_rates_bins.loss_vector(z)
        conloss *= self.dldz(z) * delta_z

        method = config.semi_lagr_method

        # -------------------------------------------------------------------
        # numpy interpolator
        # -------------------------------------------------------------------
        if method == 'intp_numpy':
            for spec in self.spec_man.species_refs:
                lbin, ubin = spec.lbin(), spec.ubin()
                lidx, uidx = spec.lidx(), spec.uidx()
                state[lidx:uidx] = self.semi_lag_solver.interpolate(
                    conloss[lbin:ubin], state[lidx:uidx])
        # -------------------------------------------------------------------
        # local gradient interpolation
        # -------------------------------------------------------------------
        elif method == 'gradient':
            for spec in self.spec_man.species_refs:
                lbin, ubin = spec.lbin(), spec.ubin()
                lidx, uidx = spec.lidx(), spec.uidx()
                state[lidx:uidx] = self.semi_lag_solver.interpolate_gradient(
                    conloss[lbin:ubin], state[lidx:uidx])

        # -------------------------------------------------------------------
        # linear lagrange weigts
        # -------------------------------------------------------------------
        elif method == 'linear':
            for spec in self.spec_man.species_refs:
                lbin, ubin = spec.lbin(), spec.ubin()
                lidx, uidx = spec.lidx(), spec.uidx()
                state[lidx:
                      uidx] = self.semi_lag_solver.interpolate_linear_weights(
                          conloss[lbin:ubin], state[lidx:uidx])

        # -------------------------------------------------------------------
        # quadratic lagrange weigts
        # -------------------------------------------------------------------
        elif method == 'quadratic':
            for spec in self.spec_man.species_refs:
                lbin, ubin = spec.lbin(), spec.ubin()
                lidx, uidx = spec.lidx(), spec.uidx()
                state[
                    lidx:
                    uidx] = self.semi_lag_solver.interpolate_quadratic_weights(
                        conloss[lbin:ubin], state[lidx:uidx])

        # -------------------------------------------------------------------
        # cubic lagrange weigts
        # -------------------------------------------------------------------
        elif method == 'cubic':
            for spec in self.spec_man.species_refs:
                lbin, ubin = spec.lbin(), spec.ubin()
                lidx, uidx = spec.lidx(), spec.uidx()
                state[lidx:
                      uidx] = self.semi_lag_solver.interpolate_cubic_weights(
                          conloss[lbin:ubin], state[lidx:uidx])

        # -------------------------------------------------------------------
        # 4th order lagrange weigts
        # -------------------------------------------------------------------
        elif method == '4th_order':
            for spec in self.spec_man.species_refs:
                lbin, ubin = spec.lbin(), spec.ubin()
                lidx, uidx = spec.lidx(), spec.uidx()
                state[
                    lidx:
                    uidx] = self.semi_lag_solver.interpolate_4thorder_weights(
                        conloss[lbin:ubin], state[lidx:uidx])

        # -------------------------------------------------------------------
        # 5th order lagrange weigts
        # -------------------------------------------------------------------
        elif method == '5th_order':
            for spec in self.spec_man.species_refs:
                lbin, ubin = spec.lbin(), spec.ubin()
                lidx, uidx = spec.lidx(), spec.uidx()
                state[
                    lidx:
                    uidx] = self.semi_lag_solver.interpolate_5thorder_weights(
                        conloss[lbin:ubin], state[lidx:uidx])

        # -------------------------------------------------------------------
        # finite diff euler steps
        # -------------------------------------------------------------------
        elif method == 'finite_diff':
            conloss = np.zeros_like(self.adia_loss_rates_grid.energy_vector)
            if self.enable_adiabatic_losses:
                conloss += self.adia_loss_rates_grid.loss_vector(z)
            if self.enable_pairprod_losses:
                conloss += self.pair_loss_rates_grid.loss_vector(z)

            state[:] = state + self.dldz(
                z) * delta_z * self.diff_operator.dot(conloss * state)

        # -------------------------------------------------------------------
        # if method was not in list before, raise an Expection
        # -------------------------------------------------------------------
        else:
            raise Exception(
                'Unknown semi-lagrangian method ({:})'.format(method))

        return state

    def eqn_deriv_standard(self, z, state, *args):
        z = z - self.z_offset

        self.ncallsf += 1

        # Update rates/cross sections only if solver requests to do so
        if abs(z - self.current_z_rates) > self.recomp_z_threshold:
            self._update_jacobian(z)
            self.current_z_rates = z

        r = self.jacobian.dot(state)
        
        if self.enable_injection_jacobian:
            r += self.injection(1., z)[:, np.newaxis]

        if self.enable_partial_diff_jacobian:
            conloss = np.zeros_like(self.adia_loss_rates_grid.energy_vector)
            if self.enable_adiabatic_losses:
                conloss += self.adia_loss_rates_grid.loss_vector(z)
            if self.enable_pairprod_losses:
                conloss += self.pair_loss_rates_grid.loss_vector(z)
            r += self.dldz(z) * self.diff_operator.dot(
                conloss[:, np.newaxis] * state)
        return r

    def eqn_deriv_cupy(self, z, state, *args):
        import cupy
        
        z = z - self.z_offset

        self.ncallsf += 1

        # Update rates/cross sections only if solver requests to do so
        if abs(z - self.current_z_rates) > self.recomp_z_threshold:
            self._update_jacobian(z)
            self.current_z_rates = z
        
        state = cupy.array(state)

        r = self.jacobian.dot(state)
        if self.enable_injection_jacobian:
            r += cupy.array(self.injection(1., z))[:, np.newaxis]
        if self.enable_partial_diff_jacobian:
            conloss = np.zeros_like(self.adia_loss_rates_grid.energy_vector)
            if self.enable_adiabatic_losses:
                conloss += self.adia_loss_rates_grid.loss_vector(z)
            if self.enable_pairprod_losses:
                conloss += self.pair_loss_rates_grid.loss_vector(z)
            r += self.dldz(z) * self.diff_operator.dot(
                cupy.array(conloss)[:, np.newaxis] * state)
        
        return cupy.asnumpy(r)

    def eqn_deriv_mkl(self, z, state, *args):
        from prince_cr.solvers.mkl_interface import csrmm, csrmv
        if state.shape[1] > 1:
            matrix_input = True
        else:
            matrix_input = False

        # Here starts the eqn_deriv content
        z = z - self.z_offset
        self.ncallsf += 1

        # Update rates/cross sections only if solver requests to do so
        if abs(z - self.current_z_rates) > self.recomp_z_threshold:
            self._update_jacobian(z)
            self.current_z_rates = z

        res = np.zeros(state.shape)
        
        if matrix_input:
            state = np.ascontiguousarray(state)
            res = csrmm(1., self.jacobian, state, 0., res)
        else:
            res = csrmv(1., self.jacobian, state, 0., res)
            

        if self.enable_injection_jacobian:
            res += self.injection(1., z)[:, np.newaxis]
        
        if self.enable_partial_diff_jacobian:
            conloss = np.zeros_like(self.adia_loss_rates_grid.energy_vector)
            if self.enable_adiabatic_losses:
                conloss += self.adia_loss_rates_grid.loss_vector(z)
            if self.enable_pairprod_losses:
                conloss += self.pair_loss_rates_grid.loss_vector(z)
            if matrix_input:
                res = csrmm(self.dldz(z), self.diff_operator,
                    conloss[:, np.newaxis] * state, 1., res)
            else:
                res = csrmv(self.dldz(z), self.diff_operator,
                    conloss[:, np.newaxis] * state, 1., res)

        return res

class UHECRPropagationSolverBDF(UHECRPropagationSolver):

    def __init__(self, *args, **kwargs):
        self.atol = kwargs.pop('atol', 1e40)
        self.rtol = kwargs.pop('rtol', 1e-10)
        super(UHECRPropagationSolverBDF, self).__init__(*args, **kwargs)

    def _init_solver(self, dz):
        initial_state = np.zeros(self.dim_states)

        self._update_jacobian(self.initial_z)
        self.current_z_rates = self.initial_z

        # find the maximum injection and reduce the system by this
        self.red_idx = np.nonzero(self.injection(1., 0.))[0].max()
        
        # Convert csr_matrix from GPU to scipy
        try:
            sparsity = self.had_int_rates.get_hadr_jacobian(
                self.initial_z, 1.).get()
        except AttributeError:
            sparsity = self.had_int_rates.get_hadr_jacobian(
                self.initial_z, 1.)

        from prince_cr.util import PrinceBDF
        self.r = PrinceBDF(
            self.eqn_derivative,
            self.initial_z,
            initial_state,
            self.final_z,
            max_step=np.abs(dz),
            atol=self.atol,
            rtol=self.rtol,
            #  jac = self.eqn_jac,
            jac_sparsity=sparsity,
            vectorized=True)

    def solve(self,
              dz=1e-3,
              verbose=True,
              summary=False,
              extended_output=False,
              full_reset=False,
              progressbar=False):
        from time import time
        from prince_cr.util import PrinceProgressBar

        reset_counter = 0
        stepcount = 0
        dz = -1 * dz
        start_time = time()

        info(2, 'Setting up Solver')
        self._init_solver(dz)
        info(2, 'Solver initialized in {0} s'.format(time() - start_time))

        info(2, 'Starting integration.')
        with PrinceProgressBar(
            bar_type=progressbar,
            nsteps=-(self.initial_z - self.final_z) / dz) as pbar:
            while self.r.status == 'running':
                # print '------ at', self.r.t, '------'
                if verbose:
                    info(3, "Integrating at z = {0}".format(self.r.t))
                # --------------------------------------------
                # Evaluate a step
                # --------------------------------------------
                self.r.step()
                # --------------------------------------------
                # Some last checks and resets
                # --------------------------------------------
                if verbose:
                    print('last step:', self.r.step_size)
                    print('h_abs:', self.r.h_abs)
                    print('LU decomp:', self.r.nlu)
                    print('current order:', self.r.dense_output().order)
                    print('---' * 20)

                stepcount += 1
                reset_counter += 1
                pbar.update()

        if self.r.status == 'failed':
            raise Exception(
                'Integrator failed at t = {:}, try adjusting the tolerances'.
                format(self.r.t))
        if verbose:
            print('Integrator finished with t = {:}, last step was dt = {:}'.format(
                self.r.t, self.r.step_size))

        # after each run we delete the solver to save memory
        self.state = self.r.y.copy()

        if summary or verbose:
            print('Summary:')
            print('---------------------')
            print('status:   ', self.r.status)
            print('time:     ', self.r.t)
            print('last step:', self.r.step_size)
            print('RHS eval: ', self.r.nfev)
            print('Jac eval: ', self.r.njev)
            print('LU decomp:', self.r.nlu)
            print('---------------------')

        del self.r

        end_time = time()
        info(2, 'Integration completed in {0} s'.format(end_time - start_time))

class UHECRPropagationSolverEULER(UHECRPropagationSolver):
    def __init__(self, *args, **kwargs):
        super(UHECRPropagationSolverEULER, self).__init__(*args, **kwargs)

    def solve(self, dz=1e-3, verbose=True, extended_output=False,
              initial_inj=False, disable_inj=False, progressbar=False):
        from prince_cr.util import PrinceProgressBar
        from time import time
        self._update_jacobian(self.initial_z)
        self.current_z_rates = self.initial_z

        dz = -1 * dz
        start_time = time()
        curr_z = self.initial_z
        if initial_inj:
            initial_state = np.atleast_2d(self.injection(dz, self.initial_z))
            print(initial_state)
        else:
            initial_state = np.zeros((self.dim_states,1))
        state = initial_state
        info(2, 'Starting integration.')
        with PrinceProgressBar(
            bar_type=progressbar,
            nsteps=-(self.initial_z - self.final_z) / dz) as pbar:
            while curr_z + dz >= self.final_z:
                if verbose:
                    info(3, "Integrating at z={0}".format(curr_z))

                # --------------------------------------------
                # Solve for hadronic interactions
                # --------------------------------------------
                if verbose:
                    print('Solving hadr losses at t =', curr_z)
                self._update_jacobian(curr_z)

                state += self.eqn_derivative(curr_z, state) * dz

                # --------------------------------------------
                # Apply the injection
                # --------------------------------------------
                if not disable_inj:
                    if not self.enable_injection_jacobian and not self.enable_partial_diff_jacobian:
                        if verbose:
                            print('applying injection at t =', curr_z)
                        state += self.injection(dz, curr_z)

                # --------------------------------------------
                # Apply the semi lagrangian
                # --------------------------------------------
                if not self.enable_partial_diff_jacobian:
                    if self.enable_adiabatic_losses or self.enable_pairprod_losses:
                        if verbose:
                            print('applying semi lagrangian at t =', curr_z)
                        state = self.semi_lagrangian(dz, curr_z, state)

                # --------------------------------------------
                # Some last checks and resets
                # --------------------------------------------
                if curr_z < -1 * dz:
                    print('break at z =', curr_z)
                    break
                curr_z += dz
                pbar.update()

        end_time = time()
        info(2, 'Integration completed in {0} s'.format(end_time - start_time))
        self.state = state