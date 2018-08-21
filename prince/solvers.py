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

        from data import SpeciesManager
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
            emin_log, emax_log, nbins = list(config["cosmic_ray_grid"])
            emax_log = np.log10(max_mass * 10**emax_log)
            nbins *= 4
            com_egrid = EnergyGrid(emin_log, emax_log, nbins).grid
        else:
            com_egrid = egrid

        if type(nco_ids) is list:
            pass
        elif nco_ids == 'CR':
            nco_ids = [s for s in self.known_species if s >= 100]
        elif nco_ids == 'nu':
            nco_ids = [
                s for s in self.known_species if s in [12, 13, 14, 15, 16]
            ]
        elif nco_ids == 'all':
            nco_ids = self.known_species
        elif type(nco_ids) is tuple:
            select, vmin, vmax = nco_ids
            nco_ids = [
                s for s in self.known_species if vmin <= select(s) <= vmax
            ]

        return nco_ids, com_egrid

    def get_solution_group(self, nco_ids, epow=3, egrid=None):
        """Return the summed spectrum (in total energy) for all elements in the range"""
        nco_ids, com_egrid = self._check_id_grid(nco_ids, egrid)

        spectra = np.zeros((len(nco_ids), com_egrid.size))
        for idx, pid in enumerate(nco_ids):
            curr_egrid, curr_spec = self.get_solution_scale(pid, epow=epow)
            res = np.exp(
                np.interp(
                    np.log(com_egrid),
                    np.log(curr_egrid),
                    np.log(curr_spec),
                    left=np.nan,
                    right=np.nan))
            spectra[idx] = np.nan_to_num(res)
        spectrum = spectra.sum(axis=0)

        return com_egrid, spectrum

    def get_lnA(self, nco_ids, egrid=None):
        """Return the average ln(A) as a function of total energy for all elements in the range"""
        nco_ids, com_egrid = self._check_id_grid(nco_ids, egrid)

        # collect all the spectra in 2d array of dimension
        spectra = np.zeros((len(nco_ids), com_egrid.size))
        for idx, pid in enumerate(nco_ids):
            curr_egrid, curr_spec = self.get_solution_scale(pid)
            res = np.exp(
                np.interp(
                    np.log(com_egrid),
                    np.log(curr_egrid),
                    np.log(curr_spec),
                    left=np.nan,
                    right=np.nan))
            spectra[idx] = np.nan_to_num(res)

        # get the average and variance by using the spectra as weights
        lnA = np.array([np.log(get_AZN(el)[0]) for el in nco_ids])
        average = (
            lnA[:, np.newaxis] * spectra).sum(axis=0) / spectra.sum(axis=0)
        variance = (lnA[:, np.newaxis]**2 *
                    spectra).sum(axis=0) / spectra.sum(axis=0) - average**2

        return com_egrid, average, variance

    def get_energy_density(self, nco_id):
        from scipy.integrate import trapz

        A, _, _ = get_AZN(nco_id)
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
            enable_partial_diff_jacobian=False,
            z_offset=0.,
    ):
        self.initial_z = initial_z + z_offset
        self.final_z = final_z + z_offset
        self.z_offset = z_offset

        self.current_z_rates = None
        self.recomp_z_threshold = config["update_rates_z_threshold"]

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

        self.diff_operator = DifferentialOperator(prince_run.cr_grid,
                                                  prince_run.spec_man.nspec)
        self.semi_lag_solver = SemiLagrangianSolver(prince_run.cr_grid)

        # Reset counters
        self.ncallsf = 0
        self.ncallsj = 0

    @property
    def known_species(self):
        return self.prince_run.spec_man.known_species

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

        method = config["semi_lagr_method"]

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
                z) * delta_z * self.diff_operator.operator.dot(conloss * state)

        # -------------------------------------------------------------------
        # if method was not in list before, raise an Expection
        # -------------------------------------------------------------------
        else:
            raise Exception(
                'Unknown semi-lagrangian method ({:})'.format(method))

        return state

    def eqn_deriv(self, z, state, *args):
        z = z - self.z_offset

        self.ncallsf += 1

        # print 'state', state.shape
        # Update rates/cross sections only if solver requests to do so
        if abs(z - self.current_z_rates) > self.recomp_z_threshold:
            self._update_jacobian(z)
            self.current_z_rates = z

        r = self.jacobian.dot(state)
        # print 'deriv', r.shape

        if self.enable_injection_jacobian:
            # print 'inj', self.injection(1., z).shape
            r += self.injection(1., z)[:,np.newaxis]
        if self.enable_partial_diff_jacobian:
            conloss = np.zeros_like(self.adia_loss_rates_grid.energy_vector)
            if self.enable_adiabatic_losses:
                conloss += self.adia_loss_rates_grid.loss_vector(z)
            if self.enable_pairprod_losses:
                conloss += self.pair_loss_rates_grid.loss_vector(z)
            # print 'conloss', conloss.shape
            # print 'state',state.shape
            partial_deriv = self.dldz(z) * self.diff_operator.operator.dot(
                conloss[:,np.newaxis] * state)
            # print 'pderiv', partial_deriv.shape
            r += partial_deriv

        return r

    def eqn_deriv_old(self, z, state, *args):
        z = z - self.z_offset

        self.ncallsf += 1

        # Update rates/cross sections only if solver requests to do so
        if abs(z - self.current_z_rates) > self.recomp_z_threshold:
            self._update_jacobian(z)
            self.current_z_rates = z

        if self.enable_injection_jacobian and self.enable_partial_diff_jacobian:
            conloss = np.zeros_like(self.adia_loss_rates_grid.energy_vector)
            if self.enable_adiabatic_losses:
                conloss += self.adia_loss_rates_grid.loss_vector(z)
            if self.enable_pairprod_losses:
                conloss += self.pair_loss_rates_grid.loss_vector(z)
            partial_deriv = self.dldz(z) * self.diff_operator.operator.dot(
                conloss * state)
            r = self.jacobian.dot(state) + self.injection(1.,
                                                          z) + partial_deriv
        elif self.enable_injection_jacobian:
            r = self.jacobian.dot(state) + self.injection(1., z)
        elif self.enable_partial_diff_jacobian:
            conloss = np.zeros_like(self.adia_loss_rates_grid.energy_vector)
            if self.enable_adiabatic_losses:
                conloss += self.adia_loss_rates_grid.loss_vector(z)
            if self.enable_pairprod_losses:
                conloss += self.pair_loss_rates_grid.loss_vector(z)
            partial_deriv = self.dldz(z) * self.diff_operator.operator.dot(
                conloss * state)
            r = self.jacobian.dot(state) + partial_deriv
        else:
            r = self.jacobian.dot(state)
        return r

    def eqn_jac(self, z, state, *jac_args):
        self.ncallsj += 1

        from scipy.sparse import dia_matrix
        if self.last_hadr_jac is None:
            if self.enable_photohad_losses:
                self.last_hadr_jac = self.had_int_rates.get_dense_hadr_jacobian(
                    z, self.dldz(z))
            else:
                self.last_hadr_jac = self.had_int_rates.get_dense_hadr_jacobian(
                    z, self.dldz(z))
                self.last_hadr_jac *= 0.

        return self.last_hadr_jac


class UHECRPropagationSolverLSODES(UHECRPropagationSolver):
    def __init__(*args, **kwargs):
        ode_newparams = kwargs.pop('ode_newparams')
        super(UHECRPropagationSolverLSODES, self).__init__(*args, **kwargs)
        # UHECRPropagationSolver.__init__(self,*args,**kwargs)
        self._init_solver(ode_newparams=ode_newparams)

    def _init_solver(self, ode_newparams=None):
        from scipy.integrate import ode
        from scipy.sparse import csc_matrix

        # Pre-inialize rate matrices
        self._update_jacobian(self.initial_z)
        self.current_z_rates = self.initial_z

        # Reset counters
        self.ncallsf = 0
        self.ncallsj = 0

        # Expand solver parameters with run-dependent options
        ode_params = config["ode_params"]
        ode_params['ndim'] = self.dim_states
        ode_params['nnz'] = self.jacobian.nnz
        if ode_newparams is not None:
            ode_params.update(ode_newparams)

        # info(1,'Setting solver with jacobian')
        # self.r = ode(self.eqn_deriv, self.eqn_jac).set_integrator(**ode_params)
        info(1, 'Setting solver without jacobian')
        self.r = ode(self.eqn_deriv).set_integrator(**ode_params)

    def set_initial_condition(self, spectrum=None, nco_id=None):
        self.state *= 0.
        self.result = None
        if spectrum is not None and nco_id is not None:
            sp = self.prince_run.spec_man.ncoid2sref[nco_id]
            self.state[sp.lidx():sp.uidx()] = spectrum
        # Initial value
        self.r.set_initial_value(self.state, self.initial_z)
        self._update_jacobian(self.initial_z)

    def solve(self,
              dz=1e-3,
              verbose=True,
              extended_output=False,
              full_reset=False,
              progressbar=False):
        from time import time

        reset_counter = 0
        stepcount = 0
        dz = -1 * dz
        start_time = time()

        initial_state = np.zeros(self.dim_states)
        self.r.set_initial_value(initial_state, self.r.t)

        if progressbar:
            if progressbar == 'notebook':
                from tqdm import tqdm_notebook as tqdm
            else:
                from tqdm import tqdm
            pbar = tqdm(total=-1 * int((self.r.t - self.z_offset) / dz))
            pbar.update()
        else:
            pbar = None

        info(2, 'Starting integration.')
        while self.r.successful() and (self.r.t + dz) > self.final_z:
            # print '------ at', self.r.t, '------'
            if verbose:
                info(3, "Integrating at z = {0}".format(self.r.t))
            step_start = time()
            # --------------------------------------------
            # Solve for hadronic interactions
            # --------------------------------------------
            if verbose:
                print 'Solving hadr losses at t =', self.r.t

            self.r.integrate(self.r.t + dz, relax=False, step=False)
            if verbose:
                self.print_step_info(
                    step_start, extended_output=extended_output)
            # print '------ numc', self.ncallsf, '------'
            # print ' '
            self.ncallsf = 0
            self.ncallsj = 0
            # --------------------------------------------
            # Apply the injection
            # --------------------------------------------
            if not self.enable_injection_jacobian and not self.enable_partial_diff_jacobian:
                print 'injection incoming'
                if verbose:
                    print 'applying injection at t =', self.r.t
                if full_reset:
                    if type(full_reset) is int and reset_counter == full_reset:
                        # self.r.set_initial_value(
                        #     self.r.y + self.injection(dz, self.r.t), self.r.t)
                        self.r._integrator.call_args[3] = 1
                        self.r._y += self.injection(dz, self.r.t)
                        reset_counter = 0
                        print 'resetting solver inj'
                    if type(full_reset) is bool:
                        self.r.set_initial_value(
                            self.r.y + self.injection(dz, self.r.t), self.r.t)
                    else:
                        self.r._integrator.call_args[3] = 20
                        self.r._y += self.injection(dz, self.r.t)
                else:
                    self.r._integrator.call_args[3] = 20
                    self.r._y += self.injection(dz, self.r.t)

            # --------------------------------------------
            # Apply the semi lagrangian
            # --------------------------------------------
            if not self.enable_partial_diff_jacobian:
                print 'semi lag incoming'
                if self.enable_adiabatic_losses or self.enable_pairprod_losses:
                    if verbose:
                        print 'applying semi lagrangian at t =', self.r.t
                    if full_reset:
                        if type(full_reset
                                ) is int and reset_counter == full_reset:
                            self.r.set_initial_value(
                                self.semi_lagrangian(dz, self.r.t, self.r.y),
                                self.r.t)
                            print reset_counter
                            reset_counter = 0
                            print 'resetting solver lag'
                        if type(full_reset) is bool:
                            self.r.set_initial_value(
                                self.semi_lagrangian(dz, self.r.t, self.r.y),
                                self.r.t)
                        else:
                            self.r._integrator.call_args[3] = 20
                            self.r._y = self.semi_lagrangian(
                                dz, self.r.t, self.r.y)
                    else:
                        self.r._integrator.call_args[3] = 20
                        self.r._y = self.semi_lagrangian(
                            dz, self.r.t, self.r.y)
            # --------------------------------------------
            # Some last checks and resets
            # --------------------------------------------
            if self.r.t < -1 * dz:
                print 'break at z =', self.r.t
                break

            print 'time', self.r.t, self.r._integrator.rwork[12]
            print 'last_step', self.r._integrator.rwork[10]
            print 'next_step', self.r._integrator.rwork[11]
            print 'tolerance_scale', self.r._integrator.rwork[13]
            # print 'nsteps', self.r._integrator.iwork[10]
            # print 'nderiv', self.r._integrator.iwork[11]
            print 'last_order', self.r._integrator.iwork[13]
            print 'next_order', self.r._integrator.iwork[13]
            print '---' * 20
            if verbose:
                print '---' * 20

            stepcount += 1
            reset_counter += 1
            if pbar is not None:
                pbar.update()
        self.r.integrate(self.final_z)
        if pbar is not None:
            pbar.update()
        if pbar is not None:
            pbar.close()
        if not self.r.successful():
            raise Exception(
                'Integration failed. Change integrator setup and retry.')

        self.state = self.r.y
        end_time = time()
        info(2, 'Integration completed in {0} s'.format(end_time - start_time))

    def print_step_info(self, step_start, extended_output=False):
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
            print 'LAST STEP {0:4.3e}'.format(self.r._integrator.rwork[10])


class UHECRPropagationSolverEULER(UHECRPropagationSolver):
    def __init__(*args, **kwargs):
        super(UHECRPropagationSolverEULER, self).__init__(*args, **kwargs)
        # UHECRPropagationSolver.__init__(self,*args,**kwargs)

    def solve(self, dz=1e-3, verbose=True, extended_output=False):
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

            state += self.eqn_deriv(curr_z, state) * dz

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
                    state = self.semi_lagrangian(dz, curr_z, state)

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


class UHECRPropagationSolverBDF(UHECRPropagationSolver):
    def __init__(self, *args, **kwargs):
        self.atol = kwargs.pop('atol',1e40)
        self.rtol = kwargs.pop('rtol',1e-10)
        super(UHECRPropagationSolverBDF, self).__init__(*args, **kwargs)
        # UHECRPropagationSolver.__init__(self,*args,**kwargs)

    def _init_solver(self, dz):
        initial_state = np.zeros(self.dim_states)

        self._update_jacobian(self.initial_z)
        self.current_z_rates = self.initial_z

        # find the maximum injection and reduce the system by this
        self.red_idx = np.nonzero(self.injection(1., 0.))[0].max()

        from scipy.integrate import BDF
        self.r = BDF(
            self.eqn_deriv,
            self.initial_z,
            initial_state,
            self.final_z,
            max_step=np.abs(dz),
            atol=self.atol,
            rtol=self.rtol,
            #  jac = self.eqn_jac,
            jac_sparsity=self.eqn_jac(self.initial_z, initial_state),
            vectorized=True)

    def solve(self,
              dz=1e-3,
              verbose=True,
              extended_output=False,
              full_reset=False,
              progressbar=False):
        from time import time

        reset_counter = 0
        stepcount = 0
        dz = -1 * dz
        start_time = time()

        info(2, 'Setting up Solver')
        self._init_solver(dz)
        info(2, 'Solver initialized in {0} s'.format(time() - start_time))

        if progressbar:
            if progressbar == 'notebook':
                from tqdm import tqdm_notebook as tqdm
            else:
                from tqdm import tqdm
            pbar = tqdm(total=-(self.initial_z - self.final_z) / dz)
            pbar.update()
        else:
            pbar = None

        info(2, 'Starting integration.')
        while self.r.status == 'running':
            # print '------ at', self.r.t, '------'
            if verbose:
                info(3, "Integrating at z = {0}".format(self.r.t))
            step_start = time()
            # --------------------------------------------
            # Solve for hadronic interactions
            # --------------------------------------------
            if verbose:
                print 'Solving hadr losses at t =', self.r.t
            self.r.step()
            # --------------------------------------------
            # Some last checks and resets
            # --------------------------------------------
            if verbose:
                print 'last step:', self.r.step_size
                print 'LU decomp:', self.r.nlu
                print 'current order:', self.r.dense_output().order
                print '---' * 20

            stepcount += 1
            reset_counter += 1
            if pbar is not None:
                pbar.update()

        if self.r.status == 'failed':
            raise Exception(
                'Integrator failed at t = {:}, try adjusting the tolerances'.
                format(self.r.t))
        if verbose:
            print 'Integrator finished with t = {:}, last step was dt = {:}'.format(
                self.r.t, self.r.step_size)

        # after each run we delete the solver to save memory
        self.state = self.r.y.copy()

        print 'Summary:'
        print '---------------------'
        print 'status:   ', self.r.status
        print 'time:     ', self.r.t
        print 'last step:', self.r.step_size
        print 'RHS eval: ', self.r.nfev
        print 'Jac eval: ', self.r.njev
        print 'LU decomp:', self.r.nlu
        print '---------------------'

        del self.r

        end_time = time()
        info(2, 'Integration completed in {0} s'.format(end_time - start_time))
