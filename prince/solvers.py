"""Contains solvers, i.e. integrators, kernels, steppers, for PriNCe."""

import numpy as np
from prince.cosmology import H
from prince.util import info, PRINCE_UNITS, get_AZN, EnergyGrid
from prince_config import config

class UHECRPropagationSolver(object):
    def __init__(self, initial_z, final_z, prince_run,
                 enable_adiabatic_losses=True,
                 enable_pairprod_losses=True,
                 enable_photohad_losses=True,
                 enable_injection_jacobian=True
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

        self.had_int_rates = self.prince_run.int_rates
        self.adia_loss_rates_grid = self.prince_run.adia_loss_rates_grid
        self.pair_loss_rates_grid = self.prince_run.pair_loss_rates_grid 
        self.adia_loss_rates_bins = self.prince_run.adia_loss_rates_bins
        self.pair_loss_rates_bins = self.prince_run.pair_loss_rates_bins
        self.intp = None

        self.state = np.zeros(prince_run.dim_states)
        self.dim_states = prince_run.dim_states

        self.solution_vector = []
        self.list_of_sources = []
        self._init_solver()

    def add_source_class(self, source_instance):
        self.list_of_sources.append(source_instance)

    # def get_solution(self, nco_id):
    #     """Returns the spectrum in energy per nucleon"""
    #     sp = self.prince_run.spec_man.ncoid2sref[nco_id]
    #     return self.egrid, self.r.y[sp.lidx():sp.uidx()]

    # def get_solution_scale(self, nco_id, epow=0):
    #     """Returns the spectrum scaled back to total energy"""
    #     spec = self.prince_run.spec_man.ncoid2sref[nco_id]
    #     egrid = spec.A * self.egrid
    #     return egrid, egrid**epow * self.r.y[
    #         spec.lidx():spec.uidx()] / spec.A

    def get_solution(self, nco_id):
        """Returns the spectrum in energy per nucleon"""
        sp = self.prince_run.spec_man.ncoid2sref[nco_id]
        return self.egrid, self.state[sp.lidx():sp.uidx()]

    def get_solution_scale(self, nco_id, epow=0):
        """Returns the spectrum scaled back to total energy"""
        spec = self.prince_run.spec_man.ncoid2sref[nco_id]
        egrid = spec.A * self.egrid
        return egrid, egrid**epow * self.state[
            spec.lidx():spec.uidx()] / spec.A

    def get_energy_density(self, nco_id):
        from scipy.integrate import trapz

        A, _, _ = get_AZN(nco_id)
        return trapz(A * self.egrid * self.get_solution(nco_id), self.egrid)

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
        # -------------------------------------------------------------------
        # method 1:
        # - shift the bin centers and use gradient for derivative
        # - use linear interpolation on x = log(E)
        # -------------------------------------------------------------------
        if config["semi_lagr_method"] == 'method1':
            # add the different contributions to continuous losses
            conloss = np.zeros_like(self.adia_loss_rates_grid.energy_vector)
            if self.enable_adiabatic_losses:
                conloss += self.adia_loss_rates_grid.loss_vector(z)
            if self.enable_pairprod_losses:
                conloss += self.pair_loss_rates_grid.loss_vector(z)
            conloss *= self.dldz(z) * delta_z 
            for spec in self.spec_man.species_refs:
                lidx, uidx = spec.lidx(), spec.uidx()
                newgrid = self.egrid - conloss[lidx:uidx]
                oldgrid = self.egrid

                newgrid_log = np.log(newgrid)
                oldgrid_log = np.log(oldgrid)
                gradient = np.gradient(newgrid_log,oldgrid_log) * newgrid / oldgrid

                newstate = state[lidx:uidx] / gradient
                newstate = np.where(newstate > 1e-200, newstate, 1e-200)
                newstate_log = np.log(newstate)

                state[lidx:uidx] = np.exp(np.interp(oldgrid_log, newgrid_log, newstate_log))
        # -------------------------------------------------------------------
        # method 2:
        # - shift the bin edges and bin widths for derivative
        # - use linear interpolation on x = log(E)
        # -------------------------------------------------------------------
        elif config["semi_lagr_method"] == 'method2':
            conloss = np.zeros_like(self.adia_loss_rates_bins.energy_vector)
            if self.enable_adiabatic_losses:
                conloss += self.adia_loss_rates_bins.loss_vector(z)
            if self.enable_pairprod_losses:
                conloss += self.pair_loss_rates_bins.loss_vector(z)
            conloss *= self.dldz(z) * delta_z 
            for spec in self.spec_man.species_refs:
                lbin, ubin = spec.lbin(), spec.ubin()
                lidx, uidx = spec.lidx(), spec.uidx()
                newbins = self.ebins - conloss[lbin:ubin]
                oldbins = self.ebins

                newcenters = (newbins[1:] + newbins[:-1])/2
                newwidths  = (newbins[1:] - newbins[:-1])
                oldcenters = (oldbins[1:] + oldbins[:-1])/2
                oldwidths  = (oldbins[1:] - oldbins[:-1])

                newgrid_log = np.log(newcenters)
                oldgrid_log = np.log(oldcenters)
                gradient = newwidths / oldwidths

                newstate = state[lidx:uidx] / gradient
                newstate = np.where(newstate > 1e-200, newstate, 1e-200)
                newstate_log = np.log(newstate)

                # if spec.ncoid == 101:
                #     print newstate_log
                #     print newstate_log[1:] - newstate_log[:-1]
                #     print '---'*30

                state[lidx:uidx] = np.exp(np.interp(oldgrid_log, newgrid_log, newstate_log))

        # -------------------------------------------------------------------
        # method 3:
        # - shift the bin edges and bin widths for derivative
        # - second order derivative for interpolation
        # -------------------------------------------------------------------
        elif config["semi_lagr_method"] == 'method3':
            conloss = np.zeros_like(self.adia_loss_rates_bins.energy_vector)
            if self.enable_adiabatic_losses:
                conloss += self.adia_loss_rates_bins.loss_vector(z)
            if self.enable_pairprod_losses:
                conloss += self.pair_loss_rates_bins.loss_vector(z)
            conloss *= self.dldz(z) * delta_z 
            # print 'using new semi lagrangian'
            for spec in self.spec_man.species_refs:
                lbin, ubin = spec.lbin(), spec.ubin()
                lidx, uidx = spec.lidx(), spec.uidx()
                newbins = self.ebins - conloss[lbin:ubin]
                oldbins = self.ebins

                newcenters = (newbins[1:] + newbins[:-1])/2
                newwidths  = (newbins[1:] - newbins[:-1])
                oldcenters = (oldbins[1:] + oldbins[:-1])/2
                oldwidths  = (oldbins[1:] - oldbins[:-1])

                newgrid_log = np.log(newcenters)
                oldgrid_log = np.log(oldcenters)
                gradient = newwidths / oldwidths

                newstate = state[lidx:uidx] / gradient
                newstate = np.where(newstate > 1e-200, newstate, 1e-200)
                newstate_log = np.log(newstate)

                # deriv = np.zeros_like(newstate_log)
                # deriv[:-1] = (newstate_log[1:] - newstate_log[:-1]) / (newgrid_log[1:] - newgrid_log[:-1])
                # deriv[-1]  = (newstate_log[-1] - newstate_log[-2]) / (newgrid_log[-1] - newgrid_log[-2])


                deriv2 = np.gradient(newstate_log,newgrid_log,edge_order=2)
                # if spec.ncoid == 101:
                #     print deriv
                #     print deriv
                #     print deriv - deriv2
                #     print '-------'
                state[lidx:uidx] = np.exp(newstate_log + deriv2 * (oldgrid_log - newgrid_log))
                # state[lidx:uidx] = np.exp(np.interp(oldgrid_log, newgrid_log, newstate_log))

        # -------------------------------------------------------------------
        # method x:
        # - shift the bin edges and bin widths for derivative
        # - use moment conserving special interpolator
        # -------------------------------------------------------------------
        elif config["semi_lagr_method"] == 'methodx':
            if self.intp is None:
                from prince.util import TheInterpolator
                intp = TheInterpolator(self.ebins)
                self.intp = intp

            conloss = np.zeros_like(self.adia_loss_rates_bins.energy_vector)
            if self.enable_adiabatic_losses:
                conloss += self.adia_loss_rates_bins.loss_vector(z)
            if self.enable_pairprod_losses:
                conloss += self.pair_loss_rates_bins.loss_vector(z)
            conloss *= self.dldz(z) * delta_z 

            for spec in self.spec_man.species_refs:
                lbin, ubin = spec.lbin(), spec.ubin()
                lidx, uidx = spec.lidx(), spec.uidx()
                newbins = self.ebins - conloss[lbin:ubin]
                newcenters = (newbins[1:] + newbins[:-1])/2
                newwidths  = (newbins[1:] - newbins[:-1])
                self.intp.set_initial_spectrum(
                newcenters,state[lidx:uidx]*newwidths)
                
                state[lidx:uidx] = self.intp.get_solution()

        # -------------------------------------------------------------------
        # method xx:
        # - two bin interpolation
        # -------------------------------------------------------------------
        elif config["semi_lagr_method"] == 'methodxx':
            # print 'using new interpolator'
            conloss = np.zeros_like(self.adia_loss_rates_bins.energy_vector)
            if self.enable_adiabatic_losses:
                conloss += self.adia_loss_rates_bins.loss_vector(z)
            if self.enable_pairprod_losses:
                conloss += self.pair_loss_rates_bins.loss_vector(z)
            conloss *= self.dldz(z) * delta_z 
            for spec in self.spec_man.species_refs:
                lbin, ubin = spec.lbin(), spec.ubin()
                lidx, uidx = spec.lidx(), spec.uidx()
                newbins = self.ebins - conloss[lbin:ubin]
                oldbins = self.ebins

                newcenters = (newbins[1:] + newbins[:-1])/2
                newwidths  = (newbins[1:] - newbins[:-1])
                oldcenters = (oldbins[1:] + oldbins[:-1])/2
                oldwidths  = (oldbins[1:] - oldbins[:-1])

                newgrid_log = np.log(newcenters)
                oldgrid_log = np.log(oldcenters)
                newbins_log = np.log(newbins)
                oldbins_log = np.log(oldbins)
                gradient = newwidths / oldwidths

                newstate = state[lidx:uidx] / gradient
                # newstate = np.where(newstate > 1e-200, newstate, 1e-200)
                newstate_log = np.log(newstate)

                # delta = (newbins_log[1:] - oldbins_log[1:]) / (newbins_log[1:] - newbins_log[:-1])
                # state[lidx:uidx] = np.exp(newstate_log * (1-delta) + newstate_log * delta)

                # delta = (newbins[1:] - oldbins[1:]) / (newbins[1:] - newbins[:-1])
                # state[lidx:uidx] = newstate * (1-delta) + newstate * delta

                # print 'integrals:'
                # print np.sum(state[lidx:uidx] * oldwidths), np.sum(state[lidx:uidx] * oldcenters * oldwidths)
                # print np.sum(newstate * newwidths), np.sum(newstate * newcenters * newwidths)

                # delta = (newbins[1:] - oldbins[:-1]) / oldwidths
                # finalstate = newstate[:-1] * (1 - delta[:-1]) + newstate[1:] * delta[:-1]
                # state[lidx:uidx-1] = finalstate
                # state[uidx-1:uidx] = 0.
                state[lidx:uidx] = np.exp(np.interp(oldgrid_log, newgrid_log, newstate_log))

        # -------------------------------------------------------------------
        # if method was not in list before, raise an Expection
        # -------------------------------------------------------------------
        else:
            raise Exception('Unknown semi-lagrangian method ({:})'.format(config["semi_lagr_method"]))

        return state

    # def conloss_deriv(self, z, state, delta_z=1e-4):
    #     conloss = self.continuous_losses * delta_z * self.dldz(z)
    #     conloss_deriv = np.zeros_like(state)
    #     for spec in self.spec_man.species_refs:
    #         lidx, uidx = spec.lidx(), spec.uidx()
    #         sup = np.interp(self.egrid, self.egrid + conloss[lidx:uidx],
    #                         state[lidx:uidx])
    #         sd = np.interp(self.egrid, self.egrid - conloss[lidx:uidx],
    #                        state[lidx:uidx])
    #         conloss_deriv[lidx:uidx] = (sup - sd) / (2. * delta_z)
    #     return -conloss_deriv

    def eqn_deriv(self, z, state, *args):
        self.ncallsf += 1
        if self.enable_injection_jacobian:
            r = self.jacobian.dot(state) + self.injection(1., z)
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

        ode_params_lsodes = {
            'name': 'lsodes',
            'method': 'bdf',
            # 'nsteps': 10000,
            'rtol': 1e-4,
            # 'atol': 1e5,
            'ndim': self.dim_states,
            'nnz': self.jacobian.nnz,
            # 'csc_jacobian': self.jacobian.tocsc(),
            # 'max_order_s': 5,
            # 'with_jacobian': True
        }

        # ode_params_vode = {
        #     'name': 'vode',
        #     'method': 'bdf',
        #     'nsteps': 10000,
        #     'rtol': 1e-1,
        #     'atol': 1e10,
        #     # 'order': 5,
        #     'max_step': 0.2,
        #     'with_jacobian': False
        # }

        # ode_params_lsoda = {
        #     'name': 'lsoda',
        #     'nsteps': 10000,
        #     'rtol': 0.2,
        #     'max_order_ns': 5,
        #     'max_order_s': 2,
        #     'max_step': 0.2,
        #     # 'first_step': 1e-4,
        # }

        ode_params = ode_params_lsodes

        # Setup solver

        # info(1,'Setting solver with jacobian')
        # self.r = ode(self.eqn_deriv, self.eqn_jac).set_integrator(**ode_params)
        info(1,'Setting solver without jacobian')
        self.r = ode(self.eqn_deriv).set_integrator(**ode_params)

    def solve(self, dz=1e-2, verbose=True, extended_output=False, full_reset=False):
        from time import time

        # stepcount = 0
        dz = -1 * dz
        start_time = time()
        self.r.set_initial_value(np.zeros(self.dim_states), self.r.t)

        info(2, 'Starting integration.')

        while self.r.successful() and (self.r.t + dz) > self.final_z:
            if verbose:
                info(3, "Integrating at z={0}".format(self.r.t))
            step_start = time()
            # --------------------------------------------
            # Solve for hadronic interactions
            # --------------------------------------------
            if verbose:
                print 'Solving hadr losses at t=', self.r.t
            self._update_jacobian(self.r.t)
            self.r.integrate(self.r.t + dz)  #,
            if verbose:
                print 'step took', time() - step_start
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
            # --------------------------------------------
            # Apply the injection 
            # --------------------------------------------
            if not self.enable_injection_jacobian:
                if verbose:
                    print 'applying injection at t=', self.r.t
                if full_reset:
                    self.r.set_initial_value(self.r.y + self.injection(dz, self.r.t), self.r.t)
                else:
                    self.r._integrator.call_args[3] = 20
                    self.r._y += self.injection(dz, self.r.t)

            # --------------------------------------------
            # Apply the semi lagrangian
            # --------------------------------------------
            if self.enable_adiabatic_losses or self.enable_pairprod_losses:
                if verbose:
                    print 'applying semi lagrangian at t=', self.r.t
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
            # if stepcount == -1:
            #     full_reset = True
            #     stepcount = 0
            # else:
            #     stepcount += 1
            # print 'reset',full_reset

        self.r.integrate(self.final_z)

        if not self.r.successful():
            raise Exception(
                'Integration failed. Change integrator setup and retry.')

        self.state = self.r.y
        end_time = time()
        info(2, 'Integration completed in {0} s'.format(end_time - start_time))

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
                print 'Solving hadr losses at t=', self.r.t
            self._update_jacobian(curr_z)

            state += self.eqn_deriv(curr_z, state)*dz

            # --------------------------------------------
            # Apply the injection 
            # --------------------------------------------
            if not self.enable_injection_jacobian:
                if verbose:
                    print 'applying injection at t=', self.r.t
                state += self.injection(dz, curr_z)
            
            # --------------------------------------------
            # Apply the semi lagrangian
            # --------------------------------------------
            if self.enable_adiabatic_losses or self.enable_pairprod_losses:
                if verbose:
                    print 'applying semi lagrangian at t=', self.r.t
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