
import numpy as np
import prince_cr.config as config

class SemiLagrangianSolver(object):
    """Contains routines to project spectra from shifted grids back to old grid"""

    def __init__(self, cr_grid):
        # remember energy grid to project back to
        self.grid = cr_grid
        self.widths = cr_grid.widths
        self.grid_log = np.log(cr_grid.grid)
        self.bins_log = np.log(cr_grid.bins)

    def get_shifted_state(self, conloss, state):
        newbins = self.grid.bins - conloss

        # get shifted state from gradient
        newwidths  = (newbins[1:] - newbins[:-1])
        gradient = newwidths / self.widths
        newstate = state / gradient
        newstate = np.where(newstate > 1e-250, newstate, 1e-250)
        newstate_log = np.log(newstate)

        # get new centers
        newcenters = (newbins[1:] + newbins[:-1])/2
        newgrid_log = np.log(newcenters)

        return newgrid_log, newstate_log

    def interpolate(self, conloss, state):
        "Uses linear interpolation to find the new state old grid"
        # get shifted state
        newgrid_log, newstate_log = self.get_shifted_state(conloss, state)
        
        # return interpolated value as newstate
        return np.exp(np.interp(self.grid_log, newgrid_log, newstate_log))

    def interpolate_gradient(self, conloss, state):
        """Uses a linear approximation arround x_i to find the new state old grid"""
        # get shifted state
        newgrid_log, newstate_log = self.get_shifted_state(conloss, state)
        
        # find second order finite difference at x_i
        # and use for a linear approximation arround x_i
        deriv = np.gradient(newstate_log, newgrid_log, edge_order=2)
        return np.exp(newstate_log + deriv * (self.grid_log - newgrid_log))

    def interpolate_linear_weights(self, conloss, state):
        """Uses linear interpolation with lagrange polynomials
        (same as interpolate, but directly implemented for testing)"""
        # get shifted state
        newgrid_log, newstate_log = self.get_shifted_state(conloss, state)
        
        # slices for indexing
        i0 = slice(None,-1)
        i1 = slice(1,None)
        # # calculate weights from old and new grid
        weight0 = (self.grid_log[i0] - newgrid_log[i1]) / (newgrid_log[i0] - newgrid_log[i1])
        weight1 = (self.grid_log[i0] - newgrid_log[i0]) / (newgrid_log[i1] - newgrid_log[i0])

        # calculate state by interpolating with weights
        state = np.zeros_like(self.grid_log)
        state[i0] = np.exp(weight0 * newstate_log[i0] + weight1 * newstate_log[i1])
        return state

    def interpolate_quadratic_weights(self, conloss, state):
        """Uses quadratic interpolation with lagrange polynomials"""
        # get shifted state
        newgrid_log, newstate_log = self.get_shifted_state(conloss, state)
        
        i0 = slice(None,-3)
        i1 = slice(1,-2)
        i2 = slice(2,-1)
        i3 = slice(3,None)

        # calculate backward weights from old and new grid
        weight_b0 = ((self.grid_log[i1] - newgrid_log[i1]) / (newgrid_log[i0] - newgrid_log[i1])
                    *(self.grid_log[i1] - newgrid_log[i2]) / (newgrid_log[i0] - newgrid_log[i2])
                    )
        weight_b1 = ((self.grid_log[i1] - newgrid_log[i0]) / (newgrid_log[i1] - newgrid_log[i0])
                    *(self.grid_log[i1] - newgrid_log[i2]) / (newgrid_log[i1] - newgrid_log[i2])
                    )
        weight_b2 = ((self.grid_log[i1] - newgrid_log[i0]) / (newgrid_log[i2] - newgrid_log[i0])
                    *(self.grid_log[i1] - newgrid_log[i1]) / (newgrid_log[i2] - newgrid_log[i1])
                    )

        weight_f1 = ((self.grid_log[i1] - newgrid_log[i2]) / (newgrid_log[i1] - newgrid_log[i2])
                    *(self.grid_log[i1] - newgrid_log[i3]) / (newgrid_log[i1] - newgrid_log[i3])
                    )
        weight_f2 = ((self.grid_log[i1] - newgrid_log[i1]) / (newgrid_log[i2] - newgrid_log[i1])
                    *(self.grid_log[i1] - newgrid_log[i3]) / (newgrid_log[i2] - newgrid_log[i3])
                    )
        weight_f3 = ((self.grid_log[i1] - newgrid_log[i1]) / (newgrid_log[i3] - newgrid_log[i1])
                    *(self.grid_log[i1] - newgrid_log[i2]) / (newgrid_log[i3] - newgrid_log[i2])
                    )

        # calculate state by interpolating with weights
        state = np.zeros_like(self.grid_log)
        # backward only
        # state[i1] = np.exp(weight_b0 * newstate_log[i0] + weight_b1 * newstate_log[i1] + weight_b2 * newstate_log[i2])
        # forward only
        # state[i1] = np.exp(weight_f1 * newstate_log[i1] + weight_f2 * newstate_log[i2] + weight_f3 * newstate_log[i3])

        # forward and backward average
        state[i1] = np.exp(weight_b0 / 2 * newstate_log[i0] + (weight_f1 + weight_b1) / 2 * newstate_log[i1]
                          + (weight_f2 + weight_b2) / 2 * newstate_log[i2] + weight_f3 / 2 * newstate_log[i3]
                          )
        return state

    def interpolate_cubic_weights(self, conloss, state):
        """Uses cubic interpolation with lagrange polynomials"""
        # get shifted state
        newgrid_log, newstate_log = self.get_shifted_state(conloss, state)
        
        i0 = slice(None,-3)
        i1 = slice(1,-2)
        i2 = slice(2,-1)
        i3 = slice(3,None)

        # calculate weights from old and new grid
        weight0 = ((self.grid_log[i1] - newgrid_log[i1]) / (newgrid_log[i0] - newgrid_log[i1])
                    *(self.grid_log[i1] - newgrid_log[i2]) / (newgrid_log[i0] - newgrid_log[i2])
                    *(self.grid_log[i1] - newgrid_log[i3]) / (newgrid_log[i0] - newgrid_log[i3])
                    )
        weight1 = ((self.grid_log[i1] - newgrid_log[i0]) / (newgrid_log[i1] - newgrid_log[i0])
                    *(self.grid_log[i1] - newgrid_log[i2]) / (newgrid_log[i1] - newgrid_log[i2])
                    *(self.grid_log[i1] - newgrid_log[i3]) / (newgrid_log[i1] - newgrid_log[i3])
                    )
        weight2 = ((self.grid_log[i1] - newgrid_log[i0]) / (newgrid_log[i2] - newgrid_log[i0])
                    *(self.grid_log[i1] - newgrid_log[i1]) / (newgrid_log[i2] - newgrid_log[i1])
                    *(self.grid_log[i1] - newgrid_log[i3]) / (newgrid_log[i2] - newgrid_log[i3])
                    )
        weight3 = ((self.grid_log[i1] - newgrid_log[i0]) / (newgrid_log[i3] - newgrid_log[i0])
                    *(self.grid_log[i1] - newgrid_log[i1]) / (newgrid_log[i3] - newgrid_log[i1])
                    *(self.grid_log[i1] - newgrid_log[i2]) / (newgrid_log[i3] - newgrid_log[i2])
                    )

        # reverse0 = ((newgrid_log[i2] - self.grid_log[i1]) / (self.grid_log[i0] - self.grid_log[i1])
        #             *(newgrid_log[i2] - self.grid_log[i2]) / (self.grid_log[i0] - self.grid_log[i2])
        #             *(newgrid_log[i2] - self.grid_log[i3]) / (self.grid_log[i0] - self.grid_log[i3])
        #             )
        # reverse1 = ((newgrid_log[i2] - self.grid_log[i0]) / (self.grid_log[i1] - self.grid_log[i0])
        #             *(newgrid_log[i2] - self.grid_log[i2]) / (self.grid_log[i1] - self.grid_log[i2])
        #             *(newgrid_log[i2] - self.grid_log[i3]) / (self.grid_log[i1] - self.grid_log[i3])
        #             )
        # reverse2 = ((newgrid_log[i2] - self.grid_log[i0]) / (self.grid_log[i2] - self.grid_log[i0])
        #             *(newgrid_log[i2] - self.grid_log[i1]) / (self.grid_log[i2] - self.grid_log[i1])
        #             *(newgrid_log[i2] - self.grid_log[i3]) / (self.grid_log[i2] - self.grid_log[i3])
        #             )
        # reverse3 = ((newgrid_log[i2] - self.grid_log[i0]) / (self.grid_log[i3] - self.grid_log[i0])
        #             *(newgrid_log[i2] - self.grid_log[i1]) / (self.grid_log[i3] - self.grid_log[i1])
        #             *(newgrid_log[i2] - self.grid_log[i2]) / (self.grid_log[i3] - self.grid_log[i2])
        #             )


        # weight_sum = weight0[i3] + weight1[i2] + weight2[i1] + weight3[i0]
        # weight_sum = weight0[i0] + weight1[i1] + weight2[i2] + weight3[i3]

        # if spid == 101:
        #     print (1 - weight_sum) 
        # weight0[i3] += (1 - weight_sum) * reverse0[i3]
        # weight1[i2] += (1 - weight_sum) * reverse1[i2]
        # weight2[i1] += (1 - weight_sum) * reverse2[i1]
        # weight3[i0] += (1 - weight_sum) * reverse3[i0]

        # weight0[i0] += (1 - weight_sum) * reverse0[i0]
        # weight1[i1] += (1 - weight_sum) * reverse1[i1]
        # weight2[i2] += (1 - weight_sum) * reverse2[i2]
        # weight3[i3] += (1 - weight_sum) * reverse3[i3]
        state = np.zeros_like(self.grid_log)
        state[i1] = np.exp( weight0 * newstate_log[i0] + weight1 * newstate_log[i1]
                          + weight2 * newstate_log[i2] + weight3 * newstate_log[i3]
                          )
        return state

    def interpolate_4thorder_weights(self, conloss, state):
        """Uses quadratic interpolation with lagrange polynomials"""
        # get shifted state
        newgrid_log, newstate_log = self.get_shifted_state(conloss, state)
        
        i0 = slice(None,-5)
        i1 = slice(1,-4)
        i2 = slice(2,-3)
        i3 = slice(3,-2)
        i4 = slice(4,-1)
        # i5 = slice(5,None)

        # calculate backward weights from old and new grid
        weight_b0 = ((self.grid_log[i2] - newgrid_log[i1]) / (newgrid_log[i0] - newgrid_log[i1])
                    *(self.grid_log[i2] - newgrid_log[i2]) / (newgrid_log[i0] - newgrid_log[i2])
                    *(self.grid_log[i2] - newgrid_log[i3]) / (newgrid_log[i0] - newgrid_log[i3])
                    *(self.grid_log[i2] - newgrid_log[i4]) / (newgrid_log[i0] - newgrid_log[i4])
                    )
        weight_b1 = ((self.grid_log[i2] - newgrid_log[i0]) / (newgrid_log[i1] - newgrid_log[i0])
                    *(self.grid_log[i2] - newgrid_log[i2]) / (newgrid_log[i1] - newgrid_log[i2])
                    *(self.grid_log[i2] - newgrid_log[i3]) / (newgrid_log[i1] - newgrid_log[i3])
                    *(self.grid_log[i2] - newgrid_log[i4]) / (newgrid_log[i1] - newgrid_log[i4])
                    )
        weight_b2 = ((self.grid_log[i2] - newgrid_log[i0]) / (newgrid_log[i2] - newgrid_log[i0])
                    *(self.grid_log[i2] - newgrid_log[i1]) / (newgrid_log[i2] - newgrid_log[i1])
                    *(self.grid_log[i2] - newgrid_log[i3]) / (newgrid_log[i2] - newgrid_log[i3])
                    *(self.grid_log[i2] - newgrid_log[i4]) / (newgrid_log[i2] - newgrid_log[i4])
                    )
        weight_b3 = ((self.grid_log[i2] - newgrid_log[i0]) / (newgrid_log[i3] - newgrid_log[i0])
                    *(self.grid_log[i2] - newgrid_log[i1]) / (newgrid_log[i3] - newgrid_log[i1])
                    *(self.grid_log[i2] - newgrid_log[i2]) / (newgrid_log[i3] - newgrid_log[i2])
                    *(self.grid_log[i2] - newgrid_log[i4]) / (newgrid_log[i3] - newgrid_log[i4])
                    )
        weight_b4 = ((self.grid_log[i2] - newgrid_log[i0]) / (newgrid_log[i4] - newgrid_log[i0])
                    *(self.grid_log[i2] - newgrid_log[i1]) / (newgrid_log[i4] - newgrid_log[i1])
                    *(self.grid_log[i2] - newgrid_log[i2]) / (newgrid_log[i4] - newgrid_log[i2])
                    *(self.grid_log[i2] - newgrid_log[i3]) / (newgrid_log[i4] - newgrid_log[i3])
                    )

        # weight_f1 = ((self.grid_log[i2] - newgrid_log[i2]) / (newgrid_log[i1] - newgrid_log[i2])
        #             *(self.grid_log[i2] - newgrid_log[i3]) / (newgrid_log[i1] - newgrid_log[i3])
        #             *(self.grid_log[i2] - newgrid_log[i4]) / (newgrid_log[i1] - newgrid_log[i4])
        #             *(self.grid_log[i2] - newgrid_log[i5]) / (newgrid_log[i1] - newgrid_log[i5])
        #             )
        # weight_f2 = ((self.grid_log[i2] - newgrid_log[i1]) / (newgrid_log[i2] - newgrid_log[i1])
        #             *(self.grid_log[i2] - newgrid_log[i3]) / (newgrid_log[i2] - newgrid_log[i3])
        #             *(self.grid_log[i2] - newgrid_log[i4]) / (newgrid_log[i2] - newgrid_log[i4])
        #             *(self.grid_log[i2] - newgrid_log[i5]) / (newgrid_log[i2] - newgrid_log[i5])
        #             )
        # weight_f3 = ((self.grid_log[i2] - newgrid_log[i1]) / (newgrid_log[i3] - newgrid_log[i1])
        #             *(self.grid_log[i2] - newgrid_log[i2]) / (newgrid_log[i3] - newgrid_log[i2])
        #             *(self.grid_log[i2] - newgrid_log[i4]) / (newgrid_log[i3] - newgrid_log[i4])
        #             *(self.grid_log[i2] - newgrid_log[i5]) / (newgrid_log[i3] - newgrid_log[i5])
        #             )
        # weight_f4 = ((self.grid_log[i2] - newgrid_log[i1]) / (newgrid_log[i4] - newgrid_log[i1])
        #             *(self.grid_log[i2] - newgrid_log[i2]) / (newgrid_log[i4] - newgrid_log[i2])
        #             *(self.grid_log[i2] - newgrid_log[i3]) / (newgrid_log[i4] - newgrid_log[i3])
        #             *(self.grid_log[i2] - newgrid_log[i5]) / (newgrid_log[i4] - newgrid_log[i5])
        #             )
        # weight_f5 = ((self.grid_log[i2] - newgrid_log[i1]) / (newgrid_log[i5] - newgrid_log[i1])
        #             *(self.grid_log[i2] - newgrid_log[i2]) / (newgrid_log[i5] - newgrid_log[i2])
        #             *(self.grid_log[i2] - newgrid_log[i3]) / (newgrid_log[i5] - newgrid_log[i3])
        #             *(self.grid_log[i2] - newgrid_log[i4]) / (newgrid_log[i5] - newgrid_log[i4])
        #             )

        # calculate state by interpolating with weights
        state = np.zeros_like(self.grid_log)
        # backward only
        state[i2] = np.exp(weight_b0 * newstate_log[i0] + weight_b1 * newstate_log[i1]
                   + weight_b2 * newstate_log[i2] + weight_b3 * newstate_log[i3] + weight_b4 * newstate_log[i4])
        # forward only
        # state[i2] = np.exp(weight_f1 * newstate_log[i1] + weight_f2 * newstate_log[i2]
        #            + weight_f3 * newstate_log[i3] + weight_f4 * newstate_log[i4] + weight_f5 * newstate_log[i5])
        # forward and backward average
        # state[i2] = np.exp(weight_b0 / 2 * newstate_log[i0] + (weight_f1 + weight_b1) / 2 * newstate_log[i1]
        #                   + (weight_f2 + weight_b2) / 2 * newstate_log[i2] + (weight_f3 + weight_b3) / 2 * newstate_log[i3] 
        #                   + (weight_f4 + weight_b4) / 2 * newstate_log[i4] + weight_f5 / 2 * newstate_log[i5]
        #                   )
        return state


    def interpolate_5thorder_weights(self, conloss, state):
        """Uses cubic interpolation with lagrange polynomials"""
        # get shifted state
        newgrid_log, newstate_log = self.get_shifted_state(conloss, state)
        
        i0 = slice(None,-5)
        i1 = slice(1,-4)
        i2 = slice(2,-3)
        i3 = slice(3,-2)
        i4 = slice(4,-1)
        i5 = slice(5,None)
        # calculate weights from old and new grid
        weight0 = ((self.grid_log[i2] - newgrid_log[i1]) / (newgrid_log[i0] - newgrid_log[i1])
                    *(self.grid_log[i2] - newgrid_log[i2]) / (newgrid_log[i0] - newgrid_log[i2])
                    *(self.grid_log[i2] - newgrid_log[i3]) / (newgrid_log[i0] - newgrid_log[i3])
                    *(self.grid_log[i2] - newgrid_log[i4]) / (newgrid_log[i0] - newgrid_log[i4])
                    *(self.grid_log[i2] - newgrid_log[i5]) / (newgrid_log[i0] - newgrid_log[i5])
                    )
        weight1 = ((self.grid_log[i2] - newgrid_log[i0]) / (newgrid_log[i1] - newgrid_log[i0])
                    *(self.grid_log[i2] - newgrid_log[i2]) / (newgrid_log[i1] - newgrid_log[i2])
                    *(self.grid_log[i2] - newgrid_log[i3]) / (newgrid_log[i1] - newgrid_log[i3])
                    *(self.grid_log[i2] - newgrid_log[i4]) / (newgrid_log[i1] - newgrid_log[i4])
                    *(self.grid_log[i2] - newgrid_log[i5]) / (newgrid_log[i1] - newgrid_log[i5])
                    )
        weight2 = ((self.grid_log[i2] - newgrid_log[i0]) / (newgrid_log[i2] - newgrid_log[i0])
                    *(self.grid_log[i2] - newgrid_log[i1]) / (newgrid_log[i2] - newgrid_log[i1])
                    *(self.grid_log[i2] - newgrid_log[i3]) / (newgrid_log[i2] - newgrid_log[i3])
                    *(self.grid_log[i2] - newgrid_log[i4]) / (newgrid_log[i2] - newgrid_log[i4])
                    *(self.grid_log[i2] - newgrid_log[i5]) / (newgrid_log[i2] - newgrid_log[i5])
                    )
        weight3 = ((self.grid_log[i2] - newgrid_log[i0]) / (newgrid_log[i3] - newgrid_log[i0])
                    *(self.grid_log[i2] - newgrid_log[i1]) / (newgrid_log[i3] - newgrid_log[i1])
                    *(self.grid_log[i2] - newgrid_log[i2]) / (newgrid_log[i3] - newgrid_log[i2])
                    *(self.grid_log[i2] - newgrid_log[i4]) / (newgrid_log[i3] - newgrid_log[i4])
                    *(self.grid_log[i2] - newgrid_log[i5]) / (newgrid_log[i3] - newgrid_log[i5])
                    )
        weight4 = ((self.grid_log[i2] - newgrid_log[i0]) / (newgrid_log[i4] - newgrid_log[i0])
                    *(self.grid_log[i2] - newgrid_log[i1]) / (newgrid_log[i4] - newgrid_log[i1])
                    *(self.grid_log[i2] - newgrid_log[i2]) / (newgrid_log[i4] - newgrid_log[i2])
                    *(self.grid_log[i2] - newgrid_log[i3]) / (newgrid_log[i4] - newgrid_log[i3])
                    *(self.grid_log[i2] - newgrid_log[i5]) / (newgrid_log[i4] - newgrid_log[i5])
                    )
        weight5 = ((self.grid_log[i2] - newgrid_log[i0]) / (newgrid_log[i5] - newgrid_log[i0])
                    *(self.grid_log[i2] - newgrid_log[i1]) / (newgrid_log[i5] - newgrid_log[i1])
                    *(self.grid_log[i2] - newgrid_log[i2]) / (newgrid_log[i5] - newgrid_log[i2])
                    *(self.grid_log[i2] - newgrid_log[i3]) / (newgrid_log[i5] - newgrid_log[i3])
                    *(self.grid_log[i2] - newgrid_log[i4]) / (newgrid_log[i5] - newgrid_log[i4])
                    )

        # calculate state by interpolating with weights
        state = np.zeros_like(self.grid_log)
        state[i2] = np.exp( weight0 * newstate_log[i0] + weight1 * newstate_log[i1]
                          + weight2 * newstate_log[i2] + weight3 * newstate_log[i3]
                          + weight4 * newstate_log[i4] + weight5 * newstate_log[i5]
                          )
        return state

class DifferentialOperator(object):
    def __init__(self, cr_grid, nspec):
        self.ebins = cr_grid.bins
        self.egrid = cr_grid.grid
        self.ewidths = cr_grid.widths
        self.dim_e = cr_grid.d
        # binsize in log(e)
        self.log_width = np.log(self.ebins[1] / self.ebins[0])

        self.nspec = nspec
        self.operator = self.construct_differential_operator()
        if config.linear_algebra_backend.lower() == 'cupy':
            import cupyx
            self.operator = cupyx.scipy.sparse.csr_matrix(self.operator)

    def construct_differential_operator(self):
        from scipy.sparse import coo_matrix, block_diag
        # # Construct a 
        # # First rows of operator matrix
        # diags_leftmost = [0, 1, 2, 3]
        # coeffs_leftmost = [-11, 18, -9, 2]
        # denom_leftmost = 6.
        # diags_left_1 = [-1, 0, 1, 2, 3]
        # coeffs_left_1 = [-3, -10, 18, -6, 1]
        # denom_left_1 = 12.
        # diags_left_2 = [-2, -1, 0, 1, 2, 3]
        # coeffs_left_2 = [3, -30, -20, 60, -15, 2]
        # denom_left_2 = 60.

        # # Centered diagonals
        # diags = [-3, -2, -1, 1, 2, 3]
        # coeffs = [-1, 9, -45, 45, -9, 1]
        # denom = 60.

        # # First rows of operator matrix
        # diags_leftmost = [0, 1]
        # coeffs_leftmost = [-1, 1]
        # denom_leftmost = 1.
        # diags_left_1 = [0, 1]
        # coeffs_left_1 = [-1, 1]
        # denom_left_1 = 1.
        # diags_left_2 = [0, 1]
        # coeffs_left_2 = [-1, 1]
        # denom_left_2 = 1.

        # # Centered diagonals
        # diags = [0, 1]
        # coeffs = [-1, 1]
        # denom = 1.


        # First rows of operator matrix
        diags_leftmost = [1, 2, 3]
        coeffs_leftmost = [-3, 4, -1]
        denom_leftmost = 2.
        diags_left_1 = [-1, 0, 1, 2]
        coeffs_left_1 = [-2, -3, 6, -1]
        denom_left_1 = 6.
        diags_left_2 = [-1, 0, 1, 2]
        coeffs_left_2 = [-2, -3, 6, -1]
        denom_left_2 = 6.

        # # Centered diagonals
        # diags = [-1, 0, 1, 2]
        # coeffs = [-2, -3, 6, -1]
        # denom = 6.

        


        # # First rows of operator matrix
        # diags_leftmost = [0, 1, 2, 3]
        # coeffs_leftmost = [-11, 18, -9, 2]
        # denom_leftmost = 6.
        # diags_left_1 = [-1, 0, 1, 2, 3]
        # coeffs_left_1 = [-3, -10, 18, -6, 1]
        # denom_left_1 = 12.
        # diags_left_2 = [-2, -1, 0, 1, 2, 3]
        # coeffs_left_2 = [3, -30, -20, 60, -15, 2]
        # denom_left_2 = 60.

        # # Centered diagonals
        # diags = [-2, -1, 0, 1, 2, 3]
        # coeffs = [3, -30, -20, 60, -15, 2]
        # denom = 60.


        # print diags
        # print coeffs
        # diags = [-d for d in diags[::-1]]
        # coeffs = [-d for d in coeffs[::-1]]
        # denom = denom
        # print diags
        # print coeffs
        # diags = [-3, -2, -1, 1, 2, 3]
        # coeffs = [-1, 9, -45, 45, -9, 1]
        # denom = 60.

        diags = [-1, 0, 1, 2, 3]
        coeffs = [-3, -10, 18, -6, 1]
        denom = 12.

        # diags = [0, 1, 2, 3]
        # coeffs = [-11, 18, -9, 2]
        # denom = 6.



        # diags_leftmost = [0, 1]
        # coeffs_leftmost = [-1, 1]
        # denom_leftmost = 1.
        # diags_left_1 = [0, 1]
        # coeffs_left_1 = [-1, 1]
        # denom_left_1 = 1.
        # diags_left_2 = [0, 1]
        # coeffs_left_2 = [-1, 1]
        # denom_left_2 = 1.
        # diags = [0, 1]
        # coeffs = [-1, 1]
        # denom = 1.

        # Last rows at the right of operator matrix
        diags_right_2 = [-d for d in diags_left_2[::-1]]
        coeffs_right_2 = [-d for d in coeffs_left_2[::-1]]
        denom_right_2 = denom_left_2
        diags_right_1 = [-d for d in diags_left_1[::-1]]
        coeffs_right_1 = [-d for d in coeffs_left_1[::-1]]
        denom_right_1 = denom_left_1
        diags_rightmost = [-d for d in diags_leftmost[::-1]]
        coeffs_rightmost = [-d for d in coeffs_leftmost[::-1]]
        denom_rightmost = denom_leftmost

        h = self.log_width
        dim_e = self.dim_e
        last = dim_e - 1

        op_matrix = np.zeros((dim_e, dim_e))
        op_matrix[0, np.asarray(diags_leftmost)] = np.asarray(
            coeffs_leftmost) / (denom_leftmost * h)
        op_matrix[1, 1 + np.asarray(diags_left_1)] = np.asarray(
            coeffs_left_1) / (denom_left_1 * h)
        op_matrix[2, 2 + np.asarray(diags_left_2)] = np.asarray(
            coeffs_left_2) / (denom_left_2 * h)
        op_matrix[last,
                  last + np.asarray(diags_rightmost)] = np.asarray(
                      coeffs_rightmost) / (denom_rightmost * h)
        op_matrix[last - 1, last - 1 + np.asarray(
            diags_right_1)] = np.asarray(coeffs_right_1) / (denom_right_1 * h)
        op_matrix[last - 2, last - 2 + np.asarray(
            diags_right_2)] = np.asarray(coeffs_right_2) / (denom_right_2 * h)
        for row in range(3, dim_e - 3):
            op_matrix[row, row + np.asarray(diags)] = np.asarray(coeffs) / (
                denom * h)
        # Construct an operator by left multiplication of the back-substitution
        # dlnE to dE. The right energy loss has to be later multiplied in every step
        single_op = coo_matrix(
            np.diag(1 / self.egrid).dot(op_matrix)
        )

        # construct the operator for the whole matrix, by repeating
        return block_diag(self.nspec*[single_op]).tocsr()


    def solve_coefficients(self,stencils,degree = 1):
        """Calculates the finite difference coefficients for given stencils.

        Note: The function sets up a linear equation system and solves it numerically
              Do not expect the result to be 100% accurate, as the coefficients are usually fractions

        Args:
            stencils (list of integers): position of stencils on regular grid
            degree (integer): degree of derviative, default: 1
        """
        if len(stencils) < degree + 1:
            raise Exception('Not enough stencils to solve for dervative of degree {:}, stencils given: {}'.format(degree,stencils))

        # setup of equation system
        exponents = np.arange(len(stencils))
        matrix = np.power.outer(stencils,exponents).T # pylint: disable=no-member
        right = np.zeros_like(stencils)
        right[degree] = np.math.factorial(degree)

        # solution
        return  np.linalg.solve(matrix,right)
