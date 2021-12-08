import numpy as np


class TheInterpolator(object):
    def __init__(self, bins):
        self.n_ext = 2
        self.n_window = 3
        if self.n_window % 2 == 0:
            raise Exception("Window size must be odd.")
        self.bins = bins
        self._init_grids()
        self._init_matrices()

    def _init_grids(self):
        # Bin edges extended by n_ext points on both sides
        nwi2 = (self.n_window - 1) / 2
        self.bins_ext = np.zeros(self.bins.size + 2 * self.n_ext + self.n_window - 1)
        grid_spacing = np.log10(self.bins[1] / self.bins[0])
        # Copy nominal grid
        self.bins_ext[nwi2 + self.n_ext : -nwi2 - self.n_ext] = self.bins
        for i in range(1, self.n_ext + nwi2 + 1):
            self.bins_ext[nwi2 + self.n_ext - i] = self.bins[0] * 10 ** (
                -grid_spacing * i
            )
            self.bins_ext[nwi2 + self.n_ext + self.bins.size + i - 1] = self.bins[
                -1
            ] * 10 ** (grid_spacing * i)

        self.dim = self.bins.size - 1
        self.dim_ext = self.dim + 2 * self.n_ext

        self.grid = np.sqrt(self.bins[1:] * self.bins[:-1])
        self.grid_ext = np.sqrt(self.bins_ext[1:] * self.bins_ext[:-1])

        self.widths = self.bins[1:] - self.bins[:-1]
        self.widths_ext = self.bins_ext[1:] - self.bins_ext[:-1]

        self.b = np.zeros(self.n_window * self.dim_ext)

    def _init_matrices(self):
        from scipy.sparse.linalg import factorized
        from scipy.sparse import csc_matrix

        intp_mat = np.zeros(
            (self.n_window * self.dim_ext, self.n_window * self.dim_ext)
        )
        sum_mat = np.zeros((self.dim_ext, self.n_window * self.dim_ext))

        # nex = self.n_ext
        nwi = self.n_window
        nwi2 = (self.n_window - 1) / 2
        # print self.dim_ext
        for i in range(0, self.dim_ext):
            for m in range(nwi):
                intp_mat[nwi * i + m, nwi * i : nwi * (1 + i)] = (
                    self.grid_ext[i : i + nwi] ** m * self.widths_ext[i : i + nwi]
                )

        def idx(i):
            return [
                (i - k) * nwi + k + nwi2
                for k in range(-nwi2, nwi2 + 1)
                if 0 <= ((i - k) * nwi + k + nwi2) < self.n_window * self.dim_ext
            ]

        for i in range(self.dim_ext):
            sum_mat[i, idx(i)] = 1.0

        self.intp_mat = csc_matrix(intp_mat)
        self.sum_mat = csc_matrix(sum_mat)

        self.solver = factorized(self.intp_mat)

    def set_initial_delta(self, norm, energy):
        # Setup initial state
        self.b *= 0.0
        cenbin = np.argwhere(energy < self.bins_ext)[0][0] - 1
        if cenbin < 0:
            # print 'energy too low', energy
            raise Exception()

        #       print energy, cenbin, self.bins_ext[cenbin:cenbin+2]
        norm *= self.widths_ext[cenbin]
        for m in range(self.n_window):
            self.b[self.n_window * cenbin + m] = norm * energy ** m

    def set_initial_spectrum(self, fx, fy):
        self.b *= 0.0
        for i, x in enumerate(fx):
            cenbin = np.argwhere(x < self.bins_ext)[0][0] - 1
            # print i, x, cenbin, self.bins_ext[cenbin:cenbin+2]
            if cenbin < 0:
                continue
            for m in range(0, self.n_window):
                self.b[self.n_window * cenbin + m] += fy[i] * x ** m

    def set_initial_spectrum2(self, fx, fy):
        self.b *= 0.0
        for m in range(0, self.n_window):
            self.b[
                self.n_ext * self.n_window
                + m : -self.n_ext * self.n_window
                + m : self.n_window
            ] += (fy * fx ** m)

    def get_solution(self):
        return self.sum_mat.dot(self.solver(self.b))[self.n_ext : -self.n_ext]

    def get_moment(self, m):
        nwi2 = self.n_window / 2
        return np.sum(
            (
                self.widths_ext[nwi2:-nwi2]
                * self.grid_ext[nwi2:-nwi2] ** m
                * self.sum_mat.dot(self.solver(self.b))
            )
        )
