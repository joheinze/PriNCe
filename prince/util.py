

m_proton = 0.9382720813 # GeV

def E_nucleon(E_tot, p_id):
	pass
	return E/A

def Z_A(self, corsika_id):
        """Returns mass number :math:`A` and charge :math:`Z` corresponding
        to ``corsika_id``

        Args:
          corsika_id (int): corsika id of nucleus/mass group
        Returns:
          (int,int): (Z,A) tuple
        """
        Z, A = 1, 1
        if corsika_id > 14:
            Z = corsika_id % 100
            A = (corsika_id - Z) / 100
        return Z, A


def get_y(E, eps, particle_id):
    A = particle_id / 100
    return E*eps /(A * m_proton)

class EnergyGrid(object):
    def __init__(self, lower, upper, bins_dec):
        import numpy as np
        self.bins = np.logspace(lower,upper,(upper-lower) * bins_dec + 1)
        self.grid = 0.5*(self.bins[1:] + self.bins[:-1])
        self.widths = self.bins[1:] - self.bins[:-1]