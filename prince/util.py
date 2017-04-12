

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


def get_y(E, eps):
	
	return E*eps (*A?) # see internal note