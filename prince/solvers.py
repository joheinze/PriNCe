"""Contains solvers, i.e. integrators, kernels, steppers, for PriNCe."""


class ExtragalacticSpace(object):
    def __init__(self, interaction_rates, ):
        self.initial_z = 6.
        self.final_z = 0.

        self.list_of_sources = []

    def int_rate(self, z):
        
        pass

    def injection(self, z):
        """This needs to return the injection rate
        at each redshift value z"""
        return np.sum([s.injection(z) for s in self.list_of_sources])

    