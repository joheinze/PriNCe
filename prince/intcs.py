


class PhotoNuclearInteractionRate(object):
	def __init__(self, e_grid, photon_field, *args, **kwargs):
		self.photon_field = None # object of CombinedPhotonFiled
		pass

	def get_interation_rate(self, proj_id, E, z):
		#proj_id = PDG & neucosma_codes
		# http://pdg.lbl.gov/2010/reviews/rpp2010-rev-monte-carlo-numbering.pdf

		# Nuclei CORSIKA_ID A*100 + Z, 54Fe  = 5426

		# Requirements: vectorized in E (optional in z) or result output directly on egrid

	