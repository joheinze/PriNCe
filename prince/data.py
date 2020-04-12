"""Module inteded to contain some prince-specific data structures."""

from prince.util import info, get_AZN, convert_to_namedtuple
from prince_config import config, spec_data
import numpy as np


class PrinceSpecies(object):
    """Bundles different particle properties for simplified
    availability of particle properties in :class:`prince.core.PriNCeRun`.

    Args:
      pdgid (int): PDG ID of the particle
      particle_db (object): a dictionary with particle properties
      d (int): dimension of the energy grid
    """

    def __init__(self, ncoid, princeidx, d):

        info(5, 'Initializing new species', ncoid)

        #: Neucosma ID of particle
        self.ncoid = ncoid
        #: (bool) particle is a hadron (meson or baryon)
        self.is_hadron = False
        #: (bool) particle is a meson
        self.is_meson = False
        #: (bool) particle is a baryon
        self.is_baryon = False
        #: (bool) particle is a lepton
        self.is_lepton = False
        #: (bool) if it's an electromagnetic particle
        self.is_em = False
        #: (bool) particle is a lepton
        self.is_charged = False
        #: (bool) particle is a nucleus
        self.is_nucleus = False
        #: (bool) particle has an energy redistribution
        self.has_redist = False
        #: (bool) particle is stable
        self.is_stable = True
        #: (float) lifetime
        self.lifetime = np.inf
        #: (bool) particle is an alias (PDG ID encodes special scoring behavior)
        self.is_alias = False
        #: (str) species name in string representation
        self.sname = None
        #: decay channels if any
        self.decay_channels = {}
        #: Mass, charge, neutron number
        self.A, self.Z, self.N = 1, None, None
        #: Mass in atomic units or GeV
        self.mass = None

        #: (int) Prince index (in state vector)
        self.princeidx = princeidx

        # (dict) Dimension of energy grids (for idx calculations)
        self.grid_dims = {'default': d}

        # Obtain values for the attributes
        self._init_species()

    def _init_species(self):
        """Fill all class attributes with values from
        :var:`spec_data`, depending on ncoid."""

        ncoid = self.ncoid
        dbentry = spec_data[ncoid]

        if ncoid < 200:
            self.is_nucleus = False
            if ncoid == 0:
                self.is_em = True
            elif ncoid in [100, 101]:
                self.is_hadron = True
                self.is_baryon = True
                self.is_nucleus = True
                self.A, self.Z, self.N = get_AZN(ncoid)
            elif ncoid not in [2, 3, 4, 50]:
                self.is_hadron = True
                self.is_meson = True
            else:
                self.is_lepton = True
                if ncoid in [20, 21]:
                    self.is_em = True
                elif ncoid in [7, 10]:
                    self.is_alias = True
        else:
            self.is_nucleus = True
            self.A, self.Z, self.N = get_AZN(ncoid)

        if ncoid <= config["redist_threshold_ID"]:
            self.has_redist = True

        if "name" not in dbentry:
            info(5, "Name for species", ncoid, "not defined")
            self.sname = "nucleus_{0}".format(ncoid)
        else:
            self.sname = dbentry["name"]

        self.charge = dbentry["charge"]
        self.is_charged = self.charge != 0
        self.is_stable = dbentry["stable"]
        self.lifetime = dbentry["lifetime"]
        self.mass = dbentry["mass"]
        self.decay_channels = dbentry["branchings"]

    @property
    def sl(self):
        """Return the slice for this species on the grid
           can be used as spec[s.sl]

        Returns:
          (slice): a slice object pointing to the species in the state vecgtor
        """
        idx = self.princeidx
        dim = self.grid_dims['default']
        return slice(idx * dim, (idx + 1) * dim)

    def lidx(self, grid_tag='default'):
        """Returns lower index of particle range in state vector.

        Returns:
          (int): lower index in state vector :attr:`MCEqRun.phi`
        """
        return self.princeidx * self.grid_dims[grid_tag]

    def uidx(self, grid_tag='default'):
        """Returns upper index of particle range in state vector.

        Returns:
          (int): upper index in state vector :attr:`MCEqRun.phi`
        """
        return (self.princeidx + 1) * self.grid_dims[grid_tag]

    def lbin(self, grid_tag='default'):
        """Returns lower bin of particle range in state vector.

        Returns:
          (int): lower bin in state vector :attr:`MCEqRun.phi`
        """
        return self.princeidx * (self.grid_dims[grid_tag] + 1)

    def ubin(self, grid_tag='default'):
        """Returns upper bin of particle range in state vector.

        Returns:
          (int): upper bin in state vector :attr:`MCEqRun.phi`
        """
        return (self.princeidx + 1) * (self.grid_dims[grid_tag] + 1)

    def indices(self, grid_tag='default'):
        """Returns a list of all indices in the state vector.

        Returns:
          (numpy.array): array of indices in state vector :attr:`MCEqRun.phi`
        """
        idx = self.princeidx
        dim = self.grid_dims[grid_tag]
        return np.arange(idx * dim, (idx + 1) * dim)


class SpeciesManager(object):
    """Provides a database with particle and species."""

    def __init__(self, ncoid_list, ed):
        # (dict) Dimension of primary grid
        self.grid_dims = {'default': ed}
        # Particle index shortcuts
        #: (dict) Converts Neucosma ID to index in state vector
        self.ncoid2princeidx = {}
        #: (dict) Converts particle name to index in state vector
        self.sname2princeidx = {}
        #: (dict) Converts Neucosma ID to reference of
        # :class:`data.PrinceSpecies`
        self.ncoid2sref = {}
        #: (dict) Converts particle name to reference of
        #:class:`data.PrinceSpecies`
        self.sname2sref = {}
        #: (dict) Converts prince index to reference of
        #:class:`data.PrinceSpecies`
        self.princeidx2sref = {}
        #: (dict) Converts index in state vector to Neucosma ID
        self.princeidx2ncoid = {}
        #: (dict) Converts index in state vector to reference
        # of :class:`data.PrinceSpecies`
        self.princeidx2pname = {}
        #: (int) Total number of species
        self.nspec = 0

        self._gen_species(ncoid_list)
        self._init_species_tables()

    def _gen_species(self, ncoid_list):
        info(4, "Generating list of species.")

        # ncoid_list += spec_data["non_nuclear_species"]

        # Make sure list is unique and sorted
        ncoid_list = sorted(list(set(ncoid_list)))

        self.species_refs = []
        # Define position in state vector (princeidx) by simply
        # incrementing it with the (sorted) list of Neucosma IDs
        for princeidx, ncoid in enumerate(ncoid_list):
            info(
                4, "Appending species {0} at position {1}".format(
                    ncoid, princeidx))
            self.species_refs.append(
                PrinceSpecies(ncoid, princeidx, self.grid_dims['default']))

        self.known_species = [s.ncoid for s in self.species_refs]
        self.redist_species = [
            s.ncoid for s in self.species_refs if s.has_redist
        ]
        self.boost_conserv_species = [
            s.ncoid for s in self.species_refs if not s.has_redist
        ]

    def _init_species_tables(self):
        for s in self.species_refs:
            self.ncoid2princeidx[s.ncoid] = s.princeidx
            self.sname2princeidx[s.sname] = s.princeidx
            self.princeidx2ncoid[s.princeidx] = s.ncoid
            self.princeidx2pname[s.princeidx] = s.sname
            self.ncoid2sref[s.ncoid] = s
            self.princeidx2sref[s.princeidx] = s
            self.sname2sref[s.sname] = s

        self.nspec = len(self.species_refs)

    def add_grid(self, grid_tag, dimension):
        """Defines additional grid dimensions under a certain tag.

        Propagates changes to this variable to all known species.
        """
        info(2, 'New grid_tag', grid_tag, 'with dimension', dimension)
        self.grid_dims[grid_tag] = dimension

        for s in self.species_refs:
            s.grid_dims = self.grid_dims

    def __repr__(self):
        str_out = ""
        ident = 3 * ' '
        for s in self.species_refs:
            str_out += s.sname + '\n' + ident
            str_out += 'NCO id : ' + str(s.ncoid) + '\n' + ident
            str_out += 'PriNCe idx : ' + str(s.princeidx) + '\n\n'

        return str_out