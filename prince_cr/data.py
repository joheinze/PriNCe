"""Module inteded to contain some prince-specific data structures."""
import pickle as pickle
import os.path as path
import numpy as np
import scipy.constants as spc
import h5py


from prince_cr.util import convert_to_namedtuple, info
import prince_cr.config as config

#: Dictionary containing particle properties, like mass, charge
#: lifetime or branching ratios
try:
    spec_data = pickle.load(
        open(path.join(config.data_dir, "particle_data.ppo"), "rb"))
except UnicodeDecodeError:
    spec_data = pickle.load(
        open(path.join(config.data_dir, "particle_data.ppo"), "rb"), encoding='latin1')
except FileNotFoundError:
    info(0, 'Warning, particle database "particle_data.ppo" file not found.')


# Default units in Prince are ***cm, s, GeV***
# Define here all constants and unit conversions and use
# throughout the code. Don't write c=2.99.. whatever.
# Write clearly which units a function returns.
# Convert them if not standard unit
# Accept only arguments in the units above

UNITS_AND_CONVERSIONS_DEF = dict(
    c=1e2 * spc.c,
    cm2Mpc=1. / (spc.parsec * spc.mega * 1e2),
    Mpc2cm=spc.mega * spc.parsec * 1e2,
    m_proton=spc.physical_constants['proton mass energy equivalent in MeV'][0]
    * 1e-3,
    m_electron=spc.physical_constants['electron mass energy equivalent in MeV']
    [0] * 1e-3,
    r_electron=spc.physical_constants['classical electron radius'][0] * 1e2,
    fine_structure=spc.fine_structure,
    GeV2erg=1. / 624.15,
    erg2GeV=624.15,
    km2cm=1e5,
    yr2sec=spc.year,
    Gyr2sec=spc.giga * spc.year,
    cm2sec=1e-2 / spc.c,
    sec2cm=spc.c * 1e2)

# This is the immutable unit object to be imported throughout the code
PRINCE_UNITS = convert_to_namedtuple(UNITS_AND_CONVERSIONS_DEF, "PriNCeUnits")


class PrinceDB(object):
    """Provides access to data stored in an HDF5 file.

    The file contains all tables for runnin PriNCe. Currently
    the only still required file is the particle database. The tools
    to generate this database are publicly available in
    `PriNCe-data-utils <https://github.com/joheinze/PriNCe-data-utils>`_.

    """

    def __init__(self):

        info(2, 'Opening HDF5 file', config.db_fname)
        self.prince_db_fname = path.join(config.data_dir, config.db_fname)
        if not path.isfile(self.prince_db_fname):
            raise Exception(
                'Prince DB file {0} not found in "data" directory.'.format(
                    config.db_fname))

        with h5py.File(self.prince_db_fname, 'r') as prince_db:
            self.version = (prince_db.attrs['version'])

    def _check_subgroup_exists(self, subgroup, mname):
        available_models = list(subgroup)
        if mname not in available_models:
            info(0, 'Invalid choice/model', mname)
            info(0, 'Choose from:\n', '\n'.join(available_models))
            raise Exception('Unknown selections.')

    def photo_nuclear_db(self, model_tag):
        info(10, 'Reading photo-nuclear db. tag={0}'.format(model_tag))
        db_entry = {}
        with h5py.File(self.prince_db_fname, 'r') as prince_db:
            self._check_subgroup_exists(prince_db['photo_nuclear'],
                                        model_tag)
            for entry in ['energy_grid', 'fragment_yields', 'inel_mothers',
                          'inelastic_cross_sctions', 'mothers_daughters']:
                info(10, 'Reading entry {0} from db.'.format(entry))
                db_entry[entry] = prince_db['photo_nuclear'][model_tag][entry][:]
        return db_entry

    def photo_meson_db(self, model_tag):
        info(10, 'Reading photo-nuclear db. tag={0}'.format(model_tag))
        db_entry = {}
        with h5py.File(self.prince_db_fname, 'r') as prince_db:
            self._check_subgroup_exists(prince_db['photo_nuclear'],
                                        model_tag)
            for entry in ['energy_grid', 'xbins', 'fragment_yields', 'inel_mothers',
                          'inelastic_cross_sctions', 'mothers_daughters']:
                info(10, 'Reading entry {0} from db.'.format(entry))
                db_entry[entry] = prince_db['photo_nuclear'][model_tag][entry][:]
        return db_entry

    def ebl_spline(self, model_tag, subset='base'):
        from scipy.interpolate import interp2d
        info(10, 'Reading EBL field splines. tag={0}'.format(model_tag))
        with h5py.File(self.prince_db_fname, 'r') as prince_db:
            self._check_subgroup_exists(prince_db['EBL_models'],
                                        model_tag)
            self._check_subgroup_exists(prince_db['EBL_models'][model_tag],
                                        subset)
            spl_gr = prince_db['EBL_models'][model_tag][subset]

            return interp2d(spl_gr['x'], spl_gr['y'], spl_gr['z'],
                            fill_value=0., kind='linear')

#: db_handler is the HDF file interface
db_handler = PrinceDB()

class EnergyGrid(object):
    """Class for constructing a grid for discrete distributions.

    Since we discretize everything in energy, the name seems appropriate.
    All grids are log spaced.

    Args:
        lower (float): log10 of low edge of the lowest bin
        upper (float): log10 of upper edge of the highest bin
        bins_dec (int): bins per decade of energy
    """

    def __init__(self, lower, upper, bins_dec):
        self.bins = np.logspace(lower, upper,
                                int((upper - lower) * bins_dec + 1))
        self.grid = 0.5 * (self.bins[1:] + self.bins[:-1])
        self.widths = self.bins[1:] - self.bins[:-1]
        self.d = self.grid.size
        info(
            5, 'Energy grid initialized {0:3.1e} - {1:3.1e}, {2} bins'.format(
                self.bins[0], self.bins[-1], self.grid.size))


class PrinceSpecies(object):
    """Bundles different particle properties for simplified
    availability of particle properties in :class:`prince.core.PriNCeRun`.

    Args:
      pdgid (int): PDG ID of the particle
      particle_db (object): a dictionary with particle properties
      d (int): dimension of the energy grid
    """
    @staticmethod
    def calc_AZN(nco_id):
        """Returns mass number :math:`A`, charge :math:`Z` and neutron
        number :math:`N` of ``nco_id``."""
        Z, A = 1, 1

        if nco_id >= 100:
            Z = nco_id % 100
            A = (nco_id - Z) // 100
        else:
            Z, A = 0, 0

        return A, Z, A - Z

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
                self.A, self.Z, self.N = self.calc_AZN(ncoid)
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
            self.A, self.Z, self.N = self.calc_AZN(ncoid)

        self.AZN = self.A, self.Z, self.N

        if ncoid <= config.redist_threshold_ID:
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
          (int): lower index in state vector :attr:`PrinceRun.phi`
        """
        return self.princeidx * self.grid_dims[grid_tag]

    def uidx(self, grid_tag='default'):
        """Returns upper index of particle range in state vector.

        Returns:
          (int): upper index in state vector :attr:`PrinceRun.phi`
        """
        return (self.princeidx + 1) * self.grid_dims[grid_tag]

    def lbin(self, grid_tag='default'):
        """Returns lower bin of particle range in state vector.

        Returns:
          (int): lower bin in state vector :attr:`PrinceRun.phi`
        """
        return self.princeidx * (self.grid_dims[grid_tag] + 1)

    def ubin(self, grid_tag='default'):
        """Returns upper bin of particle range in state vector.

        Returns:
          (int): upper bin in state vector :attr:`PrinceRun.phi`
        """
        return (self.princeidx + 1) * (self.grid_dims[grid_tag] + 1)

    def indices(self, grid_tag='default'):
        """Returns a list of all indices in the state vector.

        Returns:
          (numpy.array): array of indices in state vector :attr:`PrinceRun.phi`
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
