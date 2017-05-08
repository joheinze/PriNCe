import numpy as np
from abc import abstractmethod
from prince_config import config
from os.path import join
from scipy.interpolate import InterpolatedUnivariateSpline
# -------------------------------------------------
# NOTE (JH): 
# For now I used scipy.InterpolatedUnivariateSpline 
# as the interpolator for cross section and response values without further testing
# should this become a computation critical part, we might test other interpolators
# -------------------------------------------------


class PhotoNuclearInteractionRate(object):
    def __init__(self, photon_field, cross_section, *args, **kwargs):
        from util import EnergyGrid
        self.photon_field = photon_field # object of CombinedPhotonFiled
        self.cross_section = cross_section
        self.e_photon = EnergyGrid(-20,-5,200)

    def get_interation_rate(self, proj_id, E, z):
        # proj_id = PDG & neucosma_codes
        # http://pdg.lbl.gov/2010/reviews/rpp2010-rev-monte-carlo-numbering.pdf

        # Nuclei CORSIKA_ID A*100 + Z, 54Fe  = 5426

        # Requirements: vectorized in E (optional in z) or result output directly on egrid
        from util import get_y

        x,y = np.meshgrid(self.e_photon.grid, E)
    
        M = self.cross_section.resp_nonel[proj_id](get_y(x, y, proj_id))

        photon_vector = self.photon_field.get_photon_density(self.e_photon.grid, z)
        return M.dot(self.e_photon.widths * photon_vector)




class CrossSection(object):
    """Base class for constructing cross section models.
    
    Derived classes have to implement the method :func:`Crossection.load_data`.
    """
    def __init__(self):
        print "init on base class called"
        self._nonel, self._incl, self.incl_channels = self.load_data()

        # now also precompute the response function
        self.precomp_response_func()

    @abstractmethod
    def load_data(self):
        """Abstract method, implement this for each model.
        Should return two dictionaries containing the interpolated nonelastic/inlcusive cross section with particle id as the key

        Energy is assumed to be in GeV and cross sections in cm^2
        """
        raise Exception(self.__class__.__name__ + '::load_data():' +
                        'Base class method called accidentally.')

    # @abstractmethod
    # def get_nonel(self, energy, particle_id):
    #     raise Exception(self.__class__.__name__ + '::get_nonel():' +
    #                     'Base class method called accidentally.')

    # @abstractmethod
    # def get_incl(self, energy, tup_pids):
    #     raise Exception(self.__class__.__name__ + '::get_incl():' +
    #                     'Base class method called accidentally.')

    def get_nonel(self, energy, particle_id):
        if particle_id in self._nonel:
            return self._nonel[particle_id](energy)
        elif type(particle_id) is not int:
            raise Exception(self.__class__.__name__ + '::get_nonel():' +
                            'Method was called with invalid particle ID: {:}.'.format(particle_id))
        else:
            raise Exception(self.__class__.__name__ + '::get_nonel():' +
                            'The interaction model has no data for the particle ID: {:}.'.format(particle_id))

    def get_incl(self, energy, tup_pids):
        particle_in, particle_out = tup_pids
        
        if tup_pids in self._incl:
            return self._incl[tup_pids](energy)
        elif type(tup_pids[0]) is not int or type(tup_pids[1]) is not int :
            raise Exception(self.__class__.__name__ + '::get_incl():' +
                            'Method was called with invalid particle ID: {:}.'.format(particle_id))
        else:
            raise Exception(self.__class__.__name__ + '::get_incl():' +
                            'The interaction model has no data for the particle ID: {:}.'.format(particle_id))

    @staticmethod
    def _calc_response_func(cross_section):
        from scipy import integrate

        e = cross_section.get_knots()
        c = cross_section.get_coeffs()
        y = e / 2

        integral = integrate.cumtrapz(e * c, x = e)

        res = integral / (2 * y[1:]**2)
        return InterpolatedUnivariateSpline(y[1:], res, k = 1, ext = 'zeros')

    def precomp_response_func(self):
        self.resp_nonel = {}
        for key, cs in self._nonel.items():
            self.resp_nonel[key] = self._calc_response_func(cs)

        self.resp_incl = {}
        for key, cs in self._incl.items():
            self.resp_incl[key] = self._calc_response_func(cs)

class CrossSectionCombined(CrossSection):
    """Combination of two  Cross setion models"""

    def __init__(self, models, e_trans):
        print "init on combined class called"
        nonel1, incl1, channels1 = models[0].load_data()
        nonel2, incl2, channels2 = models[1].load_data()

        # combine the models and interpolate
        self._nonel = {}
        for key in nonel1:
            if key in nonel2:
                self._nonel[key] = self._get_combined_cs(nonel1[key], nonel2[key], e_trans)
            else:
                self._nonel[key] = nonel1[key]
        # now also check for particles, that are only the second
        for key in nonel2:
            if key not in self._nonel:
                self._nonel[key] = nonel2[key]

        self._incl = {}
        for key in incl1:
            if key in incl2:
                self._incl[key] = self._get_combined_cs(incl1[key], incl2[key], e_trans)
            else:
                self._incl[key] = incl1[key]
        # now also check for particles, that are only the second
        for key in incl2:
            if key not in self._incl:
                self._incl[key] = incl2[key]

        # now also precompute the response function
        self.precomp_response_func()

    @staticmethod
    def _get_combined_cs(csec1, csec2, e_trans,):
        egrid1 = csec1.get_knots()
        egrid2 = csec2.get_knots()
        
        cgrid1 = csec1.get_coeffs()
        cgrid2 = csec2.get_coeffs()
        
        egrid = np.concatenate([egrid1[egrid1 < e_trans], egrid2[egrid2 > e_trans]])
        cgrid = np.concatenate([cgrid1[egrid1 < e_trans], cgrid2[egrid2 > e_trans]])
        
        return InterpolatedUnivariateSpline(egrid, cgrid, k = 1, ext = 'zeros')


class CrossSectionPeanut(CrossSection):
    """ Cross sections generated using the Peanut event generator.
    Data available from 1 MeV to 1 GeV
    """

    @classmethod
    def load_data(self):
        cspath = config['data_dir']
        print "DEBUG: Data file loading now"
        # The energy grid is given in MeV, so we convert to GeV
        egrid = np.loadtxt(join(cspath, "peanut_egrid.grid"), dtype = 'float') * 1e-3
        print "DEBUG: Egrid loading finished"
        # the first columns denote particle ids for the crosssections
        pid_nonel = np.loadtxt(join(cspath, "peanut_IAS_nonel.dat"), usecols = (0,), dtype = 'int')
        pids_incl = np.loadtxt(join(cspath, "peanut_IAS_incl_i_j.dat"), usecols = (0,1), dtype = 'int')
        print "DEBUG: Pids loading finished"
        # the rest of the line denotes the crosssection on the egrid in mbarn, which is converted here to cm^2
        crosssec_nonel_raw = np.loadtxt(join(cspath, "peanut_IAS_nonel.dat"), dtype = 'float') * 1e-27
        crosssec_incl_raw = np.loadtxt(join(cspath, "peanut_IAS_incl_i_j.dat"), dtype = 'float') * 1e-27
        print "DEBUG: Data file loading finished"

        # Now write the raw data into a dict structure
        crosssec_nonel_grid = {}
        for idx, line in zip(pid_nonel, crosssec_nonel_raw):
            crosssec_nonel_grid[idx] = line[1:]
        crosssec_incl_grid = {}
        for (idx1, idx2), line in zip(pids_incl, crosssec_incl_raw):
            crosssec_incl_grid[idx1,idx2] = line[2:]

        # now interpolate these as Spline
        crosssec_nonel = {}
        for key in crosssec_nonel_grid:
            crosssec_nonel[key] = InterpolatedUnivariateSpline(egrid, crosssec_nonel_grid[key], k = 1, ext = 'zeros')
        crosssec_incl = {}
        for key in crosssec_incl_grid:
            crosssec_incl[key] = InterpolatedUnivariateSpline(egrid, crosssec_incl_grid[key], k = 1, ext = 'zeros')
        print "DEBUG: Dictionaries created"

        # construct a dict listing all the channels for the incl. cross section
        inclusive_channels = {}
        for key in crosssec_incl:
            p_id_in, p_id_out = key
            if p_id_in not in inclusive_channels:
                inclusive_channels[p_id_in] = {}
            inclusive_channels[p_id_in][p_id_out] = key

        return crosssec_nonel, crosssec_incl, inclusive_channels

class CrossSectionSophia(CrossSection):
    """ Cross sections generated using the Sophia event generator for protons and neutrons.
    Data available from 10 MeV to 10^10 GeV
    """

    @classmethod
    def load_data(self):

        # we hardcode here, for which a superposition should be created
        pids = [4115,4116,4117,4118,4119,4120,4121,4122,2107,2108,2109,2110,2111,2112,2113,6224,6225,6226,100,101,4214,4216,4218,4219,4220,4221,4222,4224,2206,2208,2209,2210,2211,
        2212,2213,2214,6323,6324,6325,6326,201,4315,4317,4318,4319,4320,4321,4322,4324,2308,2309,2310,2311,2312,2313,2314,6424,6425,6426,301,302,4416,4417,4418,4419,4420,4421,
        4422,4423,4424,2408,2410,2411,2412,2413,2414,6524,6525,6526,401,402,403,4517,4518,4519,4520,4521,4522,4523,4524,4526,2509,2510,2511,2512,2513,2514,2515,6626,502,503,4616,
        4618,4619,4620,4621,4622,4623,4624,4625,4626,2608,2609,2610,2611,2612,2613,2614,2615,6725,6726,601,602,603,604,4718,4719,4720,4721,4722,4723,4724,4725,4726,2709,2710,2711,
        2712,2713,2714,2715,2716,6826,702,703,704,705,4816,4818,4819,4820,4821,4822,4823,4824,4825,4826,2810,2811,2812,2813,2814,2815,2816,6926,802,803,804,805,806,4919,4920,4921,
        4922,4923,4924,4925,4926,2909,2910,2911,2912,2913,2914,2915,2916,2917,902,903,904,905,906,5018,5019,5020,5021,5022,5023,5024,5025,5026,3010,3011,3012,3013,3014,3015,3016,
        3017,1003,1004,1005,1006,5117,5119,5120,5121,5122,5123,5124,5125,5126,3111,3112,3114,3115,3116,3118,1103,1104,1105,1106,5219,5220,5221,5222,5223,5224,5225,5226,3210,3211,
        3212,3213,3214,3215,3216,3217,3218,1204,1205,1206,1207,1208,5319,5320,5321,5322,5323,5324,5325,5326,3311,3312,3313,3314,3315,3316,3317,3318,1304,1305,1306,1307,1308,5420,
        5422,5423,5424,5425,5426,3412,3414,3415,3416,3417,3418,3419,1404,1405,1406,1407,1408,5520,5521,5522,5523,5524,5525,5526,3512,3515,3516,3517,3518,3519,1506,1507,1508,1509,
        5620,5621,5622,5623,5624,5625,5626,3612,3614,3615,3616,3617,3618,3619,3620,1605,1606,1607,1608,1609,1610,5722,5723,5724,5725,5726,3712,3714,3716,3717,3718,3719,3720,1705,
        1707,1708,1709,1710,5822,5823,5824,5825,5826,3814,3815,3816,3817,3818,3819,3820,3822,1806,1807,1808,1809,1810,5923,5924,5925,5926,3915,3916,3917,3918,3919,3920,3921,3922,
        1908,1909,1910,6022,6024,6025,6026,4014,4015,4016,4017,4018,4019,4020,4021,4022,2006,2008,2009,2010,2011,2012,6122,6123,6124,6125,6126]
        
        cspath = config['data_dir']
        egrid, cs_proton_grid, cs_neutron_grid = np.loadtxt(join(cspath,'sophia_csec.dat'),delimiter=',', unpack = True)
        cs_proton_grid *= 1e-30
        cs_neutron_grid *= 1e-30

        # now interpolate these as Spline
        crosssec_nonel = {}
        crosssec_incl = {}

        # create superpositions for all particle ids
        for particle_id in pids:
            proton_num = particle_id % 100
            neutron_num = particle_id / 100 - proton_num

            # the nonelastiv crosssection is just a superposition of the proton/neutron number
            cgrid = proton_num * cs_proton_grid + neutron_num * cs_neutron_grid
            crosssec_nonel[particle_id] = InterpolatedUnivariateSpline(egrid, cgrid, k = 1, ext = 'zeros')
        
            # For the inclusive cross section we asume that either a proton or a neutron interacts
            cgrid = proton_num * cs_proton_grid
            crosssec_incl[particle_id, particle_id - 101] = InterpolatedUnivariateSpline(egrid, cgrid, k = 1, ext = 'zeros')
            crosssec_incl[particle_id, 101] = InterpolatedUnivariateSpline(egrid, cgrid, k = 1, ext = 'zeros')

            cgrid = neutron_num * cs_neutron_grid
            crosssec_incl[particle_id, particle_id - 100] = InterpolatedUnivariateSpline(egrid, cgrid, k = 1, ext = 'zeros')
            crosssec_incl[particle_id, 100] = InterpolatedUnivariateSpline(egrid, cgrid, k = 1, ext = 'zeros')

        # construct a dict listing all the channels for the incl. cross section
        inclusive_channels = {}
        for key in crosssec_incl:
            p_id_in, p_id_out = key
            if p_id_in not in inclusive_channels:
                inclusive_channels[p_id_in] = {}
            inclusive_channels[p_id_in][p_id_out] = key

        return crosssec_nonel, crosssec_incl, inclusive_channels