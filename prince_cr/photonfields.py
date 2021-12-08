"""
Created on Feb 22, 2017

@author: Anatoli Fedynitch
"""
from abc import abstractmethod
from os.path import join

import numpy as np
from scipy.integrate import trapz
from scipy.interpolate import UnivariateSpline

import prince_cr.config as config


class PhotonField(object):
    """Base class for constructing target photon densities.

    Derived classes have to implement the method  :func:`PhotonField.get_photon_densities`.
    """

    @abstractmethod
    def get_photon_density(self, E, z):
        raise Exception("Base class method called accidentally.")


class CombinedPhotonField(PhotonField):
    """Class to combine (sum) several models, which inherit from :class:`PhotonField`.

    This class is useful when constructing a realistic photon spectrum, which
    is typically a superposition of CMB and CIB. The list of models can be passed
    to the constructor or each model can be added separately using the :func:`add_model`.

    Args:
          list_of_classes_and_args: Can be either list of classes or list of tuples (class, args)
    """

    def __init__(self, list_of_classes_and_args):

        self.model_list = []

        for arg in list_of_classes_and_args:
            cl, cl_arg = None, None
            if type(arg) not in [tuple, list]:
                cl = arg
                cl_arg = []
            else:
                cl = arg[0]
                cl_arg = arg[1:]

            self.model_list.append(cl(*cl_arg))

    def add_model(self, model_class, model_args=()):
        """Adds a model class to the combination."""
        self.model_list.append(model_class(*model_args))

    def get_photon_density(self, E, z):
        """Returns the redshift-scaled number density of photons as a
        sum of different models.

        Args:
          z (float): redshift
          E (float): photon energy (GeV)

        Returns:
          float: CMB photon spectrum in :math:`{\\rm GeV}}^{-1} {\\rm cm}}^{-3}`
        """
        E = np.atleast_1d(E)
        res = np.sum(
            [model.get_photon_density(E, z) for model in self.model_list], axis=0
        )
        return res


class FlatPhotonSpectrum(PhotonField):
    """Constant photon density for testing."""

    def get_photon_density(self, E, z):
        """Returns the redshift-scaled number density of CMB photons

        Args:
          z (float): redshift
          E (float): photon energy (GeV)

        Returns:
          float: CMB photon spectrum in :math:`{\\rm GeV}}^{-1} {\\rm cm}}^{-3}`
        """
        # density at z = 0
        nlocal = np.ones_like(E, dtype="double") * 1e12
        # nlocal = E**-1 * 1e12
        return (1.0 + z) ** 2 * nlocal


class CMBPhotonSpectrum(PhotonField):
    """Redshift-scaled number density of CMB photons

    In the CMB frame (equivalent to the observer's frame). Normalisation from Planck's spectrum.
    The scaling goes as :math:`n(E,z) = (1+z)^3 n(E/(1+z), z = 0)`.
    The CMB spectrum is a blackbody spectrum with the present-day temperature T0 = 2.725 K.

    Ref.:
        M. Ahlers, L.A. Anchordoqui, and S. Sarkar, Phys. Rev. D 79, 083009 (2009) [0902.3993]
    """

    def get_photon_density(self, E, z):
        """Returns the redshift-scaled number density of CMB photons

        Args:
          z (float): redshift
          E (float): photon energy (GeV)

        Returns:
          float: CMB photon spectrum in :math:`{\\rm GeV}}^{-1} {\\rm cm}}^{-3}`
        """
        pref = 1.31868e40  # 1/pi^2/(hbar*c)^3 [GeV^-3 cm^-3]
        Ered = E / (1.0 + z)
        # density at z = 0, for energy E / (1 + z); ECMB = kB * T0

        # Call exp only for values within dynamic range of the function
        eratio = Ered / config.E_CMB
        exp_range = eratio < 709.0  # This is the limit for 64 bits
        nlocal = np.zeros_like(Ered)
        pref *= (1.0 + z) ** 2  # Normalize
        nlocal[exp_range] = (
            pref * Ered[exp_range] ** 2 / (np.exp(eratio[exp_range]) - 1.0)
        )
        del exp_range
        return nlocal


class EBLSplined2D(PhotonField):
    def __init__(self):
        self.simple_scaling = None
        self.int2d = None

    def get_photon_density(self, E, z):
        """Returns the redshift-scaled number density of CIB photons

        Accepts scalar, vector and matrix arguments.

        Args:
          z (float): redshift
          E (float): photon energy (GeV)

        Returns:
          float: CMB photon spectrum in :math:`{\\rm GeV}}^{-1} {\\rm cm}}^{-3}`
        """
        # pylint:disable=not-callable
        if self.simple_scaling:
            Ered = E / (1.0 + z)
            nlocal = self.int2d(Ered, 0.0, assume_sorted=True)
            nz = self.int2d(Ered, z, assume_sorted=True)
            scale = trapz(nz, Ered) / trapz(nlocal, Ered) / (1 + z) ** 3
            # print(scale)
            return (1.0 + z) ** 2 * nlocal * scale
        else:
            return self.int2d(E, z, assume_sorted=True)


class CIBFranceschini2D(EBLSplined2D):
    """CIB model "1" by Fraceschini et al.

    CIB photon distribution for z = 0...2. Requires availability of
    an `scipy.interp2d` object file `data/CIB_franceschini_int2D.ppo`.

    Ref.:
        A. Franceschini et al., Astron. Astrphys. 487, 837 (2008) [arXiv:0805.1841]
    """

    def __init__(self, simple_scaling=False):
        from prince_cr.data import db_handler

        self.int2d = db_handler.ebl_spline("Francescini2008", "base")
        self.simple_scaling = simple_scaling


class CIBInoue2D(EBLSplined2D):
    """CIB model "2" by Inoue et al.

    CIB photon distribution for z = 0...10. Requires availability of
    an `scipy.interp2d` object file `data/CIB_inoue_int2D.ppo`. A low
    and high variation of the "third-population" component are also
    available, by passing

    Ref.:
        Y. Inoue et al. [arXiv:1212.1683]
    """

    def __init__(self, model="base", simple_scaling=False):
        from prince_cr.data import db_handler

        assert model in ["base", "upper", "lower"]

        self.int2d = db_handler.ebl_spline("Inoue2013", model)
        self.simple_scaling = simple_scaling


class CIBGilmore2D(EBLSplined2D):
    """CIB model "3" by Gilmore et al.

    CIB photon distribution for z = 0...7. Requires availability of
    an `scipy.interp2d` object file `data/CIB_gilmore_int2D.ppo`.

    Note: Currently uses the fixed model from the reference as standard,
          for the fiducial model, change the 'model' keyword

    Ref.:
        R.C. Gilmore et al., MNRAS Soc. 422, 3189 (2012) [arXiv:1104.0671]
    """

    def __init__(self, model="fiducial", simple_scaling=False):
        from prince_cr.data import db_handler

        assert model in ["fixed", "fiducial"]

        self.int2d = db_handler.ebl_spline("Gilmore2011", model)
        self.simple_scaling = simple_scaling


class CIBDominguez2D(EBLSplined2D):
    """CIB model "3" by Gilmore et al.

    CIB photon distribution for z = 0...2. Requires availability of
    an `scipy.interp2d` object file `data/CIB_dominguez_int2D.ppo`.

    Note: The class contains an interpolators for the upper and lower limits,
          which are not yet accessable through a function

    Ref.:
        R.C. Gilmore et al., MNRAS 410, 2556 (2011) [arXiv:1104.0671]
    """

    def __init__(self, model="base", simple_scaling=False):
        from prince_cr.data import db_handler

        assert model in ["base", "upper", "lower"]

        self.int2d = db_handler.ebl_spline("Dominguez2010", model)
        self.simple_scaling = simple_scaling


class CIBFranceschiniZ0(PhotonField):
    """CIB model "1" by Fraceschini et al.

    CIB photon distribution at z=0.

    Ref.:
        A. Franceschini et al., Astron. Astrphys. 487, 837 (2008) [arXiv:0805.1841]
    """

    def __init__(self):
        self.E = np.array(
            [
                1.00000000e-15,
                1.46217717e-12,
                2.48885732e-12,
                3.54813389e-12,
                4.97737085e-12,
                7.31139083e-12,
                8.87156012e-12,
                1.12979591e-11,
                1.38038426e-11,
                1.65958691e-11,
                1.77418948e-11,
                2.07014135e-11,
                3.10455959e-11,
                4.13999675e-11,
                4.97737085e-11,
                6.22300285e-11,
                8.27942164e-11,
                1.03609601e-10,
                1.38102010e-10,
                1.55417529e-10,
                1.85609414e-10,
                2.14387767e-10,
                2.75613191e-10,
                3.37831390e-10,
                4.17637993e-10,
                7.31307454e-10,
                9.56400953e-10,
                1.24299677e-09,
                1.55417529e-09,
                2.25995608e-09,
                3.10813590e-09,
                6.21584248e-09,
                1.00000000e01,
            ]
        )
        self.ngamma = np.array(
            [
                1.00000000e06,
                7.91224998e10,
                1.78114797e11,
                2.17470323e11,
                1.87326837e11,
                1.16654098e11,
                7.52488730e10,
                3.74541547e10,
                1.93508479e10,
                1.01765366e10,
                7.91407205e09,
                5.01880128e09,
                1.23026877e09,
                5.61047976e08,
                3.23593657e08,
                2.10377844e08,
                1.08893009e08,
                5.19517283e07,
                2.69649736e07,
                2.26725308e07,
                2.03422974e07,
                1.70529689e07,
                1.41807761e07,
                1.28617481e07,
                1.10458718e07,
                5.57057467e06,
                3.36790630e06,
                1.85152679e06,
                9.67163727e05,
                3.19815862e05,
                9.27897490e04,
                1.09774129e04,
                1.00000000e-29,
            ]
        )

        self.spl_ngamma = UnivariateSpline(self.E, self.ngamma, k=1, s=0)

    def get_photon_density(self, E, z):
        """Returns the redshift-scaled number density of CMB photons

        Args:
          z (float): redshift
          E (float): photon energy (GeV)

        Returns:
          float: CMB photon spectrum in :math:`{\\rm GeV}}^{-1} {\\rm cm}}^{-3}`
        """

        if z > 0:
            raise Exception(
                self.__class__.__name__
                + "get_photon_density(): "
                + "Redshift z > 0 not supported by this class"
            )

        return self.spl_ngamma(E, assume_sorted=True)


class CIBSteckerZ0(PhotonField):
    """CIB model "1" by Stecker et al.

    CIB photon distribution at z=0.

    Ref.:
        F.W. Stecker et al., Astrophys. J. 648, 774 (2006) [astro-ph/0510449]
    """

    def __init__(self):
        self.E = np.array(
            [
                3.30673963e-12,
                3.66859694e-12,
                4.01143722e-12,
                4.32016168e-12,
                4.72389244e-12,
                5.16416369e-12,
                5.73059874e-12,
                6.26469598e-12,
                6.95184371e-12,
                7.48858908e-12,
                8.06491882e-12,
                8.81860723e-12,
                9.49729657e-12,
                9.93116048e-12,
                1.05390145e-11,
                1.13527219e-11,
                1.25979535e-11,
                1.43979281e-11,
                1.59771753e-11,
                1.77296434e-11,
                2.08785339e-11,
                2.35125645e-11,
                2.81060620e-11,
                3.07255737e-11,
                3.46098725e-11,
                3.83972393e-11,
                4.52168183e-11,
                5.24686633e-11,
                6.55692174e-11,
                8.19407623e-11,
                9.36698878e-11,
                1.07075473e-10,
                1.24231005e-10,
                1.42010351e-10,
                1.67232237e-10,
                1.91165713e-10,
                2.05911214e-10,
                2.25117885e-10,
                2.46116086e-10,
                2.81339039e-10,
                3.21595527e-10,
                3.46409665e-10,
                3.78712814e-10,
                4.07934090e-10,
                4.45974472e-10,
                4.80374560e-10,
                5.17440002e-10,
                5.49123462e-10,
                5.82760356e-10,
                6.18457699e-10,
                6.76129679e-10,
                7.28299480e-10,
                7.96214349e-10,
                8.32530207e-10,
                9.10164726e-10,
                9.95061676e-10,
                1.10415489e-09,
                1.17179057e-09,
                1.26217624e-09,
                1.33949162e-09,
                1.42151026e-09,
                1.53119323e-09,
                1.64930130e-09,
                1.85750504e-09,
                2.12334224e-09,
                2.28712552e-09,
                2.50046051e-09,
                2.69333268e-09,
                2.85831442e-09,
                3.07878952e-09,
                3.31627090e-09,
                3.57215257e-09,
                3.79087968e-09,
                4.66766845e-09,
                5.10305207e-09,
                5.57904674e-09,
                6.19070344e-09,
                7.07668721e-09,
                8.33354913e-09,
                1.04145378e-08,
                1.15563317e-08,
                1.22642219e-08,
                1.28233058e-08,
                1.32102185e-08,
                1.40194185e-08,
                1.44444164e-08,
            ]
        )
        self.ngamma = np.array(
            [
                1.85951634e11,
                1.76116481e11,
                1.69277798e11,
                1.57180983e11,
                1.43747401e11,
                1.28262588e11,
                1.10001853e11,
                9.33899193e10,
                7.62254505e10,
                6.40766798e10,
                5.38641735e10,
                4.35211137e10,
                3.56861749e10,
                3.09100708e10,
                2.57276430e10,
                2.05825889e10,
                1.48354253e10,
                9.87802225e09,
                6.94592357e09,
                5.00691248e09,
                2.93042088e09,
                2.13348700e09,
                1.29285554e09,
                1.07082869e09,
                8.19275561e08,
                6.68574795e08,
                5.01498918e08,
                3.91399260e08,
                2.63250982e08,
                1.77063890e08,
                1.43783814e08,
                1.13899023e08,
                8.88934988e07,
                7.21855052e07,
                5.69023235e07,
                4.73685401e07,
                4.28982919e07,
                3.82771858e07,
                3.41530898e07,
                2.84308602e07,
                2.36673700e07,
                2.14338408e07,
                1.86560630e07,
                1.68958482e07,
                1.47061842e07,
                1.29921193e07,
                1.14775713e07,
                1.10868967e07,
                1.04469616e07,
                9.84396362e06,
                8.78354843e06,
                7.95463030e06,
                7.09773859e06,
                6.62186008e06,
                5.76381456e06,
                5.01683711e06,
                4.19681669e06,
                3.76313674e06,
                3.32452804e06,
                2.98098644e06,
                2.67294486e06,
                2.30350946e06,
                1.93646655e06,
                1.48156258e06,
                1.11681182e06,
                9.38858192e05,
                7.58577575e05,
                6.22085387e05,
                5.44126654e05,
                4.35281290e05,
                3.56960367e05,
                3.00088947e05,
                2.43652011e05,
                1.33045442e05,
                9.97860887e04,
                7.48393488e04,
                5.39473356e04,
                3.50404762e04,
                1.90370653e04,
                8.19200106e03,
                5.34724398e03,
                4.23525925e03,
                3.66783677e03,
                3.22403687e03,
                2.55358313e03,
                2.24460538e03,
            ]
        )

        self.spl_ngamma = UnivariateSpline(self.E, self.ngamma, k=1, s=0)

    def get_photon_density(self, E, z):
        """Returns the redshift-scaled number density of CMB photons

        Args:
          z (float): redshift
          E (float): photon energy (GeV)

        Returns:
          float: CMB photon spectrum in :math:`{\\rm GeV}}^{-1} {\\rm cm}}^{-3}`
        """

        if z > 0:
            raise Exception(
                self.__class__.__name__
                + "get_photon_density(): "
                + "Redshift z > 0 not supported by this class"
            )

        return self.spl_ngamma(E, assume_sorted=True)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    erange = np.logspace(-20, -6, 100)
    cmb = CMBPhotonSpectrum()
    inoue = CIBInoue2D()
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.loglog(
        erange, erange * cmb.get_photon_density(erange, z=0.0), ls="-", lw=2, color="k"
    )
    ax.set_ylim(1e-9, 1e3)
    #     ax.fill_between(erange, CMB_photon_spectrum(erange, z=0.),
    #                     CMB_photon_spectrum(erange, z=6.), color='b', alpha=0.3)
    ax.set_ylabel(r"$\epsilon$ d$n/$d$\epsilon$ cm$^{-3}$")
    ax.set_xlabel(r"Photon energy $\epsilon$ (GeV)")

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.loglog(
        erange,
        erange * inoue.get_photon_density(erange, z=0.0),
        ls="-",
        lw=2,
        color="k",
    )
    ax.set_ylabel(r"$\epsilon$ d$n/$d$\epsilon$ cm$^{-3}$")
    ax.set_xlabel(r"Photon energy $\epsilon$ (GeV)")

    mcomb = CombinedPhotonField([CMBPhotonSpectrum, CIBInoue2D])
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.loglog(
        erange, erange * mcomb.get_photon_density(erange, 0.0), ls="-", lw=2, color="k"
    )
    ax.fill_between(
        erange,
        erange * mcomb.get_photon_density(erange, 0.0),
        erange * mcomb.get_photon_density(erange, 6.0),
        color="r",
        alpha=0.3,
    )
    ax.set_ylabel(r"$\epsilon$ d$n/$d$\epsilon$ cm$^{-3}$")
    ax.set_xlabel(r"Photon energy $\epsilon$ (GeV)")

    plt.show()
