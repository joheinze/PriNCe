import numpy as np

from prince_cr.util import get_2Dinterp_object, get_interp_object, info, get_AZN

from .base import CrossSectionBase


class ResponseFunction(object):
    """Redistribution Function based on Crossection model

        The response function is the angular average crosssection
    """
    def __init__(self, cross_section):
        self.cross_section = cross_section

        self.xcenters = cross_section.xcenters

        # Copy indices from CrossSection Model
        self.nonel_idcs = cross_section.nonel_idcs
        self.incl_idcs = cross_section.incl_idcs
        self.incl_diff_idcs = cross_section.incl_diff_idcs

        # Dictionary of reponse function interpolators
        self.nonel_intp = {}
        self.incl_intp = {}
        self.incl_diff_intp = {}
        self.incl_diff_intp_integral = {}
        
        self._precompute_interpolators()

    # forward is_differential() to CrossSectionBase
    # that might break in the future...
    def is_differential(self, mother, daughter):
        return CrossSectionBase.is_differential(self, mother, daughter)

    def get_full(self, mother, daughter, ygrid, xgrid=None):
        """Return the full response function :math:`f(y) + g(y) + h(x,y)`
        on the grid that is provided. xgrid is ignored if `h(x,y)` not in the channel.
        """
        if xgrid is not None and ygrid.shape != xgrid.shape:
            raise Exception('ygrid and xgrid do not have the same shape!!')
        if get_AZN(mother)[0] < get_AZN(daughter)[0]:
            info(
                3,
                'WARNING: channel {:} -> {:} with daughter heavier than mother!'
                .format(mother, daughter))

        res = np.zeros(ygrid.shape)

        if (mother, daughter) in self.incl_intp:
            res += self.incl_intp[(mother, daughter)](ygrid)
        elif (mother, daughter) in self.incl_diff_intp:
            #incl_diff_res = self.incl_diff_intp[(mother, daughter)](
            #    xgrid, ygrid, grid=False)
            #if mother == 101:
            #    incl_diff_res = np.where(xgrid < 0.9, incl_diff_res, 0.)
            #res += incl_diff_res
            #if not(mother == daughter):
            res += self.incl_diff_intp[(mother, daughter)].inteval(xgrid,
                                                                   ygrid,
                                                                   grid=False)

        if mother == daughter and mother in self.nonel_intp:
            # nonel cross section leads to absorption, therefore the minus
            if xgrid is None:
                res -= self.nonel_intp[mother](ygrid)
            else:
                diagonal = xgrid == 1.
                res[diagonal] -= self.nonel_intp[mother](ygrid[diagonal])

        return res

    def get_channel(self, mother, daughter=None):
        """Reponse function :math:`f(y)` or :math:`g(y)` as
        defined in the note.

        Returns :math:`f(y)` or :math:`g(y)` if a daughter
        index is provided. If the inclusive channel has a redistribution,
        :math:`h(x,y)` will be returned

        Args:
            mother (int): mother nucleus(on)
            daughter (int, optional): daughter nucleus(on)

        Returns:
            (numpy.array) Reponse function on self._ygrid_tab
        """
        from scipy import integrate

        cs_model = self.cross_section
        egrid, cross_section = None, None

        if daughter is not None:
            if (mother, daughter) in self.incl_diff_idcs:
                egrid, cross_section = cs_model.incl_diff(mother, daughter)
            elif (mother, daughter) in self.incl_idcs:
                egrid, cross_section = cs_model.incl(mother, daughter)
            else:
                raise Exception(
                    'Unknown inclusive channel {:} -> {:} for this model'.
                    format(mother, daughter))
        else:
            egrid, cross_section = cs_model.nonel(mother)

    # note that cumtrapz works also for 2d-arrays and will integrate along axis = 1
        integral = integrate.cumtrapz(egrid * cross_section, x=egrid)
        ygrid = egrid[1:] / 2.

        return ygrid, integral / (2 * ygrid**2)

    def get_channel_scale(self, mother, daughter=None, scale='A'):
        """Returns the reponse function scaled by `scale`.

        Convenience funtion for plotting, where it is important to
        compare the cross section/response function per nucleon.

        Args:
            mother (int): Mother nucleus(on)
            scale (float): If `A` then nonel/A is returned, otherwise
                           scale can be any float.

        Returns:
            (numpy.array, numpy.array): Tuple of Energy grid in GeV,
                                        scale * inclusive cross section
                                        in :math:`cm^{-2}`
        """

        ygr, cs = self.get_channel(mother, daughter)

        if scale == 'A':
            scale = 1. / get_AZN(mother)[0]

        return ygr, scale * cs

    def _precompute_interpolators(self):
        """Interpolate each response function and store interpolators.

        Uses :func:`prince.util.get_interp_object` as interpolator.
        This might result in too many knots and can be subject to
        future optimization.
        """

        info(2, 'Computing interpolators for response functions')

        info(5, 'Nonelastic response functions f(y)')
        self.nonel_intp = {}
        for mother in self.nonel_idcs:
            self.nonel_intp[mother] = get_interp_object(
                *self.get_channel(mother))

        info(5, 'Inclusive (boost conserving) response functions g(y)')
        self.incl_intp = {}
        for mother, daughter in self.incl_idcs:
            self.incl_intp[(mother, daughter)] = get_interp_object(
                *self.get_channel(mother, daughter))

        info(5, 'Inclusive (redistributed) response functions h(y)')
        self.incl_diff_intp = {}
        for mother, daughter in self.incl_diff_idcs:
            ygr, rfunc = self.get_channel(mother, daughter)
            self.incl_diff_intp[(mother, daughter)] = get_2Dinterp_object(
                self.xcenters, ygr, rfunc, self.cross_section.xbins)

            from scipy.integrate import cumtrapz
            integral = cumtrapz(rfunc, ygr, axis=1,initial=0)
            integral = cumtrapz(integral, self.xcenters, axis=0,initial=0)

            self.incl_diff_intp_integral[(mother, daughter)] = get_2Dinterp_object(
                self.xcenters, ygr, integral, self.cross_section.xbins)