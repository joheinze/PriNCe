import numpy as np

from prince_cr.data import PRINCE_UNITS


def get_AZN(nco_id):
    """Returns mass number :math:`A`, charge :math:`Z` and neutron
    number :math:`N` of ``nco_id``.

    Args:
        nco_id (int): corsika id of nucleus/mass group
    Returns:
        (int,int,int): (Z,A) tuple
    """
    Z, A = 1, 1

    if nco_id >= 100:
        Z = nco_id % 100
        A = (nco_id - Z) // 100
    else:
        Z, A = 0, 0

    return A, Z, A - Z


def e_nucleon(e_tot, nco_id):
    """Converts energy in energy per nucleon"""
    A, _, _ = get_AZN(nco_id)
    return e_tot / A


def get_y(e, eps, nco_id):
    """Retrns center of mass energy of nucleus-photon system.

    Args:
        e (float): energy (vector) of nucleus(on) in GeV
        eps (float): photon energy in GeV
        nco_id (int): particle index

    Returns:
        (float): center of mass energy :math:`y`
    """

    A = get_AZN(nco_id)[0]

    return e * eps / (A * PRINCE_UNITS.m_proton)


def bin_centers(bin_edges):
    """Computes and returns bin centers from given edges."""
    edg = np.array(bin_edges)
    return 0.5 * (edg[1:] + edg[:-1])


def bin_widths(bin_edges):
    """Computes and returns bin widths from given edges."""
    edg = np.array(bin_edges)

    return np.abs(edg[1:, ...] - edg[:-1, ...])


def bin_edges2D(bin_centers):
    lcen = np.log10(bin_centers)
    steps = lcen[1, ...] - lcen[0, ...]
    bins_log = np.zeros_like(lcen)  # (len(lcen) + 1)
    # print bins_log.shape
    bins_log = np.pad(bins_log, ((0, 1), (0, 0)), "constant", constant_values=0.0)
    # print bins_log.shape
    bins_log[: lcen.shape[0], ...] = lcen - 0.5 * steps
    bins_log[-1, ...] = lcen[-1, ...] + 0.5 * steps
    return 10 ** bins_log


def bin_edges1D(bin_centers):
    lcen = np.log10(bin_centers)
    steps = lcen[1] - lcen[0]
    bins_log = np.zeros(len(lcen) + 1)
    bins_log[: lcen.shape[0]] = lcen - 0.5 * steps
    bins_log[-1] = lcen[-1] + 0.5 * steps
    return 10 ** bins_log
