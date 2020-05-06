# Copyright (C) 2020 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Raphael Sgier and Jörg Herbel

import numpy as np
from scipy import integrate
import healpy as hp


def one_over_e(z, cosmo):
    """
    Computes the function 1 / E(z) = (Omega_m * (1 + z)^3 + Omega_Lambda)^(-1/2).
    :param z: redshift
    :param cosmo: PyCosmo.Cosmo instance, controls the cosmology used
    :return: 1 / E(z)
    """
    ez = 1 / np.sqrt(cosmo.params.omega_m * (1 + z) ** 3 + cosmo.params.omega_l)
    return ez


def dimensionless_comoving_distance(z_low, z_up, cosmo):
    """
    Computes the dimensionless comoving distance between two redshifts. Scalar input only.
    :param z_low: lower redshift
    :param z_up: upper redshift, must have same shape as z_low
    :param cosmo: PyCosmo.Cosmo instance, controls the cosmology used
    :return: dimensionless comoving distance
    """
    dimless_com = integrate.quad(one_over_e, z_low, z_up, args=(cosmo,))[0]
    return dimless_com


def comoving_distance(z_low, z_up, cosmo):
    """
    Computes the comoving distance between two redshifts. Scalar input only.
    :param z_low: lower redshift
    :param z_up: upper redshift, must have same shape as z_low
    :param cosmo: PyCosmo.Cosmo instance, controls the cosmology used
    :return: comoving distance
    """
    com = dimensionless_comoving_distance(z_low, z_up, cosmo) * cosmo.params.c / cosmo.params.H0
    return com


def kappa_to_gamma(kappa_map, lmax=None):
    """
    Computes a gamma_1- and gamma_2-map from a kappa-map, s.t. the kappa TT-spectrum equals the gamma EE-spectrum.
    :param kappa_map: kappa map
    :param lmax: maximum multipole to consider, default: 3 * nside - 1
    :return: gamma_1- and gamma_2-map
    """

    nside = hp.npix2nside(kappa_map.size)

    if lmax is None:
        lmax = 3 * nside - 1

    kappa_alm = hp.map2alm(kappa_map, lmax=lmax)
    l = hp.Alm.getlm(lmax)[0]

    # Add the appropriate factor to the kappa_alm
    fac = np.where(np.logical_and(l != 1, l != 0), np.sqrt(((l + 2.0) * (l - 1))/((l + 1) * l)), 0)
    kappa_alm *= fac
    t, q, u = hp.alm2map([np.zeros_like(kappa_alm), kappa_alm, np.zeros_like(kappa_alm)], nside=nside)

    return q, u
