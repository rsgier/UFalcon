# Copyright (C) 2020 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Raphael Sgier and Jörg Herbel

import numpy as np
from scipy import integrate
import healpy as hp
from UFalcon import constants

def dimensionless_comoving_distance(z_low, z_up, cosmo, fast_mode=False):
    """
    Computes the dimensionless comoving distance between two redshifts. Scalar input only.
    :param z_low: lower redshift
    :param z_up: upper redshift, must have same shape as z_low
    :param cosmo: Astropy.Cosmo instance, controls the cosmology used
    :param fast_mode: Instead of using quad from scipy, use a simple romberg integration rule (this works here because
                      we know that the dimless com behaves and is differentiable)
    :return: dimensionless comoving distance
    """
    if fast_mode:
        integration_range, dz = np.linspace(z_low, z_up, 32+1, retstep=True)
        dimless_com = integrate.romb(cosmo.inv_efunc(integration_range), dx=dz, axis=0)
    else:
        dimless_com = integrate.quad(cosmo.inv_efunc, z_low, z_up)[0]
    return dimless_com


def comoving_distance(z_low, z_up, cosmo):
    """
    Computes the comoving distance between two redshifts. Scalar input only.
    :param z_low: lower redshift
    :param z_up: upper redshift, must have same shape as z_low
    :param cosmo: Astropy.Cosmo instance, controls the cosmology used
    :return: comoving distance
    """
    com = dimensionless_comoving_distance(z_low, z_up, cosmo) * constants.c / cosmo.H0.value
    return com


def kappa_to_gamma(kappa_map, lmax=None, use_pixel_weights=True):
    """
    Computes a gamma_1- and gamma_2-map from a kappa-map, s.t. the kappa TT-spectrum equals the gamma EE-spectrum.
    :param kappa_map: kappa map
    :param lmax: maximum multipole to consider, default: 3 * nside - 1
    :param use_pixel_weights: Use pixelweights for the map2alm transformation. This delivers the most accurate
                              transform according to healpy, but requires the pixel weights, which will be downloaded
                              automatically.
    :return: gamma_1- and gamma_2-map
    """

    nside = hp.npix2nside(kappa_map.size)

    if lmax is None:
        lmax = 3 * nside - 1

    kappa_alm = hp.map2alm(kappa_map, lmax=lmax, use_pixel_weights=use_pixel_weights)
    l = hp.Alm.getlm(lmax)[0]

    # Add the appropriate factor to the kappa_alm
    fac = np.where(np.logical_and(l != 1, l != 0), -np.sqrt(((l + 2.0) * (l - 1))/((l + 1) * l)), 0)
    kappa_alm *= fac
    t, q, u = hp.alm2map([np.zeros_like(kappa_alm), kappa_alm, np.zeros_like(kappa_alm)], nside=nside)

    return q, u
