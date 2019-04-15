import numpy as np
from scipy import integrate


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
