def comoving_distance(z_low, z_up, cosmo):
    """
    Computes the comoving distance between two redshifts.
    :param z_low: lower redshift
    :param z_up: upper redshift, must have same shape as z_low
    :param cosmo: PyCosmo.Cosmo instance, controls the cosmology used
    :return: comoving distance, same shape as z_low
    """

    com = cosmo.background.dist_rad_a(a=1 / (1 + z_up))
    com -= cosmo.background.dist_rad_a(a=1 / (1 + z_low))

    try:
        len(z_low)
    except TypeError:
        com = com[0]

    return com


def dimensionless_comoving_distance(z_low, z_up, cosmo):
    """
    Computes the dimensionless comoving distance between two redshifts.
    :param z_low: lower redshift
    :param z_up: upper redshift, must have same shape as z_low
    :param cosmo: PyCosmo.Cosmo instance, controls the cosmology used
    :return: dimensionless comoving distance, same shape as z_low
    """
    return comoving_distance(z_low, z_up, cosmo) * cosmo.params.H0 / cosmo.params.c
