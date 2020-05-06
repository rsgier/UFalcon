# Copyright (C) 2020 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Raphael Sgier and Jörg Herbel

import pytest
import numpy as np
import PyCosmo
from UFalcon import utils


@pytest.fixture()
def cosmo():
    return PyCosmo.Cosmo()


def test_one_over_e(cosmo):
    """
    Test the function E(z).
    """
    omega_m = 0.3
    omega_l = 0.7
    cosmo.set(omega_m=omega_m, omega_l_in=omega_l)

    for z in [0, 0.5, 1]:
        assert utils.one_over_e(z, cosmo) == 1 / np.sqrt(omega_m * (1 + z) ** 3 + omega_l)


def test_comoving_distance(cosmo):
    """
    Test the computation of the comoving distance.
    """

    z_low = np.arange(0, 1, 0.1)
    z_up = z_low + np.random.uniform(low=0, high=0.2, size=z_low.size)

    for zl, zu in zip(z_low, z_up):
        com_utils = utils.comoving_distance(zl, zu, cosmo)
        com_pycosmo = cosmo.background.dist_rad_a(a=1 / (1 + zu)) - cosmo.background.dist_rad_a(a=1 / (1 + zl))
        assert (com_utils - com_pycosmo[0]) / com_pycosmo[0] < 0.01
