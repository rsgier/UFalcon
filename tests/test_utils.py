# Copyright (C) 2020 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Raphael Sgier and Jörg Herbel

import pytest
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import constants as const
from UFalcon import utils


@pytest.fixture()
def cosmo():
    omega_m = 0.3
    H0 = 70
    return FlatLambdaCDM(H0=H0, Om0=omega_m)

def test_comoving_distance(cosmo):
    """
    Test the computation of the comoving distance.
    """

    z_low = np.arange(0, 1, 0.1)
    z_up = z_low + np.random.uniform(low=0, high=0.2, size=z_low.size)

    for zl, zu in zip(z_low, z_up):
        com_utils = utils.comoving_distance(zl, zu, cosmo)
        com_astropy = cosmo.comoving_distance(zu).value - cosmo.comoving_distance(zl).value

        assert (com_utils - com_astropy) / com_astropy < 1e-8
